
import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import repeat
from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_parent_dir)

from utils.parse import show_video_boxes,size

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from funcs import batch_ddim_sampling
from utils.utils import instantiate_from_config
from utils.llm import get_full_model_name, model_names, get_parsed_layout, get_llm_kwargs
from prompt import get_prompts, template_versions, get_video_path
from utils import cache, parse
import json
from transformers import CLIPTextModel, CLIPTokenizer
import pandas as pd
import shutil


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/base_512_v2/model.ckpt", help="checkpoint path")
    parser.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml", help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=320, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=28)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    ## for conditional i2v only
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    
    # customized    
    parser.add_argument("--prompt_type", type=str, default="demo")
    parser.add_argument("--subset_ratio", type=float, default=0.1, help="if prompt-type is subset, the ratio of the subset")
    
    parser.add_argument(
        "--model",
        choices=model_names,
        default = "gpt-4-1106-preview",
        help="LLM model to load the cache from",
    )
    parser.add_argument(
        "--run-model",
        default="vc2",
        choices=[
            "vc2",
        ],
        help="The model to use for running the inference",
    )
    parser.add_argument("--template_version", choices=template_versions, default="v0.1")
    parser.add_argument("--default_param", default="assets/param/default_param_demo.json")
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--cache_backup_path", default=None, help="Backup cache path if orignial cache path does not contain prompt")
    parser.add_argument("--revised_prompt", type=bool, default=False, help="use revised prompt or not") 
    parser.add_argument("--max_stop_iter", type=int, default=5, help="max stop iteration") 
    parser.add_argument("--iter_num", type=int, default=0, help="iteration number") 
    parser.add_argument("--use_generation", action="store_true", help="use to adjust params for generation or not") # TODO
    parser.add_argument("--use_new_prompt", action="store_true", help="use to adjust prompt or not") 
    parser.add_argument("--gen_increment", type=float, default=0.05, help="use to adjust params for generation by increment") 
    parser.add_argument("--only_gen", action="store_true", help="use only generation, wo design stage") 
    
    return parser

H, W = size

def visualize_layout(parsed_layout, save_ind=None):
    condition = parse.parsed_layout_to_condition(
        parsed_layout, tokenizer=None, height=H, width=W, verbose=True
    )
    show_video_boxes(condition, ind=save_ind, save=True)

    print(f"Visualize masks at {parse.img_dir}")

def run_inference(args, gpu_num, gpu_no, cache, **kwargs):
    parent_path = os.path.join(os.path.dirname(os.path.dirname(args.savedir)), f"iter_{args.iter_num}")
    os.makedirs(parent_path, exist_ok=True)

    category = args.savedir.split("/")[-2].split("_")[-1]

    cache.cache_format = "json"
    if args.cache_path is None:
        args.cache_path = os.path.join(parent_path, f'cache/cache_{category}/cache_{args.prompt_type.replace("lmd_", "")}_{args.template_version}_{args.model}_iter_{args.iter_num}.json')
        os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)
    cache.cache_path = args.cache_path

    template_version = args.template_version

    model, llm_kwargs = get_llm_kwargs(
        model=args.model, template_version=template_version
    )
    template = llm_kwargs.template

    json_template = "json" in template_version
    
    # This is for visualizing bounding boxes
    parse.img_dir = (
        os.path.join( parent_path, f"img_generations/img_generations_{category}/imgs_{args.prompt_type}_template{template_version}_{model}_iter_{args.iter_num}")
    )
    os.makedirs(parse.img_dir, exist_ok=True)    
    
    
    print(f"Loading LLM responses from cache {cache.cache_path}")
    cache.init_cache(allow_nonexist=True)
    cache.reset_cache_access()
    
    if args.cache_backup_path is not None:
        cache_backup_cnt = 0
    
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels

    ## step 2: load data
    ## -----------------------------------------------------------------
    if isinstance(args.prompt_file, str):
        args.prompt_file = [args.prompt_file]
    prompt_list = []
    for prompt_file in args.prompt_file:
        assert os.path.exists(prompt_file), f"Error: prompt file '{prompt_file}' NOT Found!"
        prompt_list.extend(load_prompts(prompt_file, args.prompt_type, subset_ratio=args.subset_ratio))
        
    num_samples = len(prompt_list)
    filename_list = [f"{id+1:04d}" for id in range(num_samples)]

    samples_split = num_samples // gpu_num
    residual_tail = num_samples % gpu_num
    print(f'[rank:{gpu_no}] {samples_split}/{num_samples} samples loaded.')
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    if gpu_no == 0 and residual_tail != 0:
        indices = indices + list(range(num_samples-residual_tail, num_samples))
    prompt_list_rank = [prompt_list[i] for i in indices]

    if args.mode == "i2v":
        cond_inputs = get_filelist(args.cond_input, ext='[mpj][pn][4gj]')   # '[mpj][pn][4gj]'
        assert len(cond_inputs) == num_samples, f"Error: conditional input ({len(cond_inputs)}) NOT match prompt ({num_samples})!"
        filename_list = [f"{os.path.split(cond_inputs[id])[-1][:-4]}" for id in range(num_samples)]
        cond_inputs_rank = [cond_inputs[i] for i in indices]

    filename_list_rank = [filename_list[i] for i in indices]
    
    

    if args.iter_num == 0:
        if os.path.exists(f"{parent_path}/iteration_{category}.csv"):
            iter_df = pd.read_csv(f"{parent_path}/iteration_{category}.csv")
            if "correction_done" not in iter_df.columns:
                iter_df["correction_done"] = False
                iter_df.to_csv(f"{parent_path}/iteration_{category}.csv", index=False)
            
        else:
            n_rounds = len(prompt_list_rank) // args.bs
            n_rounds = n_rounds+1 if len(prompt_list_rank) % args.bs != 0 else n_rounds
            iter_df = pd.DataFrame(columns=["category","prompt_id", "prompt", "iter", "stop_iter", "correction_done"])
            for idx in range(0, n_rounds):
                if args.prompt_type == "subset": # subset, get subset_ratio of T2V-Comp prompts
                    remainder = int(idx % (100 * args.subset_ratio))
                else: 
                    remainder = int(idx % 100)
                iter_df = pd.concat([iter_df, pd.DataFrame([{ 
                    "category": int(category),
                    "prompt_id": (remainder + 1),
                    "prompt": prompt_list_rank[idx],
                    "iter": 0,
                    "stop_iter": False,
                    "correction_done": False
                }])], ignore_index=True)

            assert iter_df.shape[0] == len(prompt_list), f"Error: iter_df shape {iter_df.shape[0]} NOT equal to {len(prompt_list)}!"
            
            iter_df.to_csv(f"{parent_path}/iteration_{category}.csv", index=False)

            
    else:
        
        if os.path.exists(f"{parent_path}/iteration_{category}.csv"):
            iter_df = pd.read_csv(f"{parent_path}/iteration_{category}.csv")
            if iter_df.shape[0] != 100:
                parent_parent_path = os.path.dirname(parent_path)
                iter_df = pd.read_csv(f"{parent_parent_path}/iter_{args.iter_num-1}/iteration_{category}.csv")
                assert iter_df.shape[0] == len(prompt_list), f"Error: iter_df shape {iter_df.shape[0]} NOT equal to {len(prompt_list)}!"
                iter_df.to_csv(f"{parent_path}/iteration_{category}.csv", index=False)
        else:
            parent_parent_path = os.path.dirname(parent_path)
            iter_df = pd.read_csv(f"{parent_parent_path}/iter_{args.iter_num-1}/iteration_{category}.csv")
            assert iter_df.shape[0] == len(prompt_list), f"Error: iter_df shape {iter_df.shape[0]} NOT equal to {len(prompt_list)}!"
            iter_df.to_csv(f"{parent_path}/iteration_{category}.csv", index=False)
    

        
    
    
    max_attempts = 5

    ## step 3: run over samples
    ## -----------------------------------------------------------------
    start = time.time()
    n_rounds = len(prompt_list_rank) // args.bs
    n_rounds = n_rounds+1 if len(prompt_list_rank) % args.bs != 0 else n_rounds
    for idx in range(0, n_rounds):
        if args.prompt_type == "subset": # subset, get subset_ratio of T2V-Comp prompts
            index = int(idx // (100 * args.subset_ratio))
            remainder = int(idx % (100 * args.subset_ratio))
        else: 
            index = int(idx // 100)
            remainder = int(idx % 100)
        save_ind = int(category) * 100 + remainder +1

        if os.path.exists(f"{parent_path}/video/results_t2v_baseline_{category}/base_512_v2/{str(remainder + 1).zfill(4)}.mp4"):
            print("category: ", category, "prompt_id: ", (remainder+1), " already exists, skip!")
            continue
        

        old_iter = iter_df.loc[(iter_df['category'] == int(category)) & (iter_df['prompt_id'] == (remainder+1))]

        if old_iter["stop_iter"].bool():
            print("category: ", category, "prompt_id: ", (remainder+1), f" has been stopped for iteration {old_iter['iter']}!")
            print("copy from last generation") 
            parent_parent_path = os.path.dirname(parent_path)
            os.makedirs(f"{parent_parent_path}/iter_{args.iter_num}/video/results_t2v_baseline_{category}/base_512_v2", exist_ok=True)
            shutil.copy(f"{parent_parent_path}/iter_{args.iter_num-1}/video/results_t2v_baseline_{category}/base_512_v2/{str(remainder + 1).zfill(4)}.mp4", f"{parent_parent_path}/iter_{args.iter_num}/video/results_t2v_baseline_{category}/base_512_v2/{str(remainder + 1).zfill(4)}.mp4")
            time.sleep(2)
            continue
        else:
            iter_df.loc[(iter_df['category'] == int(category)) & (iter_df['prompt_id'] == (remainder+1)), "iter"] = args.iter_num
            if args.iter_num == args.max_stop_iter:
                iter_df.loc[(iter_df['category'] == int(category)) & (iter_df['prompt_id'] == (remainder+1)), "stop_iter"] = True
                
            assert iter_df.shape[0] == len(prompt_list), f"Error: iter_df shape {iter_df.shape[0]} NOT equal to {len(prompt_list)}!"
            iter_df.to_csv(f"{parent_path}/iteration_{category}.csv", index=False)
            old_iter = iter_df.loc[(iter_df['category'] == int(category)) & (iter_df['prompt_id'] == (remainder+1))]

        if old_iter["iter"].item() == args.max_stop_iter:
            print("category: ", category, "prompt_id: ", (remainder+1), f" has been stopped for iteration {old_iter['iter']}!")
            print("copy from last generation") 
            parent_parent_path = os.path.dirname(parent_path)
            os.makedirs(f"{parent_parent_path}/iter_{args.iter_num}/video/results_t2v_baseline_{category}/base_512_v2", exist_ok=True)
            shutil.copy(f"{parent_parent_path}/iter_{args.iter_num-1}/video/results_t2v_baseline_{category}/base_512_v2/{str(remainder + 1).zfill(4)}.mp4", f"{parent_parent_path}/iter_{args.iter_num}/video/results_t2v_baseline_{category}/base_512_v2/{str(remainder + 1).zfill(4)}.mp4")
            time.sleep(2)            
            continue 
        elif old_iter["iter"].item() > 0 :
            parent_parent_path = os.path.dirname(parent_path)
            try: 
                if args.iter_num == 1:
                    template_version_last = "v0.1" #TODO
                else:
                    template_version_last = args.template_version
                if not (args.iter_num == 1 and args.only_gen): 
                    with open(f"{parent_parent_path}/iter_{args.iter_num-1}/cache/cache_{category}/cache_{args.prompt_type.replace('lmd_', '')}_{template_version_last}_{args.model}_iter_{args.iter_num-1}.json", "r") as f:
                        original_bboxes = json.load(f)
            except:
                if args.iter_num == 1:
                    template_version_last = "vdemo"
                else:
                    template_version_last = args.template_version
                if not (args.iter_num == 1 and args.only_gen): 
                    with open(f"{parent_parent_path}/iter_{args.iter_num-1}/cache/cache_{category}/cache_{args.prompt_type.replace('lmd_', '')}_{template_version_last}_{args.model}_iter_{args.iter_num-1}.json", "r") as f:
                        original_bboxes = json.load(f)
                

            original_video_path = f"{parent_parent_path}/iter_{args.iter_num-1}/video"
            video_paths = get_video_path(original_video_path, args.prompt_type)
        
        print(f'[rank:{gpu_no}] batch-{idx+1} ({args.bs})x{args.n_samples} ...')
        idx_s = idx*args.bs
        idx_e = min(idx_s+args.bs, len(prompt_list_rank))
        batch_size = idx_e - idx_s
        filenames = filename_list_rank[idx_s:idx_e]
        noise_shape = [batch_size, channels, frames, h, w]
        fps = torch.tensor([args.fps]*batch_size).to(model.device).long()
        prompts = prompt_list_rank[idx_s:idx_e]
        if isinstance(prompts, list):
            prompt = prompts[0]
        else:
            prompt = prompts


        original_bbox = None
        video_path = None
        if old_iter["iter"].item() > 0:
            prompt = prompt.strip().rstrip(".")

        prompt_cache = prompt.strip().rstrip(".")
        resp = cache.get_cache(prompt_cache)


        if resp is None:
            print(f"Cache miss, skipping prompt: {prompt}")

        try: 
            if old_iter["iter"].item() > 0:
                if not (args.iter_num == 1 and args.only_gen): 
                    original_bbox = original_bboxes[prompt][0]

                video_path = video_paths[category][int(remainder)]
            
            attempts = 0
            while True:
                attempts += 1
                try:
                    parsed_layout, resp_new = get_parsed_layout(
                        prompt,
                        llm_kwargs=llm_kwargs,
                        json_template=json_template,
                        verbose=False,
                        video_path = video_path,
                        original_bbox = original_bbox,
                        template_version = template_version,
                        savedir_answer_path = cache.cache_path,
                        override_response=resp,
                        use_generation=args.use_generation,
                        use_new_prompt = args.use_new_prompt,
                        
                    )
                except (ValueError, SyntaxError, TypeError) as e:
                    if attempts > max_attempts:
                        print("Retrying too many times, skipping")
                        break
                    print(
                        f"Encountered invalid data with prompt {prompt} and response {resp}: {e}, retrying"
                    )
                    print ("copy from last generation", prompt)
                    parent_parent_path = os.path.dirname(parent_path)
                    
                    os.makedirs(f"{parent_parent_path}/iter_{args.iter_num}/video/results_t2v_baseline_{category}/base_512_v2", exist_ok=True)
                    shutil.copy(f"{parent_parent_path}/iter_{args.iter_num-1}/video/results_t2v_baseline_{category}/base_512_v2/{str(remainder + 1).zfill(4)}.mp4", f"{parent_parent_path}/iter_{args.iter_num}/video/results_t2v_baseline_{category}/base_512_v2/{str(remainder + 1).zfill(4)}.mp4")
                    
                    time.sleep(2)
                    continue
                break

            if resp is None and resp_new is not None:
                cache.add_cache(prompt_cache, resp_new)
                visualize_layout(parsed_layout, save_ind)
                resp = resp_new    
            
            try:      
                if parsed_layout =="stop sign":
                    print("stop sign for prompt: ", prompt)
                    parent_parent_path = os.path.dirname(parent_path)
                    os.makedirs(f"{parent_parent_path}/iter_{args.iter_num}/video/results_t2v_baseline_{category}/base_512_v2", exist_ok=True)
                    shutil.copy(f"{parent_parent_path}/iter_{args.iter_num-1}/video/results_t2v_baseline_{category}/base_512_v2/{str(remainder + 1).zfill(4)}.mp4", f"{parent_parent_path}/iter_{args.iter_num}/video/results_t2v_baseline_{category}/base_512_v2/{str(remainder + 1).zfill(4)}.mp4")
                    time.sleep(2)

                    iter_df.loc[(iter_df['category'] == int(category)) & (iter_df['prompt_id'] == (remainder+1)), "stop_iter"] = True
                    iter_df.loc[(iter_df['category'] == int(category)) & (iter_df['prompt_id'] == (remainder+1)), "correction_done"] = True
                    assert iter_df.shape[0] == len(prompt_list), f"Error: iter_df shape {iter_df.shape[0]} NOT equal to {len(prompt_list)}!"
                    iter_df.to_csv(f"{parent_path}/iteration_{category}.csv", index=False)
                    continue
            except:
                continue

            
            print("parsed_layout:", parsed_layout)

            parsed_layout["Prompt"] = parsed_layout["Prompt"].lower()
            if args.use_new_prompt:
                parsed_layout["New prompt"] = parsed_layout["New prompt"].lower()

            tokenizer = CLIPTokenizer.from_pretrained("checkpoints/tokenizer")
            condition = parse.parsed_layout_to_condition(
                parsed_layout,
                tokenizer=tokenizer,
                height=args.height,
                width=args.width,
                num_condition_frames=frames, 
                verbose=True,
                use_generation=args.use_generation,
                use_new_prompt = args.use_new_prompt,
            )

            prompts = parsed_layout["Prompt"]
            print("original prompts: ", prompts)

            if args.use_generation and args.use_new_prompt:
                bboxes, phrases, object_positions, token_map, generation_param_dic, new_prompt = (
                    condition.boxes,
                    condition.phrases,
                    condition.object_positions,
                    condition.token_map,
                    condition.generation_param_dic,
                    condition.new_prompt,
                )  

                new_prompt = new_prompt.lower().rstrip(".")
                prompts = new_prompt
                print("new prompts: ", prompts)
                
            elif args.use_generation:
                bboxes, phrases, object_positions, token_map, generation_param_dic = (
                    condition.boxes,
                    condition.phrases,
                    condition.object_positions,
                    condition.token_map,
                    condition.generation_param_dic,
                )    
            else:           
                bboxes, phrases, object_positions, token_map = (
                    condition.boxes,
                    condition.phrases,
                    condition.object_positions,
                    condition.token_map,
                )
            

            if args.use_generation and args.iter_num > 0:
                save_name = filenames[0]
                if not os.path.exists(f"{parent_parent_path}/iter_{args.iter_num-1}/gen/gen_{category}/{save_name}.json"):
                    with open(args.default_param, "r") as f:
                        params = json.load(f)
                    os.makedirs(f"{parent_parent_path}/iter_{args.iter_num-1}/gen/gen_{category}", exist_ok=True)
                    with open(f"{parent_parent_path}/iter_{args.iter_num-1}/gen/gen_{category}/{save_name}.json", "w") as f:
                        json.dump(params, f, indent = 4)
                else:
                    with open(f"{parent_parent_path}/iter_{args.iter_num-1}/gen/gen_{category}/{save_name}.json", "r") as f: # open previous param.json
                        params = json.load(f)
                if isinstance(params["fg_weight"], float):
                    params["fg_weight"] = [params["fg_weight"]]*len(object_positions)
                    
                if isinstance(params["fg_weight"], list):
                    if len(params["fg_weight"]) == len(object_positions):
                        
                        for k in range(len(generation_param_dic["increase"])):
                            value = generation_param_dic["increase"][k]
                            params["fg_weight"][value] += args.gen_increment #TODO
                        for k in range(len(generation_param_dic["decrease"])):
                            value = generation_param_dic["decrease"][k]
                            params["fg_weight"][value] -= args.gen_increment 
                    else:
                        params["fg_weight"] = [1.0]*len(object_positions)
                os.makedirs(f"{parent_path}/gen/gen_{category}", exist_ok=True)
                with open(f"{parent_path}/gen/gen_{category}/{save_name}.json", "w") as f:
                    json.dump(params, f, indent = 4)


            else:
                with open(args.default_param, "r") as f:
                    params = json.load(f)
        
                if args.use_generation: 
                    save_name = filenames[0]
                    os.makedirs(f"{parent_path}/gen/gen_{category}", exist_ok=True)
                    with open(f"{parent_path}/gen/gen_{category}/{save_name}.json", "w") as f:
                        json.dump(params, f, indent=4)
            backward_guidance_kwargs = dict(
                bboxes=bboxes,
                object_positions=object_positions,
                loss_scale=params['loss_scale'],
                loss_threshold=params['loss_threshold'],
                max_iter=params['max_iter'],
                max_index_step=params['max_index_step'],
                fg_top_p=params['fg_top_p'],
                bg_top_p=params['bg_top_p'],
                fg_weight=params['fg_weight'],
                bg_weight=params['bg_weight'],
                use_ratio_based_loss=params['use_ratio_based_loss'],
                guidance_attn_keys=params['guidance_attn_keys'],
                exclude_bg_heads=params['exclude_bg_heads'],
                upsample_scale=params['upsample_scale'],
                upsample_mode=params['upsample_mode'],
                base_attn_dim=params['base_attn_dim'],
                attn_sync_weight=params['attn_sync_weight'],
                boxdiff_loss_scale=params['boxdiff_loss_scale'],
                boxdiff_normed=params['boxdiff_normed'],
                com_loss_scale=params['com_loss_scale'],
                verbose=params['verbose'],
            )
            kwargs.update({"backward_guidance_kwargs": backward_guidance_kwargs})
            kwargs.update({"prompt_type": args.prompt_type})
        
            
            
            text_emb = model.get_learned_conditioning(prompts)

            if args.mode == 'base':
                cond = {"c_crossattn": [text_emb], "fps": fps}
            elif args.mode == 'i2v':
                cond_images = load_image_batch(cond_inputs_rank[idx_s:idx_e], (args.height, args.width))
                cond_images = cond_images.to(model.device)
                img_emb = model.get_image_embeds(cond_images)
                imtext_cond = torch.cat([text_emb, img_emb], dim=1)
                cond = {"c_crossattn": [imtext_cond], "fps": fps}
            else:
                raise NotImplementedError


            try:
                batch_samples = batch_ddim_sampling(model, cond, noise_shape, args.n_samples, \
                                                        args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, **kwargs)
            except:
                continue

            output_path = os.path.join(parent_path,"video", f"results_t2v_baseline_{category}", f"base_512_v2")
            os.makedirs(output_path, exist_ok=True)
            save_videos(batch_samples, output_path, filenames, fps=args.savefps)
            
        except: 
            
            if args.iter_num > 0:
                # copy from last generation
                print("copy from last generation: ", prompt)
                parent_parent_path = os.path.dirname(parent_path)

                try:
                    os.makedirs(f"{parent_parent_path}/iter_{args.iter_num}/video/results_t2v_baseline_{category}/base_512_v2", exist_ok=True)
                    shutil.copy(f"{parent_parent_path}/iter_{args.iter_num-1}/video/results_t2v_baseline_{category}/base_512_v2/{str(remainder + 1).zfill(4)}.mp4", f"{parent_parent_path}/iter_{args.iter_num}/video/results_t2v_baseline_{category}/base_512_v2/{str(remainder + 1).zfill(4)}.mp4")
                    time.sleep(2)
                except:
                    continue
                continue            
        
            continue
         
        
    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")
    
    # output backup cache count to txt
    if args.cache_backup_path is not None:
        save_backup_path = os.path.dirname(args.savedir)
        with open(f"{save_backup_path}/backup_cache_count.txt", "a") as f:
            f.write(f"Backup cache count: {cache_backup_cnt} for prompt file {args.prompt_file}\n")  



if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1

    prompt_file = args.prompt_file       

    run_inference(args, gpu_num, rank, cache)
