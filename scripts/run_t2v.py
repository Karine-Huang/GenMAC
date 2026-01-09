import subprocess
import os
import time
import tqdm


now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
name = "base_512_v2"
ckpt = 'checkpoints/base_512_v2/model.ckpt'
config = 'configs/inference_t2v_512_v2.0.yaml'

default_param = "assets/param/default_param_demo.json" 
seed = "12345678"
res_base = f"results/{seed}_{now_time}" 
res_dirs = [
    f"{res_base}/results_t2v_baseline_0",
]
prompt_files = [
    "assets/prompt/prompt.txt",
]

MAX_ITER = 5 


while True:
    try:
        for res_dir, prompt_file in zip(res_dirs, prompt_files):
            category = res_dir.split("/")[-1].split("_")[-1]
            try:
                os.makedirs(res_base, exist_ok=True)
                with open(os.path.join(res_base, f"iter_stop_{category}.txt"), "r") as f:
                    iter_stop = int(f.read())
            except FileNotFoundError:
                iter_stop = 0 
                
            for iteration in tqdm.tqdm(range(MAX_ITER)): 
                if iteration < iter_stop:
                    continue
                iter_stop = iteration

                with open(os.path.join(res_base, f"iter_stop_{category}.txt"), "w") as f:
                    f.write(str(iter_stop))
                print("iteration: ", iteration)
            
                command = [
                    "CUDA_VISIBLE_DEVICES=0", 
                    "python3", "scripts/evaluation/inference_pipeline_iter.py",
                    "--seed", seed,
                    "--mode", "base",
                    "--ckpt_path", ckpt,
                    "--config", config,
                    "--savedir", f"{res_dir}/{name}",
                    "--n_samples", "1",
                    "--bs", "1",
                    "--height", "512", 
                    "--width", "512",
                    "--unconditional_guidance_scale", "12.0",
                    "--ddim_steps", "50",
                    "--ddim_eta", "1.0",
                    "--prompt_file", prompt_file,
                    "--fps", "28",
                    "--prompt_type", "demo", 
                    "--default_param", default_param,
                    "--iter_num", int(iteration),
                    
                    "--subset_ratio", "1", 
                    "--template_version", "v0.1" if iteration == 0 else "v_redesign",
                    "--model", "gpt-4o",
                    "--max_stop_iter", MAX_ITER,
                ]

                full_command = " ".join([str(item) for item in command])

                subprocess.run(full_command, shell=True, check=True)
        break            
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}") 
        time.sleep(5)
        
            