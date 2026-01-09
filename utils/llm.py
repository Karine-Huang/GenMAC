import requests
from prompt import templates, stop, required_lines_gen, required_lines_ast_gen, required_lines_gen_prompt, required_lines_ast_gen_prompt
from prompt import required_lines as ori_required_lines
from prompt import required_lines_ast as ori_required_lines_ast
from easydict import EasyDict
from utils.cache import get_cache, add_cache
import ast
import traceback
import time
import pyjson5
import cv2
import os
import base64
import json
import re

model_names = [
    "vicuna",
    "vicuna-13b",
    "vicuna-13b-v1.3",
    "vicuna-33b-v1.3",
    "Llama-2-7b-hf",
    "Llama-2-13b-hf",
    "Llama-2-70b-hf",
    "FreeWilly2",
    "gpt-3.5-turbo",
    "gpt-3.5",
    "gpt-4",
    "gpt-4-1106-preview",
    "gpt-4o",
]

def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    print(f"Extracted {len(base64Frames)} frames")
    return base64Frames



def get_full_chat_prompt(template, prompt, suffix=None, query_prefix="Caption: ", original_bbox=None, video_path=None, template_version=None, previous_answers = None):
    if isinstance(template, str):
        full_prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": get_full_prompt(template, prompt, suffix, original_bbox).strip(),
            },
        ]
    else:
        print("**Using chat prompt**")
        assert suffix is None
        if video_path: #TODO
            if  ("v_redesign" in template_version):  # add structure overall requirements, division of roles + gen param + new prompt from structure output

                base64Frames = process_video(video_path, seconds_per_frame=1)
                role_requirements = template[0]
                role_text = template[1]
                system_content = {
                    "role": "system",
                    "content": role_requirements}
                image_content = [
                    {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}
                    for x in base64Frames
                ]
                prefix_text =[{"type": "text", "text": "These are the frames from the video."}] 
                
                content_list = prefix_text + image_content 
                
                text_list = prefix_text 

                if "alignment check phase" in role_text:
                    template_text = role_text +"\n"+"Here is the prompt:"+" \'" + prompt +"\'"
                    template_final =     {
                        "role": "user",
                        "content": template_text, 
                    }
                            
                elif "correction suggestion phase" in role_text:
                    content_list += [{"type": "text", "text": "Here is the alignment check result:"}]
                    content_list += [{"type": "text", "text": previous_answers}]  
                    
                    template_final =    {
                        "role": "user",
                        "content": role_text
                    }

                    text_list += [{"type": "text", "text": "Here is the alignment check result:"}]
                    text_list += [{"type": "text", "text": previous_answers}] 
                    

                elif "correction phase" in role_text:
                    content_list += [{"type": "text", "text": "Here is the correction suggestion:"}]
                    content_list += [{"type": "text", "text": previous_answers}]
                    
                    template_text = role_text +"\n"+"Here are the previous bounding boxes:"+" \'" + original_bbox +"\'"
                    template_final =     {
                        "role": "user",
                        "content": template_text,
                    }
                    
                    text_list += [{"type": "text", "text": "Here are the correction suggestions:"}]
                    text_list += [{"type": "text", "text": previous_answers}]  
                    
                    
                    text_list += [{"type": "text", "text": "Here are the previous bounding boxes:"}]
                    text_list += [{"type": "text", "text": original_bbox}]
                    
                full_prompt=[
                    system_content,
                    template_final, {"role": "user", "content": content_list}
                ]

      
                text_prompt = [
                    system_content,
                    template_final, {"role": "user", "content": text_list}
                ]
                
                if "structured output of correction phase" in role_text:
                    content_list = [{"type": "text", "text": "Here are the corrected bounding boxes:"}]
                    content_list += [{"type": "text", "text": previous_answers}]  
                    
                    content_list += [{"type": "text", "text": "Here is the original prompt:"}]
                    content_list += [{"type": "text", "text": prompt}]
                    
                    template_final =     {
                        "role": "user",
                        "content": role_text
                    }
                    full_prompt=[
                        system_content,
                        template_final, {"role": "user", "content": content_list}
                    ]      

                    text_list = [{"type": "text", "text": "Here are the corrected bounding boxes:"}]
                    text_list += [{"type": "text", "text": previous_answers}] 
                    
                    text_list += [{"type": "text", "text": "Here is the original prompt:"}]
                    text_list += [{"type": "text", "text": prompt}] 
                    
                    text_prompt=[
                        system_content,
                        template_final, {"role": "user", "content": text_list}
                    ]     
                
                return full_prompt, text_prompt   
     
            else: 
                base64Frames = process_video(video_path, seconds_per_frame=1)
                prompt_tmp = "Original bounding boxes:" + original_bbox + '\n' + query_prefix + prompt
                image_content = [
                    {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}
                    for x in base64Frames
                ]
                text_content = [{"type": "text", "text": prompt_tmp}]
                content_list = image_content + text_content
                
                full_prompt = [*template, 
                            {"role": "user", "content": 
                                content_list,
                    }
                    ]
                
                text_prompt= [*template, 
                            {"role": "user", "content": 
                                text_content,
                    }
                    ]
                
                return full_prompt, text_prompt

        else:
            full_prompt = [*template, {"role": "user", "content": query_prefix + prompt}]

    return full_prompt


def get_full_prompt(template, prompt, suffix=None, original_bbox=None):
    assert isinstance(template, str), "Chat template requires `get_full_chat_prompt`"
    full_prompt = template.replace("{prompt}", prompt)

    if original_bbox:
        full_prompt = full_prompt.replace("{bbox}", original_bbox)
    if suffix:
        full_prompt = full_prompt.strip() + suffix
    return full_prompt


def get_full_model_name(model):
    if model == "gpt-3.5":
        model = "gpt-3.5-turbo"
    elif model == "vicuna":
        model = "vicuna-13b"
    elif model == "gpt-4":
        model = "gpt-4"

    return model


def get_llm_kwargs(model, template_version):
    model = get_full_model_name(model)
    
    print(f"Using template: {template_version}")

    template = templates[template_version]

    if (
        "vicuna" in model.lower()
        or "llama" in model.lower()
        or "freewilly" in model.lower()
    ):
        api_base = "http://localhost:8000/v1"
        max_tokens = 900
        temperature = 0.25
        headers = {}
    else:
        from utils.api_key import api_key

        api_base = "https://api.openai.com/v1"
        max_tokens = 2000
        temperature = 0.25
        headers = {"Authorization": f"Bearer {api_key}"}

    llm_kwargs = EasyDict(
        model=model,
        template=template,
        api_base=api_base,
        max_tokens=max_tokens,
        temperature=temperature,
        headers=headers,
        stop=stop,
    )

    return model, llm_kwargs


def get_layout(prompt, llm_kwargs, suffix="", query_prefix="Caption: ", verbose=False, original_bbox=None, video_path=None, template_version=None, savedir_answer_path=None):
    # No cache in this function
    model, template, api_base, max_tokens, temperature, stop, headers = (
        llm_kwargs.model,
        llm_kwargs.template,
        llm_kwargs.api_base,
        llm_kwargs.max_tokens,
        llm_kwargs.temperature,
        llm_kwargs.stop,
        llm_kwargs.headers,
    )

    if verbose:
        print("prompt:", prompt, "with suffix", suffix)

    done = False
    attempts = 0
    while not done:
        if "gpt" in model:
            if ("v_redesign" in template_version): 

                json_dict = {}
                json_dict["prompt"] = prompt
                answer = ""
                with open (template[0], "r") as f:
                    role_requirements = f.read()
                flag = 3 
                for i in range(1, len(template)):
                    if i == 3:
                        answer_lower = answer.lower()
                        match = re.search(r'(?:correction agent|choose the suitable correction agent)[*:\s]*([A-Z](?:\d)?)', answer_lower, re.IGNORECASE)
                        if match:
                            agent = match.group(1)
                            print("The selected correction agent is:", agent)
                        if agent =="B1" or agent =="b1":
                            flag = 4
                            continue
                        
                        elif agent =="B2" or agent =="b2":
                            flag = 5
                            continue

                        else: 
                            flag = 3
                    
                    if i == 4 and (flag != 4):
                        continue  
                    
                    if i == 5 and (flag != 5):
                        continue

                    with open (template[i], "r") as f:
                        role_text = f.read()
                    
                    text_query = [role_requirements, role_text]
                    
                    
                    full_prompt, text_prompt = get_full_chat_prompt(
                            text_query, prompt, suffix, query_prefix=query_prefix,
                            original_bbox=original_bbox, video_path=video_path,
                            template_version=template_version,
                            previous_answers= answer
                        )
                    r = requests.post(
                        f"{api_base}/chat/completions",
                        json={
                            "model": model,
                            "messages": full_prompt,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                        headers=headers,
                    )
                    
                    answer = ""
                    answer += r.json()["choices"][0]["message"]["content"] + "\n"
                    print(answer)   
                    if i ==1: 
                        answer_tmp = answer.lower()
                        if "total good alignment" in answer_tmp or "good alignment in total" in answer_tmp: 
                            return "stop sign"                    

                    json_dict["Q"+str(i)] = text_prompt
                    json_dict["A"+str(i)] = answer

                savedir_answer_path_real = savedir_answer_path.split(".json")[0] + "_answers"+ ".json"
                try:
                    with open(savedir_answer_path_real, "r") as f:
                        json_answer = json.load(f)
                except:
                    json_answer = []
                json_answer.append(json_dict)
                with open(savedir_answer_path_real, "w") as f:
                    json.dump(json_answer, f, indent=4) 
             
           
            else: 
                r = requests.post(
                    f"{api_base}/chat/completions",
                    json={
                        "model": model,
                        "messages": get_full_chat_prompt(
                            template, prompt, suffix, query_prefix=query_prefix,
                            original_bbox=original_bbox, video_path=video_path,
                            template_version=template_version,
                        ),
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    headers=headers,
                )
        else:
            r = requests.post(
                f"{api_base}/completions",
                json={
                    "model": model,
                    "prompt": get_full_prompt(template, prompt, suffix).strip(),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                },
                headers=headers,
            )

        done = r.status_code == 200

        if not done:
            print(r.json())
            attempts += 1
        if attempts >= 3 and "gpt" in model:
            print("Retrying after 1 minute")
            time.sleep(60)
        if attempts >= 5 and "gpt" in model:
            print("Exiting due to many non-successful attempts")
            exit()

    if "gpt" in model:
        if verbose > 1:
            print(f"***{r.json()}***")
        response = r.json()["choices"][0]["message"]["content"]
    else:
        response = r.json()["choices"][0]["text"]

    if verbose:
        print("resp", response)

    return response


def get_parsed_layout(*args, json_template=False, **kwargs):
    if json_template:
        return get_parsed_layout_json_resp(*args, **kwargs)
    else:
        return get_parsed_layout_text_resp(*args, **kwargs)


def get_parsed_layout_text_resp(
    prompt,
    llm_kwargs=None,
    max_partial_response_retries=1,
    override_response=None,
    strip_chars=" \t\n`",
    save_leading_text=True,
    use_generation = False,
    use_new_prompt = False,
    **kwargs,
):
    """
    override_response: override the LLM response (will not query the LLM), useful for parsing existing response
    """
    if override_response is not None:
        assert (
            max_partial_response_retries == 1
        ), "override_response is specified so no partial queries are allowed"

    process_index = 0
    retries = 0
    suffix = None
    parsed_layout = {}
    reconstructed_response = ""
    required_lines = ori_required_lines
    required_lines_ast = ori_required_lines_ast
    

    if use_generation and use_new_prompt: 
        required_lines = required_lines_gen_prompt
        required_lines_ast = required_lines_ast_gen_prompt
    elif use_generation:
        required_lines = required_lines_gen
        required_lines_ast = required_lines_ast_gen
    
        
    while process_index < len(required_lines):
        retries += 1
        if retries > max_partial_response_retries:
            raise ValueError(
                f"Erroring due to many non-successful attempts on prompt: {prompt} with response {response}"
            )
        if override_response is not None:
            response = override_response
        else:
            response = get_layout(
                prompt, llm_kwargs=llm_kwargs, suffix=suffix, **kwargs
            )
            if response == "stop sign":
                return "stop sign", None
        if required_lines[process_index] in response:
            response_split = response.split(required_lines[process_index])
            
            if save_leading_text:
                reconstructed_response += (
                    response_split[0] + required_lines[process_index]
                )
            response = response_split[1]
        while process_index < len(required_lines):
            required_line = required_lines[process_index]
            next_required_line = (
                required_lines[process_index + 1]
                if process_index + 1 < len(required_lines)
                else ""
            )
            if next_required_line in response:
                if next_required_line != "":
                    required_line_idx = response.find(next_required_line)
                    line_content = response[:required_line_idx].strip(strip_chars)
                else:
                    line_content = response.strip(strip_chars)
                if required_lines_ast[process_index]:

                    line_content = line_content.split(" - ")[0].strip()

                    if line_content.startswith("-"):
                        line_content = line_content[
                            line_content.find("-") + 1 :
                        ].strip()

                    try:
                        line_content = ast.literal_eval(line_content)
                    except SyntaxError as e:
                        print(
                            f"Encountered SyntaxError with content {line_content}: {e}"
                        )
                        raise e
                parsed_layout[required_line.rstrip(":")] = line_content
                if next_required_line != "":
                    
                    reconstructed_response += response[
                        : required_line_idx + len(next_required_line)
                    ]
                else:
                    reconstructed_response += response[
                        : 
                    ]
                response = response[required_line_idx + len(next_required_line) :]
                process_index += 1
            else:
                break
        if process_index == 0:
            continue
        elif process_index < len(required_lines):
            suffix = (
                "\n"
                + response.rstrip(strip_chars)
                + "\n"
                + required_lines[process_index]
            )

    parsed_layout["Prompt"] = prompt
    return parsed_layout, reconstructed_response


def get_parsed_layout_json_resp(
    prompt,
    llm_kwargs=None,
    max_partial_response_retries=1,
    override_response=None,
    strip_chars=" \t\n`",
    save_leading_text=True,
    **kwargs,
):
    """
    override_response: override the LLM response (will not query the LLM), useful for parsing existing response
    save_leading_text: ignored since we do not allow leading text in JSON
    max_partial_response_retries: ignored since we do not allow partial response in JSON
    """
    assert (
        max_partial_response_retries == 1
    ), "no partial queries are allowed in with JSON format templates"
    if override_response is not None:
        response = override_response
    else:
        response = get_layout(prompt, llm_kwargs=llm_kwargs, suffix=None, **kwargs)

    response = response.strip(strip_chars)

    # Alternatively we can use `removeprefix` in `str`.
    response = (
        response[len("Response:") :] if response.startswith("Response:") else response
    )

    response = response.strip(strip_chars)

    # print("Response:", response)

    try:
        parsed_layout = pyjson5.loads(response)
    except (
        ValueError,
        pyjson5.Json5Exception,
        pyjson5.Json5EOF,
        pyjson5.Json5DecoderException,
        pyjson5.Json5IllegalCharacter,
    ) as e:
        print(
            f"Encountered exception in parsing the response with content {response}: {e}"
        )
        raise e

    reconstructed_response = response

    parsed_layout["Prompt"] = prompt

    return parsed_layout, reconstructed_response


def get_parsed_layout_with_cache(
    prompt,
    llm_kwargs,
    verbose=False,
    max_retries=3,
    cache_miss_allowed=True,
    json_template=False,
    **kwargs,
):
    """
    Get parsed_layout with cache support. This function will only add to cache after all lines are obtained and parsed. If partial response is obtained, the model will query with the previous partial response.
    """

    response = get_cache(prompt)

    if response is not None:
        print(f"Cache hit: {prompt}")
        parsed_layout, _ = get_parsed_layout(
            prompt,
            llm_kwargs=llm_kwargs,
            max_partial_response_retries=1,
            override_response=response,
            json_template=json_template,
        )
        return parsed_layout

    print(f"Cache miss: {prompt}")

    assert cache_miss_allowed, "Cache miss is not allowed"

    done = False
    retries = 0

    while not done:
        retries += 1
        if retries >= max_retries:
            raise ValueError(
                f"Erroring due to many non-successful attempts on prompt: {prompt}"
            )
        try:
            parsed_layout, reconstructed_response = get_parsed_layout(
                prompt, llm_kwargs=llm_kwargs, json_template=json_template, **kwargs
            )
        except Exception as e:
            print(f"Error: {e}, retrying")
            traceback.printq_exc()
            continue

        done = True

    add_cache(prompt, reconstructed_response)

    if verbose:
        print(f"parsed_layout = {parsed_layout}")

    return parsed_layout
