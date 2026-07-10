'''

local

'''



import os, sys
sys.stdout.reconfigure(line_buffering=True)  
import csv
from PIL import Image
import torch
from transformers import pipeline
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)


def load_model(model_path):
    if "Qwen" in model_path:
        mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        proc = AutoProcessor.from_pretrained(model_path)
        return mdl, proc
    elif "llava" in model_path:
        print(f"Loading Llava model from {model_path}")  
        pipe = pipeline(
            "image-text-to-text",
            model=model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            top_k=1,
            temperature=0.0
        )
        if pipe is None:
            print(f"Error: Failed to load pipeline for model {model_path}.")  
        return pipe
    else:
        raise ValueError(f"Unsupported model type for path: {model_path}")



def run_inference(model, proc, img_path, adv_text):
    if isinstance(model, Qwen2_5_VLForConditionalGeneration):
        img = Image.open(img_path).convert("RGB")
        
        system_prompt = """
        You are a system that helps classify the relationship between the image and the input text. 
        Your task is to strictly analyze the given image and its accompanying input text, and determine whether the description in the text is present in the image.
       
        Instructions:
        1. If the description in the text matches the content of the image, respond with "[match]" and explain why the description matches the image.
        2. If the description does not match the content of the image, respond with "[mismatch]" and provide a brief explanation.
        
        Output Examples:
        Example 1: "[match] The text describes the image accurately."
        Example 2: "[mismatch] The text describes a dog, but it is not present in the image."
        """
        
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": img},  # This is the image to be analyzed.
                {"type": "text", "text": f"Input text describing the image: {adv_text}"}  # Clarify this is input text.
            ]},
        ]
                
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        im_i, vi_i = process_vision_info(msgs)
        inputs = proc(text=[txt], images=im_i, videos=vi_i, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outs = model.generate(**inputs, do_sample=False, max_new_tokens=1024)
        gen = outs[0][len(inputs.input_ids[0]):]
        response = proc.decode(gen, skip_special_tokens=True)

    elif model is not None:
        image = Image.open(img_path)
        
        user_prompt = f"""
        Determine if the input text describes the content of the image. 
        Please strictly analyze the image and text thoroughly and provide a match/mismatch result.

        Instructions:
        1. If the text describes the image accurately, respond with "[match]" and explain why the text matches the image.
        2. If the text does not accurately describe the image, respond with "[mismatch]" and provide an explanation. 
        
        Output Examples:
        Example 1: "[match] The text describes the image accurately."
        Example 2: "[mismatch] The text describes a dog, but it is not present in the image."

        Input Text: "{adv_text}"
        """

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]}]
        out = model(text=messages, max_new_tokens=1024)
        response = out[0]['generated_text'][1]['content'].strip().lower()

    else:
        response = "Error: pipe is None"

    return response



total_count = 0
success_count = 0
attack_success = defaultdict(int)

csv_file = "vlp/saaet_2_255/info.csv"  
model_paths = [
    "vlm/Qwen2.5-VL-3B-Instruct",  
    "vlm/Qwen2.5-VL-7B-Instruct",  
    "vlm/llava-v1.6-mistral-7b-hf"   
]

models = []
for model_path in model_paths:
    models.append(load_model(model_path))

model_success_count = {model_path: 0 for model_path in model_paths}  
model_total_count = {model_path: 0 for model_path in model_paths}  

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  
    for idx, row in enumerate(tqdm(reader, desc="Processing", unit=" Pair")):
        image_path = row[0]
        adv_text = row[1]

        print(f"\nImage: {image_path}\nText: {adv_text}\n")

        for model_info in models:
            if isinstance(model_info, tuple): 
                model, proc = model_info
                model_name = model.config._name_or_path  
                response = run_inference(model, proc, image_path, adv_text)
            else:  
                pipe = model_info
                model_name = model_paths[models.index(pipe)]  
                if pipe is None:
                    print(f"Error: pipe is None for model {model_name}. Skipping this model.")
                    continue
                system_prompt = None
                user_prompt = f"Determine if the following text describes the content of the image. If it matches, respond with [match] and explain. If not, respond with [mismatch]. Text: {adv_text}"
                response = run_inference(pipe, None, image_path, user_prompt)

            print(f"\nExplanation: {response}")

            if "[mismatch]" in response.lower():
                attack_success["mismatch"] += 1
            else:
                attack_success["match"] += 1

            model_total_count[model_name] += 1
            if "[mismatch]" in response.lower():
                model_success_count[model_name] += 1  

            success_rate = (model_success_count[model_name] / model_total_count[model_name]) * 100 if model_total_count[model_name] > 0 else 0
            print(f"ASR for {model_name}: {success_rate:.2f}%", flush=True)

        print('\n###########################################\n')

for model_path in model_paths:
    final_success_rate = (model_success_count[model_path] / model_total_count[model_path]) * 100 if model_total_count[model_path] > 0 else 0
    print(f"\n### Final Attack success rate for {model_path}: {final_success_rate:.2f}% ###\n")

print('Eval data:', csv_file)
