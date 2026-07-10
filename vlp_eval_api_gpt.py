'''
mini    nano

'''

import argparse
from openai import OpenAI
import openai  
import base64
import os, sys
import csv
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

ATTACK_TO_CSV = {
    "our_2_255":   "vlp/our_2_255/info.csv",   
    "rida_2_255":  "vlp/rida_2_255/info.csv",
    "saaet_2_255": "vlp/saaet_2_255/info.csv",
    "sga_2_255":   "vlp/sga_2_255/info.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate image-text pairs with GPT.")
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-xxxxxxx",             
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="saaet_2_255",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1-chat-latest",
    )
    return parser.parse_args()


_image_cache = {}

def image_file_to_data_url(image_path: str) -> str:
    if image_path in _image_cache:
        return _image_cache[image_path]

    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    else:
        mime = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    data_url = f"data:{mime};base64,{b64}"
    _image_cache[image_path] = data_url
    return data_url


system_prompt = (
    "You are a system that judges whether a text description matches an image.\n"
    "Analyze the image and the input text, and decide if the text description is present in the image.\n"
    "\n"
    "Matching Rules (be slightly flexible):\n"
    "- Focus on main objects and key attributes; small differences (extra colors or minor details) are acceptable.\n"
    "- If the image mostly fits the text (e.g., an orange-and-white hat for 'an orange hat'), treat it as a match; "
    "use \"[mismatch]\" only when a main object or key attribute is clearly absent or contradicted.\n"
    "\n"
    "Instructions:\n"
    "1. If the text matches the image under the above rules, reply with \"[match]\" and briefly explain why it matches.\n"
    "2. If the text does not match the image, reply with \"[mismatch]\" and briefly explain why.\n"
    "\n"
    "Output Examples:\n"
    "Example 1: \"[match] The text describes the image accurately.\"\n"
    "Example 2: \"[mismatch] The text describes a dog, but it is not present in the image.\""
)


def main():
    args = parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key 未提供，请通过 --api_key 或环境变量 OPENAI_API_KEY 提供。")

    client = OpenAI(
        api_key=api_key,
        base_url="xxxxxxx",
    )

    attack = args.attack
    if attack not in ATTACK_TO_CSV:
        raise ValueError(
            f"未知 attack '{attack}'，可选值为: {', '.join(ATTACK_TO_CSV.keys())}"
        )
    csv_path = ATTACK_TO_CSV[attack]
    model_name = args.model

    print(f"使用数据集 attack = {attack}, csv_path = {csv_path}", flush=True)
    print(f"使用模型 model = {model_name}", flush=True)

    index = 0          
    valid_total = 0    
    success = 0        

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index += 1  

            sample_start_time = time.time()

            image_path = row["image_path"].strip()
            adv_text = row["adv_text"].strip()

            image_data_url = image_file_to_data_url(image_path)

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            },
                        },
                        {
                            "type": "text",
                            "text": adv_text,
                        },
                    ],
                },
            ]

            max_retries = 5
            output_text = None
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        reasoning_effort="minimal",
                        messages=messages,
                        verbosity="low",
                    )

                    if not getattr(resp, "choices", None):
                        raise RuntimeError(f"No choices field in response: {resp}")

                    choice0 = resp.choices[0]
                    message = getattr(choice0, "message", None)
                    content = getattr(message, "content", None) if message is not None else None
                    if content is None:
                        raise RuntimeError(f"choices[0].message.content is None: {resp}")

                    output_text = content.strip()
                    break

                except openai.APIConnectionError as e:
                    last_exception = e
                    print("=" * 80, flush=True)
                    print(f"[WARN] API 连接失败 (sample #{index}, attempt {attempt}/{max_retries})", flush=True)
                    print(f"图片路径: {image_path}", flush=True)
                    print(f"攻击文本: {adv_text}", flush=True)
                    print(f"异常: {repr(e)}", flush=True)
                    print("=" * 80 + "\n", flush=True)
                    time.sleep(1.0)

                except Exception as e:
                    last_exception = e
                    print("=" * 80, flush=True)
                    print(f"[WARN] 调用异常 (sample #{index}, attempt {attempt}/{max_retries})", flush=True)
                    print(f"图片路径: {image_path}", flush=True)
                    print(f"攻击文本: {adv_text}", flush=True)
                    print(f"异常: {repr(e)}", flush=True)
                    print("=" * 80 + "\n", flush=True)
                    time.sleep(1.0)

            if output_text is None:
                sample_elapsed = time.time() - sample_start_time  
                print("=" * 80, flush=True)
                print(f"[SKIP] 样本 #{index} 最终失败，跳过，不计入统计。", flush=True)
                print(f"图片路径: {image_path}", flush=True)
                print(f"攻击文本: {adv_text}", flush=True)
                if last_exception is not None:
                    print(f"最后一次异常: {repr(last_exception)}", flush=True)
                print(f"该样本总耗时: {sample_elapsed:.3f} 秒", flush=True)
                print("=" * 80 + "\n", flush=True)
                continue

            valid_total += 1

            if "[mismatch]" in output_text:
                success += 1

            success_rate = success / valid_total if valid_total > 0 else 0.0

            sample_elapsed = time.time() - sample_start_time

            print("=" * 80, flush=True)
            print(f"样本序号: {index}", flush=True)
            print(f"当前攻击结果: {success}/{valid_total}  (success rate = {success_rate:.4f})", flush=True)
            print(f"该样本总耗时: {sample_elapsed:.3f} 秒", flush=True)
            print(f"图片路径: {image_path}", flush=True)
            print(f"攻击文本: {adv_text}", flush=True)
            print("模型输出:", flush=True)
            print(output_text, flush=True)
            print("=" * 80 + "\n", flush=True)

    print("********** 最终统计 **********", flush=True)
    print(f"总样本序号数(含失败样本): {index}", flush=True)
    print(f"有效样本数(参与成功率计算): {valid_total}", flush=True)
    print(f"攻击成功数(含 [mismatch]): {success}", flush=True)
    if valid_total > 0:
        print(f"攻击成功率: {success/valid_total:.4f}", flush=True)
    else:
        print("没有有效样本参与统计。", flush=True)



if __name__ == "__main__":
    main()
