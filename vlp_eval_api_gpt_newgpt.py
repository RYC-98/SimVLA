# -*- coding: utf-8 -*-
"""
 GPT-5.4 / GPT-5.5 
"""

import argparse
import base64
import csv
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import OpenAI


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


ATTACK_TO_CSV = {
    "our_2_255": "vlp/our_2_255/info.csv",
    "rida_2_255": "vlp/rida_2_255/info.csv",
    "saaet_2_255": "vlp/saaet_2_255/info.csv",
    "sga_2_255": "vlp/sga_2_255/info.csv",
}


SYSTEM_PROMPT = (
    "You are a system that judges whether a text description matches an image.\n"
    "Analyze the image and the input text, and decide if the text description is present in the image.\n"
    "\n"
    "Matching Rules (be slightly flexible):\n"
    "- Focus on main objects and key attributes; small differences (extra colors or minor details) are acceptable.\n"
    "- If the image mostly fits the text (e.g., an orange-and-white hat for 'an orange hat'), treat it as a match; "
    "use \"[mismatch]\" only when a main object or key attribute is clearly absent or contradicted.\n"
    "\n"
    "Instructions:\n"
    "1. If the text matches, reply with \"[match]\" followed by a brief explanation.\n"
    "2. If the text does not match, reply with \"[mismatch]\" followed by a brief explanation.\n"
    "3. Your response MUST begin with exactly \"[match]\" or \"[mismatch]\". "
    "Never output an image description without the label.\n"
    "\n"
    "Output Examples:\n"
    "Example 1: \"[mismatch] The text describes a dog, but it is not present in the image.\""
    "Example 2: \"[match] The text describes the image accurately.\"\n"
    "Example 3: \"[mismatch] The text describes a red car, but the car in the image is clearly blue.\""
)


FORMAT_RETRY_SUFFIX = (
    "\n\nStrict format requirement: Your previous response omitted or misplaced "
    "the required label. This response MUST start with exactly [match] or "
    "[mismatch], followed by one brief explanation. Do not output a standalone caption."
)


class OutputFormatError(RuntimeError):


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluate image-text pairs with GPT-5.4 or GPT-5.5."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-***********************",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="saaet_2_255",
        choices=tuple(ATTACK_TO_CSV.keys()),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.5",
        choices=("gpt-5.4", "gpt-5.5"),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--retry_interval",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


_image_cache: Dict[str, str] = {}


def image_file_to_data_url(image_path: str) -> str:

    if image_path in _image_cache:
        return _image_cache[image_path]

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()

    if ext in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"

    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    data_url = f"data:{mime};base64,{image_base64}"
    _image_cache[image_path] = data_url
    return data_url


def get_field(obj: Any, field_name: str, default: Any = None) -> Any:

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(field_name, default)

    return getattr(obj, field_name, default)


def extract_api_error(response: Any) -> Optional[str]:


    response_error = get_field(response, "error")

    if not response_error:
        return None

    error_code = get_field(response_error, "code", "unknown_error")
    error_type = get_field(response_error, "type", "unknown_type")
    error_message = get_field(
        response_error,
        "message",
        str(response_error),
    )

    return (
        f"code={error_code}, "
        f"type={error_type}, "
        f"message={error_message}"
    )


def extract_output_text(response: Any) -> str:

    response_error = extract_api_error(response)

    if response_error:
        raise RuntimeError(f"API 返回错误: {response_error}")

    choices = get_field(response, "choices")

    if not choices:
        raise RuntimeError(f"API 响应中没有 choices: {response}")

    first_choice = choices[0]
    message = get_field(first_choice, "message")

    if message is None:
        raise RuntimeError(
            f"choices[0] 中没有 message: {response}"
        )

    content = get_field(message, "content")

    if content is None:
        raise RuntimeError(
            f"choices[0].message.content is None: {response}"
        )

    output_text = str(content).strip()

    if not output_text:
        raise RuntimeError("模型返回了空文本")

    return output_text


def validate_output_format(output_text: str) -> str:

    normalized_text = output_text.strip().lower()

    match_result = re.match(
        r"^\[(match|mismatch)\]\s*(.+)$",
        normalized_text,
        flags=re.DOTALL,
    )

    if match_result is None:
        raise OutputFormatError(
            "模型输出必须以 [match] 或 [mismatch] 开头，"
            f"并在标签后给出解释。实际输出: {output_text}"
        )

    labels = re.findall(
        r"\[(match|mismatch)\]",
        normalized_text,
    )

    if len(labels) != 1:
        raise OutputFormatError(
            "模型输出必须且只能包含一个分类标签。"
            f"实际输出: {output_text}"
        )

    label = match_result.group(1)
    explanation = match_result.group(2).strip()

    if not explanation:
        raise OutputFormatError(
            "分类标签后必须包含简短解释。"
            f"实际输出: {output_text}"
        )

    return label


def is_non_retryable_error(exception: Exception) -> bool:
    """识别重试也无法解决的参数、模型或权限错误。"""

    error_text = str(exception).lower()

    keywords = (
        "unsupported value",
        "unsupported parameter",
        "unknown parameter",
        "invalid parameter",
        "invalid model",
        "model not found",
        "does not support",
        "authentication",
        "invalid api key",
        "incorrect api key",
        "permission denied",
        "not permitted",
    )

    return any(keyword in error_text for keyword in keywords)


def print_api_error(
    *,
    index: int,
    attempt: int,
    max_retries: int,
    image_path: str,
    adv_text: str,
    exception: Exception,
    error_title: str,
) -> None:
    print("=" * 80, flush=True)
    print(
        f"[WARN] {error_title} "
        f"(sample #{index}, API attempt {attempt}/{max_retries})",
        flush=True,
    )
    print(f"图片路径: {image_path}", flush=True)
    print(f"攻击文本: {adv_text}", flush=True)
    print(f"异常: {repr(exception)}", flush=True)
    print("=" * 80 + "\n", flush=True)


def build_messages(
    image_data_url: str,
    adv_text: str,
    strengthen_format: bool,
) -> List[Dict[str, Any]]:

    system_prompt = SYSTEM_PROMPT

    if strengthen_format:
        system_prompt += FORMAT_RETRY_SUFFIX

    return [
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
                        "url": image_data_url,
                    },
                },
                {
                    "type": "text",
                    "text": adv_text,
                },
            ],
        },
    ]


def request_with_api_retries(
    *,
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, Any]],
    max_retries: int,
    retry_interval: float,
    index: int,
    image_path: str,
    adv_text: str,
) -> Tuple[Optional[str], Optional[Exception]]:

    last_exception: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=model_name,
                reasoning_effort="none",
            )

            return extract_output_text(response), None

        except (
            openai.APIConnectionError,
            openai.APITimeoutError,
        ) as e:
            last_exception = e

            print_api_error(
                index=index,
                attempt=attempt,
                max_retries=max_retries,
                image_path=image_path,
                adv_text=adv_text,
                exception=e,
                error_title="API 连接失败",
            )

        except openai.RateLimitError as e:
            last_exception = e

            print_api_error(
                index=index,
                attempt=attempt,
                max_retries=max_retries,
                image_path=image_path,
                adv_text=adv_text,
                exception=e,
                error_title="API 限流",
            )

        except openai.APIStatusError as e:
            last_exception = e

            print_api_error(
                index=index,
                attempt=attempt,
                max_retries=max_retries,
                image_path=image_path,
                adv_text=adv_text,
                exception=e,
                error_title=f"API 状态码异常 status={e.status_code}",
            )

            if e.status_code in {
                400,
                401,
                403,
                404,
                405,
                422,
            }:
                return None, e

        except Exception as e:
            last_exception = e

            print_api_error(
                index=index,
                attempt=attempt,
                max_retries=max_retries,
                image_path=image_path,
                adv_text=adv_text,
                exception=e,
                error_title="API 调用异常",
            )

            if is_non_retryable_error(e):
                return None, e

        if attempt < max_retries:
            time.sleep(retry_interval)

    return None, last_exception


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "API Key 未提供，请通过 --api_key 参数提供，"
            "或者设置环境变量 OPENAI_API_KEY。"
        )

    if args.max_retries <= 0:
        raise ValueError("--max_retries 必须大于 0")

    if args.retry_interval < 0:
        raise ValueError("--retry_interval 不能小于 0")

    client = OpenAI(
        api_key=api_key,
        base_url="xxx",
    )

    attack = args.attack
    model_name = args.model
    csv_path = ATTACK_TO_CSV[attack]

    sample_limit_text = (
        str(args.max_samples)
        if args.max_samples > 0
        else "全部"
    )

    print(
        f"使用数据集 attack = {attack}, csv_path = {csv_path}",
        flush=True,
    )
    print(f"使用模型 model = {model_name}", flush=True)
    print(
        "调用方式 = client.chat.completions.create",
        flush=True,
    )
    print("reasoning_effort = none", flush=True)
    print(
        f"最多处理 CSV 前 {sample_limit_text} 个样本",
        flush=True,
    )
    print(
        "格式错误最多额外强化请求 1 次",
        flush=True,
    )

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    index = 0

    valid_total = 0

    success = 0

    skipped_total = 0

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV 没有表头: {csv_path}")

        required_columns = {"image_path", "adv_text"}
        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"CSV 缺少字段: {sorted(missing_columns)}；"
                f"实际字段: {reader.fieldnames}"
            )

        for row in reader:
            if args.max_samples > 0 and index >= args.max_samples:
                break

            index += 1
            sample_start_time = time.time()


            image_path = (row.get("image_path") or "").strip()
            adv_text = (row.get("adv_text") or "").strip()

            output_text: Optional[str] = None
            output_label: Optional[str] = None
            last_exception: Optional[Exception] = None

            if not image_path or not adv_text:
                skipped_total += 1
                sample_elapsed = time.time() - sample_start_time

                print("=" * 80, flush=True)
                print(
                    f"[SKIP] 样本 #{index} 的 image_path 或 "
                    "adv_text 为空，跳过，不计入统计。",
                    flush=True,
                )
                print(f"图片路径: {image_path}", flush=True)
                print(f"攻击文本: {adv_text}", flush=True)
                print(
                    f"该样本总耗时: {sample_elapsed:.3f} 秒",
                    flush=True,
                )
                print("=" * 80 + "\n", flush=True)
                continue

            try:
                image_data_url = image_file_to_data_url(image_path)
            except Exception as e:
                skipped_total += 1
                sample_elapsed = time.time() - sample_start_time

                print("=" * 80, flush=True)
                print(
                    f"[SKIP] 样本 #{index} 图片读取失败，"
                    "跳过，不计入统计。",
                    flush=True,
                )
                print(f"图片路径: {image_path}", flush=True)
                print(f"攻击文本: {adv_text}", flush=True)
                print(f"异常: {repr(e)}", flush=True)
                print(
                    f"该样本总耗时: {sample_elapsed:.3f} 秒",
                    flush=True,
                )
                print("=" * 80 + "\n", flush=True)
                continue

            for format_attempt in range(1, 3):
                strengthen_format = format_attempt == 2

                messages = build_messages(
                    image_data_url=image_data_url,
                    adv_text=adv_text,
                    strengthen_format=strengthen_format,
                )

                candidate_output, api_exception = request_with_api_retries(
                    client=client,
                    model_name=model_name,
                    messages=messages,
                    max_retries=args.max_retries,
                    retry_interval=args.retry_interval,
                    index=index,
                    image_path=image_path,
                    adv_text=adv_text,
                )

                if candidate_output is None:
                    last_exception = api_exception
                    break

                try:
                    candidate_label = validate_output_format(
                        candidate_output
                    )
                except OutputFormatError as e:
                    last_exception = e

                    print("=" * 80, flush=True)
                    print(
                        f"[WARN] 输出格式错误 "
                        f"(sample #{index}, format attempt {format_attempt}/2)",
                        flush=True,
                    )
                    print(f"图片路径: {image_path}", flush=True)
                    print(f"攻击文本: {adv_text}", flush=True)
                    print(f"模型输出: {candidate_output}", flush=True)
                    print(f"异常: {repr(e)}", flush=True)

                    if format_attempt == 1:
                        print(
                            "下一次请求将动态强化标签格式要求。",
                            flush=True,
                        )
                    else:
                        print(
                            "第二次仍未遵守格式，停止该样本，"
                            "不再继续产生调用费用。",
                            flush=True,
                        )

                    print("=" * 80 + "\n", flush=True)
                    continue

                output_text = candidate_output
                output_label = candidate_label
                break

            if output_text is None or output_label is None:
                skipped_total += 1
                sample_elapsed = time.time() - sample_start_time

                print("=" * 80, flush=True)
                print(
                    f"[SKIP] 样本 #{index} 最终失败，"
                    "跳过，不计入统计。",
                    flush=True,
                )
                print(f"图片路径: {image_path}", flush=True)
                print(f"攻击文本: {adv_text}", flush=True)

                if last_exception is not None:
                    print(
                        f"最后一次异常: {repr(last_exception)}",
                        flush=True,
                    )

                print(
                    f"该样本总耗时: {sample_elapsed:.3f} 秒",
                    flush=True,
                )
                print("=" * 80 + "\n", flush=True)
                continue

            valid_total += 1

            if output_label == "mismatch":
                success += 1

            success_rate = (
                success / valid_total
                if valid_total > 0
                else 0.0
            )

            sample_elapsed = time.time() - sample_start_time

            print("=" * 80, flush=True)
            print(f"样本序号: {index}", flush=True)
            print(
                f"当前攻击结果: {success}/{valid_total}  "
                f"(success rate = {success_rate:.4f})",
                flush=True,
            )
            print(
                f"该样本总耗时: {sample_elapsed:.3f} 秒",
                flush=True,
            )
            print(f"图片路径: {image_path}", flush=True)
            print(f"攻击文本: {adv_text}", flush=True)
            print("模型输出:", flush=True)
            print(output_text, flush=True)
            print("=" * 80 + "\n", flush=True)

    print("********** 最终统计 **********", flush=True)
    print(f"使用模型: {model_name}", flush=True)
    print(f"总处理样本数(含失败样本): {index}", flush=True)
    print(
        f"有效样本数(参与成功率计算): {valid_total}",
        flush=True,
    )
    print(f"跳过样本数: {skipped_total}", flush=True)
    print(
        f"攻击成功数(输出为 [mismatch]): {success}",
        flush=True,
    )

    if valid_total > 0:
        print(
            f"攻击成功率: {success / valid_total:.4f}",
            flush=True,
        )
    else:
        print("没有有效样本参与统计。", flush=True)


if __name__ == "__main__":
    main()
