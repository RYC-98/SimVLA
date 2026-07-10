
"""
robust eval
"""

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


@dataclass
class RetrievalData:
    csv_path: str
    image_paths: List[str]
    texts: List[str]
    img2txt: List[List[int]]
    txt2img: np.ndarray

    @property
    def num_images(self) -> int:
        return len(self.image_paths)

    @property
    def num_texts(self) -> int:
        return len(self.texts)


@dataclass
class ModelConfig:
    name: str
    architecture: str
    checkpoint_path: str


@dataclass
class AttackConfig:
    name: str
    csv_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline evaluation for TeCoA2 and FARE2 on adversarial "
            "image-text retrieval."
        )
    )

    parser.add_argument(
        "--tecoa2_checkpoint",
        type=str,
        default="./your_path/tecoa2-clip/open_clip_pytorch_model.bin",  # https://huggingface.co/chs20/tecoa2-clip/tree/main
    )
    parser.add_argument(
        "--fare2_checkpoint",
        type=str,
        default="./your_path/fare2-clip/open_clip_pytorch_model.bin",   # https://huggingface.co/chs20/fare2-clip/tree/main
    )

    parser.add_argument(
        "--our_csv",
        type=str,
        default="./vlp/our_2_255/info.csv",
    )
    parser.add_argument(
        "--dra_csv",
        type=str,
        default="./vlp/dra_2_255/info.csv",
    )
    parser.add_argument(
        "--saaet_csv",
        type=str,
        default="./vlp/saaet_2_255/info.csv",
    )
    parser.add_argument(
        "--sga_csv",
        type=str,
        default="./vlp/sga_2_255/info.csv",
    )

    parser.add_argument(
        "--project_root",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vlp_robust_results",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="1,10",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--image_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--text_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--use_amp",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument(
        "--use_cache",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument(
        "--expected_captions_per_image",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def set_offline_environment() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_ks(ks_text: str) -> List[int]:
    ks = sorted({int(item.strip()) for item in ks_text.split(",") if item.strip()})
    if not ks or min(ks) <= 0:
        raise ValueError(f"非法的 ks: {ks_text}")
    return ks


def choose_device(device_text: str) -> torch.device:
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA 不可用，自动切换到 CPU。")
        return torch.device("cpu")
    return torch.device(device_text)


def find_text_column(fieldnames: Optional[Sequence[str]]) -> str:
    if not fieldnames:
        raise ValueError("CSV 缺少表头。")
    for name in ("adv_text", "text", "caption"):
        if name in fieldnames:
            return name
    raise ValueError("CSV 必须包含 adv_text、text 或 caption 中的一列。")


def resolve_image_path(raw_path: str, csv_path: Path, project_root: Path) -> str:
    raw = Path(raw_path).expanduser()
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                project_root / raw,
                csv_path.parent / raw,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"找不到图片：{raw_path}\nCSV：{csv_path}\n尝试过：\n  {tried}"
    )


def load_retrieval_csv(
    csv_path_text: str,
    project_root_text: str,
    expected_captions_per_image: int,
) -> RetrievalData:
    csv_path = Path(csv_path_text).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到 CSV：{csv_path}")

    project_root = Path(project_root_text).expanduser().resolve()
    image_key_to_index: "OrderedDict[str, int]" = OrderedDict()
    image_paths: List[str] = []
    texts: List[str] = []
    img2txt: List[List[int]] = []
    txt2img: List[int] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError(f"CSV 缺少 image_path 列：{csv_path}")
        text_column = find_text_column(reader.fieldnames)

        for row_index, row in enumerate(reader):
            raw_image_path = (row.get("image_path") or "").strip()
            text_value = row.get(text_column)
            text = "" if text_value is None else str(text_value).strip()

            if not raw_image_path:
                raise ValueError(f"第 {row_index + 2} 行 image_path 为空：{csv_path}")
            if not text:
                raise ValueError(f"第 {row_index + 2} 行文本为空：{csv_path}")

            if raw_image_path not in image_key_to_index:
                image_index = len(image_paths)
                image_key_to_index[raw_image_path] = image_index
                image_paths.append(
                    resolve_image_path(
                        raw_path=raw_image_path,
                        csv_path=csv_path.resolve(),
                        project_root=project_root,
                    )
                )
                img2txt.append([])
            else:
                image_index = image_key_to_index[raw_image_path]

            text_index = len(texts)
            texts.append(text)
            img2txt[image_index].append(text_index)
            txt2img.append(image_index)

    if not texts:
        raise ValueError(f"CSV 没有有效数据：{csv_path}")

    if expected_captions_per_image > 0:
        invalid = [
            (image_index, len(text_indices))
            for image_index, text_indices in enumerate(img2txt)
            if len(text_indices) != expected_captions_per_image
        ]
        if invalid:
            preview = ", ".join(
                f"image#{image_index}:{count}" for image_index, count in invalid[:10]
            )
            raise ValueError(
                f"存在图片对应文本数不为 {expected_captions_per_image}：{preview}"
            )

    print(
        f"[Data] {csv_path}: images={len(image_paths)}, texts={len(texts)}, "
        f"text_column={text_column}"
    )
    return RetrievalData(
        csv_path=str(csv_path.resolve()),
        image_paths=image_paths,
        texts=texts,
        img2txt=img2txt,
        txt2img=np.asarray(txt2img, dtype=np.int64),
    )


def file_fingerprint(path_text: str) -> str:
    path = Path(path_text).resolve()
    stat = path.stat()
    raw = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def dataset_fingerprint(data: RetrievalData) -> str:
    csv_path = Path(data.csv_path)
    stat = csv_path.stat()
    raw = (
        f"{csv_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|"
        f"{data.num_images}|{data.num_texts}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def load_open_clip_model(
    model_config: ModelConfig,
    device: torch.device,
):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "缺少 open_clip，请先安装 open_clip_torch。"
        ) from exc

    checkpoint_path = Path(model_config.checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"找不到 {model_config.name} 权重：{checkpoint_path}"
        )

    print(f"\n[Model] Loading {model_config.name}")
    print(f"[Model] Architecture: {model_config.architecture}")
    print(f"[Model] Checkpoint: {checkpoint_path.resolve()}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_config.architecture,
        pretrained=str(checkpoint_path.resolve()),
    )
    tokenizer = open_clip.get_tokenizer(model_config.architecture)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def autocast_context(device: torch.device, use_amp: bool):
    if device.type == "cuda" and use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def encode_images(
    model,
    preprocess,
    image_paths: Sequence[str],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    all_features: List[torch.Tensor] = []

    for start in tqdm(
        range(0, len(image_paths), batch_size),
        desc="Encode images",
        dynamic_ncols=True,
    ):
        batch_paths = image_paths[start : start + batch_size]
        batch_images = []

        for image_path in batch_paths:
            with Image.open(image_path) as image:
                batch_images.append(preprocess(image.convert("RGB")))

        pixel_values = torch.stack(batch_images, dim=0).to(
            device,
            non_blocking=True,
        )
        with torch.inference_mode(), autocast_context(device, use_amp):
            features = model.encode_image(pixel_values)

        features = F.normalize(features.float(), dim=-1)
        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def encode_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    all_features: List[torch.Tensor] = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Encode texts",
        dynamic_ncols=True,
    ):
        batch_texts = list(texts[start : start + batch_size])
        tokens = tokenizer(batch_texts).to(device, non_blocking=True)

        with torch.inference_mode(), autocast_context(device, use_amp):
            features = model.encode_text(tokens)

        features = F.normalize(features.float(), dim=-1)
        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def load_or_compute_features(
    model,
    preprocess,
    tokenizer,
    model_config: ModelConfig,
    data_name: str,
    data: RetrievalData,
    output_dir: Path,
    image_batch_size: int,
    text_batch_size: int,
    device: torch.device,
    use_amp: bool,
    use_cache: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_dir = output_dir / "feature_cache" / model_config.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = (
        f"{data_name}_{dataset_fingerprint(data)}_"
        f"{file_fingerprint(model_config.checkpoint_path)}"
    )
    image_cache_path = cache_dir / f"{cache_key}_image.pt"
    text_cache_path = cache_dir / f"{cache_key}_text.pt"

    if use_cache and image_cache_path.is_file() and text_cache_path.is_file():
        print(
            f"[Cache] Loading: {image_cache_path.name}, "
            f"{text_cache_path.name}"
        )
        image_features = torch.load(image_cache_path, map_location="cpu")
        text_features = torch.load(text_cache_path, map_location="cpu")
        return image_features.float(), text_features.float()

    image_features = encode_images(
        model=model,
        preprocess=preprocess,
        image_paths=data.image_paths,
        batch_size=image_batch_size,
        device=device,
        use_amp=use_amp,
    )
    text_features = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=data.texts,
        batch_size=text_batch_size,
        device=device,
        use_amp=use_amp,
    )

    if use_cache:
        torch.save(image_features, image_cache_path)
        torch.save(text_features, text_cache_path)
        print(
            f"[Cache] Saved: {image_cache_path.name}, "
            f"{text_cache_path.name}"
        )

    return image_features, text_features


def compute_hit_masks(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    img2txt: Sequence[Sequence[int]],
    txt2img: np.ndarray,
    ks: Sequence[int],
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    max_k = max(ks)
    num_images = image_features.shape[0]
    num_texts = text_features.shape[0]

    if max_k > num_texts:
        raise ValueError(f"max(K)={max_k} 大于文本数量 {num_texts}")
    if max_k > num_images:
        raise ValueError(f"max(K)={max_k} 大于图片数量 {num_images}")

    image_features_device = image_features.to(device, non_blocking=True)
    text_features_device = text_features.to(device, non_blocking=True)

    with torch.inference_mode():
        similarity = image_features_device @ text_features_device.T
        i2t_topk = (
            similarity.topk(k=max_k, dim=1, largest=True, sorted=True)
            .indices.cpu()
            .numpy()
        )
        t2i_topk = (
            similarity.T.topk(k=max_k, dim=1, largest=True, sorted=True)
            .indices.cpu()
            .numpy()
        )

    del similarity, image_features_device, text_features_device
    if device.type == "cuda":
        torch.cuda.empty_cache()

    i2t_hits: Dict[int, np.ndarray] = {}
    t2i_hits: Dict[int, np.ndarray] = {}

    for k in ks:
        image_hit = np.zeros(len(img2txt), dtype=bool)
        for image_index, gt_text_indices in enumerate(img2txt):
            image_hit[image_index] = np.isin(
                i2t_topk[image_index, :k],
                np.asarray(gt_text_indices, dtype=np.int64),
            ).any()

        text_hit = (
            t2i_topk[:, :k] == txt2img.reshape(-1, 1)
        ).any(axis=1)

        i2t_hits[k] = image_hit
        t2i_hits[k] = text_hit

    return i2t_hits, t2i_hits


def build_result_row(
    model_name: str,
    attack_name: str,
    data: RetrievalData,
    ks: Sequence[int],
    i2t_hits: Dict[int, np.ndarray],
    t2i_hits: Dict[int, np.ndarray],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "model": model_name,
        "attack": attack_name,
        "num_images": data.num_images,
        "num_texts": data.num_texts,
    }

    for k in ks:
        i2t_success_count = int(i2t_hits[k].sum())
        t2i_success_count = int(t2i_hits[k].sum())
        i2t_failure_count = data.num_images - i2t_success_count
        t2i_failure_count = data.num_texts - t2i_success_count

        i2t_recall = round(100.0 * i2t_success_count / data.num_images, 2)
        t2i_recall = round(100.0 * t2i_success_count / data.num_texts, 2)

        row[f"I2T_R@{k}"] = i2t_recall
        row[f"I2T_ASR@{k}"] = round(100.0 - i2t_recall, 2)
        row[f"I2T_attack_success_count@{k}"] = i2t_failure_count

        row[f"T2I_R@{k}"] = t2i_recall
        row[f"T2I_ASR@{k}"] = round(100.0 - t2i_recall, 2)
        row[f"T2I_attack_success_count@{k}"] = t2i_failure_count

    return row


def print_result_row(row: Dict[str, object], ks: Sequence[int]) -> None:
    print(f"\n[Result] model={row['model']}, attack={row['attack']}")

    for k in ks:
        print(
            f"  K={k}: "
            f"TR(I2T) R@{k}={row[f'I2T_R@{k}']:.2f}%, "
            f"TR(I2T) ASR@{k}={row[f'I2T_ASR@{k}']:.2f}% "
            f"({row[f'I2T_attack_success_count@{k}']}/{row['num_images']}); "

            f"IR(T2I) R@{k}={row[f'T2I_R@{k}']:.2f}%, "
            f"IR(T2I) ASR@{k}={row[f'T2I_ASR@{k}']:.2f}% "
            f"({row[f'T2I_attack_success_count@{k}']}/{row['num_texts']})"
        )


def write_results(rows: List[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)

    print(f"\n[Saved] {csv_path.resolve()}")
    print(f"[Saved] {json_path.resolve()}")


def main() -> None:
    args = parse_args()
    set_offline_environment()
    set_seed(args.seed)

    ks = parse_ks(args.ks)
    device = choose_device(args.device)
    use_amp = bool(args.use_amp)
    use_cache = bool(args.use_cache)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_configs = [
        ModelConfig(
            name="TeCoA2",
            architecture="ViT-L-14",
            checkpoint_path=args.tecoa2_checkpoint,
        ),
        ModelConfig(
            name="FARE2",
            architecture="ViT-L-14",
            checkpoint_path=args.fare2_checkpoint,
        ),
    ]

    attack_configs = [
        AttackConfig(name="OUR", csv_path=args.our_csv),
        AttackConfig(name="DRA", csv_path=args.dra_csv),
        AttackConfig(name="SAAET", csv_path=args.saaet_csv),
        AttackConfig(name="SGA", csv_path=args.sga_csv),
    ]

    print("=" * 80)
    print("Robust VLP offline retrieval evaluation")
    print(f"Models: {[config.name for config in model_configs]}")
    print(f"Attacks: {[config.name for config in attack_configs]}")
    print(f"K: {ks}")
    print(f"Device: {device}")
    print("Metric: direct ASR from each adversarial CSV; no clean data required")
    print("=" * 80)

    attack_data: Dict[str, RetrievalData] = {}
    for attack_config in attack_configs:
        attack_data[attack_config.name] = load_retrieval_csv(
            csv_path_text=attack_config.csv_path,
            project_root_text=args.project_root,
            expected_captions_per_image=args.expected_captions_per_image,
        )

    all_rows: List[Dict[str, object]] = []
    start_time = time.time()

    for model_config in model_configs:
        model, preprocess, tokenizer = load_open_clip_model(
            model_config=model_config,
            device=device,
        )

        for attack_config in attack_configs:
            data = attack_data[attack_config.name]

            image_features, text_features = load_or_compute_features(
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                model_config=model_config,
                data_name=attack_config.name,
                data=data,
                output_dir=output_dir,
                image_batch_size=args.image_batch_size,
                text_batch_size=args.text_batch_size,
                device=device,
                use_amp=use_amp,
                use_cache=use_cache,
            )

            i2t_hits, t2i_hits = compute_hit_masks(
                image_features=image_features,
                text_features=text_features,
                img2txt=data.img2txt,
                txt2img=data.txt2img,
                ks=ks,
                device=device,
            )

            row = build_result_row(
                model_name=model_config.name,
                attack_name=attack_config.name,
                data=data,
                ks=ks,
                i2t_hits=i2t_hits,
                t2i_hits=t2i_hits,
            )
            all_rows.append(row)
            print_result_row(row=row, ks=ks)
            write_results(rows=all_rows, output_dir=output_dir)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\nDone. Total time: {elapsed / 60.0:.2f} min")


if __name__ == "__main__":
    main()
