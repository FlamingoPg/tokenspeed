# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Engine-side DeepSeek V4 Flash Vision image processor.

Port of ``inference/image_processor.py`` from
``deepseek-ai/DeepSeek-V4-Flash-Vision-Exp``. The official extra-vocab
layout is preserved so MoE ``bias_vl`` and prefill SWA can see
``input_ids >= vocab_size`` and the ``[IMAGE_START, IMAGE_END]`` sentinels.
Prefix-cache hashing rewrites only ``IMAGE`` slots.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Any
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import PretrainedConfig

from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
IMAGE_NEWLINE = IMAGE_NEW_LINE
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
COMPRESS_PAD_TO = 4
DEFAULT_VISION_MAX_N_TOKEN = 384


@dataclass(frozen=True)
class DeepseekV4VisionProcessConfig:
    vision_patch_size: int
    vision_downsample_ratio: int
    vision_max_n_token: int
    vision_min_pixels: int
    vision_max_wh_ratio: int | None
    vocab_size: int


def vision_process_config_from_hf(
    config: PretrainedConfig,
) -> DeepseekV4VisionProcessConfig:
    max_wh_ratio = getattr(config, "vision_max_wh_ratio", None)
    return DeepseekV4VisionProcessConfig(
        vision_patch_size=int(getattr(config, "vision_patch_size", 14)),
        vision_downsample_ratio=int(getattr(config, "vision_downsample_ratio", 3)),
        vision_max_n_token=int(
            getattr(config, "vision_max_n_token", DEFAULT_VISION_MAX_N_TOKEN)
        ),
        vision_min_pixels=int(getattr(config, "vision_min_pixels", 147456)),
        vision_max_wh_ratio=int(max_wh_ratio) if max_wh_ratio is not None else None,
        vocab_size=int(config.vocab_size),
    )


def extra_vocab_id(vocab_size: int, token_type: int) -> int:
    return vocab_size + int(token_type)


def is_deepseek_v4_image_token(token_id: int, vocab_size: int) -> bool:
    return int(token_id) >= int(vocab_size)


def grid_tokens(
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
) -> tuple[int, int, int]:
    """LLM tokens the aligner grid occupies (N-layout, including padding)."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int, int]:
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        if max_w <= 1:
            raise ValueError("DeepSeek V4 vision resize could not keep width > 1")
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def load_image_bytes(record: dict[str, Any]) -> bytes:
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)

    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            return base64.b64decode(source["data"])
        if source.get("url"):
            return load_image_bytes({"url": source["url"]})

    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            with urlopen(url, timeout=30) as response:
                return response.read()
        with open(url, "rb") as file:
            return file.read()

    raise ValueError(f"Cannot load image from record: {list(record.keys())}")


def load_image(
    record: dict[str, Any],
    process_config: DeepseekV4VisionProcessConfig,
) -> tuple[torch.Tensor, int, int, int, int]:
    """Load one image into ViT patches and the matching LLM grid size."""
    p = process_config.vision_patch_size
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        image = source.convert("RGB")
        width, height = image.size
        if (
            process_config.vision_max_wh_ratio is not None
            and width > height * process_config.vision_max_wh_ratio
        ):
            width = height * process_config.vision_max_wh_ratio
        if 0 < width * height < process_config.vision_min_pixels:
            ratio = (process_config.vision_min_pixels / (width * height)) ** 0.5
            width = int(width * ratio)
            height = int(height * ratio)
        best_width = math.ceil(width / p) * p
        best_height = math.ceil(height / p) * p
        n_llm_h, n_llm_w, best_height, best_width = safe_resize(
            height,
            width,
            best_height,
            best_width,
            p,
            process_config.vision_downsample_ratio,
            process_config.vision_max_n_token,
        )
        n_vit_h, n_vit_w = best_height // p, best_width // p
        if (
            process_config.vision_max_wh_ratio is not None
            and image.width >= process_config.vision_max_wh_ratio * image.height
        ):
            image = image.resize((best_width, best_height))
        else:
            image = ImageOps.pad(
                image, (best_width, best_height), color=(127, 127, 127)
            )
        x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
        x = ((x - 0.5) / 0.5).to(torch.bfloat16)
        patches = (
            x.reshape(3, n_vit_h, p, n_vit_w, p)
            .permute(1, 3, 0, 2, 4)
            .reshape(n_vit_h * n_vit_w, 3, p, p)
        )
        return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def build_image_block(
    n_llm_h: int, n_llm_w: int, start_pos: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """N-layout token types (final order) and aligner-row order for IMAGE slots."""
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = (
        torch.arange(rows * row_len)
        .view(rows // 2, 2, row_len)
        .transpose(1, 2)
        .reshape(-1)
    )
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START], dtype=torch.int64),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END], dtype=torch.int64),
        ]
    )
    return types, perm


def make_deepseek_v4_image_item(
    *,
    start: int,
    patches: torch.Tensor,
    n_vit_h: int,
    n_vit_w: int,
    types: torch.Tensor,
    perm: torch.Tensor,
    vocab_size: int,
    vision_max_n_token: int,
) -> MultimodalDataItem:
    end = start + int(types.numel()) - 1
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(start, end)],
        feature=patches.contiguous(),
        model_specific_data={
            "n_vit_h": torch.tensor(n_vit_h, dtype=torch.int64),
            "n_vit_w": torch.tensor(n_vit_w, dtype=torch.int64),
            "types": types.to(dtype=torch.int64).contiguous(),
            "perm": perm.to(dtype=torch.int64).contiguous(),
            "vocab_size": torch.tensor(vocab_size, dtype=torch.int64),
            "vision_max_n_token": torch.tensor(vision_max_n_token, dtype=torch.int64),
        },
    )


def pad_deepseek_v4_input_ids(
    input_ids: list[int],
    mm_items: list[MultimodalDataItem],
    vocab_size: int,
) -> list[int]:
    """Keep official sentinel IDs; hash-pad only IMAGE slots."""
    if not input_ids or not mm_items:
        return input_ids
    out = list(input_ids)
    for item in mm_items:
        if item.pad_value is None or not item.offsets:
            continue
        types = item.model_specific_data.get("types")
        if types is None:
            continue
        type_list = types.tolist()
        pad_value = int(item.pad_value)
        for offset_start, offset_end in item.offsets:
            span = offset_end - offset_start + 1
            if span != len(type_list):
                raise ValueError(
                    "DeepSeek V4 image offsets do not match sentinel types: "
                    f"span={span} types={len(type_list)}"
                )
            for offset, token_type in enumerate(type_list):
                position = offset_start + offset
                if token_type == IMAGE:
                    out[position] = pad_value
                else:
                    out[position] = extra_vocab_id(vocab_size, int(token_type))
    return out


def prepare_vl_inputs(
    prompt: str,
    images: list[dict[str, Any]],
    tokenizer: Any,
    process_config: DeepseekV4VisionProcessConfig,
) -> tuple[list[int], list[MultimodalDataItem]]:
    """Expand image placeholders into official sentinel blocks."""
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if image_token_id is None or image_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Token not found in tokenizer: {IMAGE_PLACEHOLDER}")
    prompt_tokens = tokenizer.encode(prompt)
    num_placeholders = sum(token == image_token_id for token in prompt_tokens)
    if num_placeholders != len(images):
        raise ValueError(
            f"Found {num_placeholders} image tokens but got {len(images)} images"
        )

    tokens: list[int] = []
    image_items: list[MultimodalDataItem] = []
    image_iter = iter(images)
    for tok in prompt_tokens:
        if tok != image_token_id:
            tokens.append(tok)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(
            next(image_iter), process_config
        )
        types, perm = build_image_block(n_llm_h, n_llm_w, len(tokens))
        image_items.append(
            make_deepseek_v4_image_item(
                start=len(tokens),
                patches=patches,
                n_vit_h=n_vit_h,
                n_vit_w=n_vit_w,
                types=types,
                perm=perm,
                vocab_size=process_config.vocab_size,
                vision_max_n_token=process_config.vision_max_n_token,
            )
        )
        tokens.extend(
            extra_vocab_id(process_config.vocab_size, int(token_type))
            for token_type in types.tolist()
        )
    return tokens, image_items
