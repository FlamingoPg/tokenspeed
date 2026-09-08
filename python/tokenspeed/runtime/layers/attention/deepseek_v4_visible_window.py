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

"""Prefill visible-window helpers for DeepSeek V4 Flash Vision.

Official SWA stays causal of width ``sliding_window`` outside an image
span. Inside ``[IMAGE_START, IMAGE_END]`` the window widens so the span
is bidirectional, bounded by ``vision_max_n_token``. Image blocks must
be prefilled in one chunk.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from tokenspeed.runtime.models.deepseek_v4_image_processor import (
    DEFAULT_VISION_MAX_N_TOKEN,
    IMAGE_END,
    IMAGE_START,
    extra_vocab_id,
)
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalInputs


def _image_offset_spans(
    mm_input: MultimodalInputs | None,
) -> Iterator[tuple[int, int, torch.Tensor]]:
    if mm_input is None:
        return
    for item in mm_input.mm_items:
        if item.modality != Modality.IMAGE or not item.offsets:
            continue
        data = item.model_specific_data
        if not data or "types" not in data:
            continue
        for start, end in item.offsets:
            yield int(start), int(end) + 1, data["types"]


def iter_image_spans(
    mm_input: MultimodalInputs | None,
) -> Iterator[tuple[int, int]]:
    """Yield half-open ``[start, end)`` image blocks, including compression pad."""
    for start, end, _types in _image_offset_spans(mm_input):
        yield start, end


def unsplittable_prefill_spans(
    mm_input: MultimodalInputs | None,
) -> list[tuple[int, int]]:
    """Image blocks the C++ scheduler must keep in one prefill chunk."""
    return list(iter_image_spans(mm_input))


def iter_visible_image_spans(
    mm_input: MultimodalInputs | None,
) -> Iterator[tuple[int, int]]:
    """Yield half-open ``[IMAGE_START, IMAGE_END]`` spans used for SWA."""
    for start, _end, types in _image_offset_spans(mm_input):
        type_list = types.tolist()
        try:
            rel_start = type_list.index(IMAGE_START)
            rel_end = len(type_list) - 1 - type_list[::-1].index(IMAGE_END)
        except ValueError:
            continue
        yield start + rel_start, start + rel_end + 1


def image_span_aligned_extend_end(
    mm_input: MultimodalInputs | None, extend_end: int
) -> int:
    """Shrink a chunk boundary to the start of any image block it cuts."""
    aligned = int(extend_end)
    for start, end in iter_image_spans(mm_input):
        if start < aligned < end:
            aligned = start
    return aligned


def image_span_cut_point(
    mm_input: MultimodalInputs | None, position: int
) -> int | None:
    aligned = image_span_aligned_extend_end(mm_input, position)
    return aligned if aligned < position else None


def get_image_visible(
    input_ids: torch.Tensor,
    vocab_size: int,
    max_image_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token visible counts to the left/right within each image span.

    ``input_ids`` is ``[batch, seq]``. Official clamp:
    ``left <= max_image_tokens - 1``, ``right <= max_image_tokens``.
    """
    seqlen = input_ids.size(1)
    idx = torch.arange(seqlen, dtype=torch.int32, device=input_ids.device).unsqueeze(0)
    start_id = extra_vocab_id(vocab_size, IMAGE_START)
    end_id = extra_vocab_id(vocab_size, IMAGE_END)
    is_start = input_ids == start_id
    is_end = input_ids == end_id
    valid = (is_start.cumsum(1) > is_end.cumsum(1)) | is_end
    starts = torch.where(is_start, idx, torch.zeros_like(idx)).cummax(1)[0]
    left = (idx - starts) * valid
    ends = (
        torch.where(is_end, idx, torch.full_like(idx, seqlen))
        .flip(1)
        .cummin(1)[0]
        .flip(1)
    )
    right = (ends - idx) * valid
    return left.clamp(max=max_image_tokens - 1), right.clamp(max=max_image_tokens)


def visible_window_for_token(
    pos: int,
    left: int,
    right: int,
    window_size: int,
    seq_len: int,
) -> tuple[int, int]:
    """Return inclusive ``[win_start, win_end]`` for one prefill token."""
    left_add = max(0, int(left) - (int(window_size) - 1))
    win_start = max(0, int(pos) - (int(window_size) - 1) - left_add)
    win_end = min(int(seq_len) - 1, int(pos) + int(right))
    return win_start, win_end


def compute_visible_window_overrides(
    *,
    mm_inputs: Sequence[MultimodalInputs | None] | None,
    extend_prefix_lens: Sequence[int],
    extend_seq_lens: Sequence[int],
    swa_window: int,
    max_image_tokens: int,
    padded_num_tokens: int,
) -> tuple[list[int], list[int]] | None:
    """Per-token extra left/right visibility for every extend token.

    Tokens outside an image span keep ``(left=0, right=0)`` — the causal
    SWA window. Tokens inside a span get official visible-window extras.
    Partial image blocks raise instead of silently changing attention.
    """
    if not mm_inputs:
        return None
    has_span = False
    lefts: list[int] = []
    rights: list[int] = []
    for req_idx, (prefix, extend_len) in enumerate(
        zip(extend_prefix_lens, extend_seq_lens, strict=True)
    ):
        prefix_i = int(prefix)
        extend_i = int(extend_len)
        extend_end = prefix_i + extend_i
        req_lefts = [0] * extend_i
        req_rights = [0] * extend_i
        mm_input = mm_inputs[req_idx] if req_idx < len(mm_inputs) else None
        for start, end in iter_image_spans(mm_input):
            if end <= prefix_i or start >= extend_end:
                continue
            if start < prefix_i or end > extend_end:
                raise ValueError(
                    f"DeepSeek V4 image block [{start}, {end}) crosses "
                    f"the prefill range [{prefix_i}, {extend_end})."
                )
        for start, end in iter_visible_image_spans(mm_input):
            if end <= prefix_i or start >= extend_end:
                continue
            has_span = True
            for pos in range(start, end):
                left = pos - start
                right = end - 1 - pos
                left = min(left, max_image_tokens - 1)
                right = min(right, max_image_tokens)
                req_lefts[pos - prefix_i] = left
                req_rights[pos - prefix_i] = right
        lefts.extend(req_lefts)
        rights.extend(req_rights)

    if not has_span:
        return None
    if padded_num_tokens > len(lefts):
        pad = padded_num_tokens - len(lefts)
        lefts.extend([0] * pad)
        rights.extend([0] * pad)
    return lefts, rights


def max_image_tokens_from_mm_inputs(
    mm_inputs: Sequence[MultimodalInputs | None] | None,
) -> int:
    """Largest ``vision_max_n_token`` advertised by the active image items."""
    found = DEFAULT_VISION_MAX_N_TOKEN
    if not mm_inputs:
        return found
    for mm_input in mm_inputs:
        if mm_input is None:
            continue
        for item in mm_input.mm_items:
            data = item.model_specific_data
            if not data or "vision_max_n_token" not in data:
                continue
            raw = data["vision_max_n_token"]
            found = max(found, int(raw.item() if hasattr(raw, "item") else raw))
    return found


def gather_window_for_visible_swa(
    window_size: int,
    max_image_tokens: int,
    has_visible_span: bool,
) -> int:
    if not has_visible_span:
        return max(int(window_size) - 1, 0)
    return max(int(window_size) + int(max_image_tokens) - 1, 0)
