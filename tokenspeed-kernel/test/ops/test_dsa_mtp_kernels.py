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

"""Multi-query (MTP verify) kernel checks for GLM DSA decode.

Spec-verify feeds next_n query rows per request through the decode kernels.
Both the DeepGEMM paged MQA logits fork and FlashMLA sparse decode must
produce results identical to running the same rows as a batch of next_n=1
requests — same math, different batching.
"""

from __future__ import annotations

import pytest
import torch

try:
    from tokenspeed_kernel.thirdparty import deep_gemm
except Exception:  # pragma: no cover - optional dependency
    deep_gemm = None

from tokenspeed_kernel.ops.attention.flash_mla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from tokenspeed_kernel.registry import error_fn

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@requires_cuda
@pytest.mark.parametrize("next_n", [2, 3, 4, 5, 6])
def test_deepgemm_paged_mqa_logits_next_n_matches_batch_expansion(next_n: int):
    if deep_gemm is None or not hasattr(deep_gemm, "fp8_paged_mqa_logits"):
        pytest.skip("deep_gemm paged MQA logits unavailable")
    from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

    torch.manual_seed(0)
    dev = "cuda"
    bs, heads, dim, page = 2, 32, 128, 64
    max_blocks = 32  # per request
    max_seq = max_blocks * page

    total_blocks = bs * max_blocks
    kv_raw = (
        torch.randn(total_blocks * page, dim, device=dev, dtype=torch.bfloat16) * 0.3
    )
    kv_fp8, kv_scale = quantize_fp8_with_scale(
        kv_raw, granularity="token_group", group_size=dim, scale_encoding="float32"
    )
    kv_cache = torch.cat(
        [
            kv_fp8.view(torch.uint8),
            kv_scale.view(-1, 1).view(torch.uint8).view(-1, 4),
        ],
        dim=-1,
    ).view(total_blocks, page, 1, dim + 4)

    q = torch.randn(bs * next_n, heads, dim, device=dev, dtype=torch.bfloat16) * 0.5
    q_fp8, q_scale = quantize_fp8_with_scale(
        q.view(-1, dim),
        granularity="token_group",
        group_size=128,
        scale_encoding="float32",
    )
    q_fp8 = q_fp8.view_as(q)
    weights = (
        (
            torch.rand(bs * next_n, heads, 1, device=dev)
            * q_scale.view(bs * next_n, heads, 1)
        )
        .squeeze(-1)
        .contiguous()
    )

    block_tables = torch.arange(total_blocks, device=dev, dtype=torch.int32).view(
        bs, max_blocks
    )
    base_lens = torch.tensor([1500, 900], device=dev, dtype=torch.int32)
    # per-token visible lengths: seq - next_n + j + 1
    lens_2d = (
        base_lens.view(bs, 1)
        - next_n
        + torch.arange(1, next_n + 1, device=dev, dtype=torch.int32).view(1, next_n)
    ).contiguous()

    sched = deep_gemm.get_paged_mqa_logits_metadata(
        lens_2d, page, deep_gemm.get_num_sms()
    )
    logits_n = deep_gemm.fp8_paged_mqa_logits(
        q_fp8.view(bs, next_n, heads, dim),
        kv_cache,
        weights,
        lens_2d,
        block_tables,
        sched,
        max_seq,
        clean_logits=False,
    )

    lens_flat = lens_2d.reshape(-1, 1).contiguous()
    bt_exp = block_tables.repeat_interleave(next_n, dim=0).contiguous()
    sched1 = deep_gemm.get_paged_mqa_logits_metadata(
        lens_flat, page, deep_gemm.get_num_sms()
    )
    logits_1 = deep_gemm.fp8_paged_mqa_logits(
        q_fp8.view(bs * next_n, 1, heads, dim),
        kv_cache,
        weights,
        lens_flat,
        bt_exp,
        sched1,
        max_seq,
        clean_logits=False,
    )

    for r in range(bs * next_n):
        n = int(lens_flat[r, 0])
        torch.testing.assert_close(
            logits_n.view(bs * next_n, -1)[r, :n],
            logits_1[r, :n],
            rtol=0,
            atol=0,
        )


@requires_cuda
@pytest.mark.parametrize("q_len", [2, 3, 4, 5])
def test_flashmla_sparse_decode_q_len_matches_batch_expansion(q_len: int):
    if flash_mla_with_kvcache is error_fn:
        pytest.skip("FlashMLA unavailable")

    torch.manual_seed(0)
    dev = "cuda"
    bs, heads, hd, dv, page, topk = 2, 64, 576, 512, 64, 2048
    total = 8192

    nope = (
        (torch.randn(total, 512, device=dev) * 0.3)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    scales = (
        (torch.rand(total, 4, device=dev, dtype=torch.float32) * 0.5 + 0.5)
        .view(torch.uint8)
        .view(total, 16)
    )
    rope = (
        (torch.randn(total, 64, device=dev) * 0.3)
        .to(torch.bfloat16)
        .view(torch.uint8)
        .view(total, 128)
    )
    k_cache = (
        torch.cat([nope, scales, rope], dim=-1)
        .contiguous()
        .view(total // page, page, 1, 656)
    )
    q = torch.randn(bs, q_len, heads, hd, device=dev, dtype=torch.bfloat16) * 0.5

    idx = torch.full((bs, q_len, topk), -1, dtype=torch.int32, device=dev)
    base_lens = (1200, 400)
    for i in range(bs):
        pool = torch.randperm(total, device=dev)[:1300].sort().values.to(torch.int32)
        for j in range(q_len):
            n = base_lens[i] - q_len + j + 1
            idx[i, j, :n] = pool[:n]

    out_n, _ = flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=dv,
        tile_scheduler_metadata=get_mla_metadata()[0],
        softmax_scale=0.1,
        is_fp8_kvcache=True,
        indices=idx,
    )
    out_1, _ = flash_mla_with_kvcache(
        q=q.reshape(bs * q_len, 1, heads, hd),
        k_cache=k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=dv,
        tile_scheduler_metadata=get_mla_metadata()[0],
        softmax_scale=0.1,
        is_fp8_kvcache=True,
        indices=idx.view(bs * q_len, 1, topk),
    )
    torch.testing.assert_close(
        out_n.reshape(bs * q_len, heads, dv),
        out_1.reshape(bs * q_len, heads, dv),
        rtol=0,
        atol=0,
    )
