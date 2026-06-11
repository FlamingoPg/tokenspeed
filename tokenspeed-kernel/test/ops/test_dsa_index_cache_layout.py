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

"""DeepGEMM paged-MQA index-K cache layout contract.

The kernel reads each page as ``[page*head fp8 | page*4B fp32 scales]`` --
scales packed after the page's fp8 block. A per-token ``[128 fp8 | 4B scale]``
interleave silently corrupts every score (reads scales as fp8 payload), which
is invisible while requests sit on the <= index_topk shortcut and then
progressively destroys generation quality beyond it. This test pins the
contract numerically against a bf16 reference.
"""

from __future__ import annotations

import pytest
import torch

try:
    from tokenspeed_kernel.thirdparty import deep_gemm
except Exception:  # pragma: no cover - optional dependency
    deep_gemm = None

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

DIM, HEADS, PAGE, TOPK = 128, 64, 64, 2048


def _quant_token(x: torch.Tensor):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = amax / 448.0
    return (x / scale).to(torch.float8_e4m3fn), scale.squeeze(-1)


def _pack_page_grouped(k_fp8, k_scale, n_pages):
    buf = torch.zeros(n_pages, PAGE * (DIM + 4), dtype=torch.uint8, device=k_fp8.device)
    buf[:, : PAGE * DIM] = k_fp8.view(torch.uint8).view(n_pages, PAGE * DIM)
    buf[:, PAGE * DIM :] = (
        k_scale.float().view(torch.uint8).view(-1, 4).view(n_pages, PAGE * 4)
    )
    return buf.view(n_pages, PAGE, 1, DIM + 4)


@requires_cuda
@pytest.mark.parametrize("seq_len", [4096, 16384])
def test_paged_mqa_logits_page_grouped_layout_matches_reference(seq_len: int):
    if deep_gemm is None or not hasattr(deep_gemm, "fp8_paged_mqa_logits"):
        pytest.skip("deep_gemm paged MQA logits unavailable")

    torch.manual_seed(0)
    dev = "cuda"
    n_pages = seq_len // PAGE
    k = torch.randn(seq_len, DIM, device=dev, dtype=torch.float32)
    k_fp8, k_scale = _quant_token(k)
    cache = _pack_page_grouped(k_fp8, k_scale, n_pages)

    q = torch.randn(1, 1, HEADS, DIM, device=dev, dtype=torch.float32) * 0.7
    q_fp8, q_scale = _quant_token(q.view(-1, DIM))
    q_fp8 = q_fp8.view(1, 1, HEADS, DIM)
    w = torch.rand(1, HEADS, device=dev) / (HEADS**0.5)
    weights = w * q_scale.view(1, HEADS) * (DIM**-0.5)

    block_tables = torch.arange(n_pages, dtype=torch.int32, device=dev).view(1, -1)
    lens = torch.tensor([[seq_len]], dtype=torch.int32, device=dev)
    sched = deep_gemm.get_paged_mqa_logits_metadata(lens, PAGE, deep_gemm.get_num_sms())
    logits = deep_gemm.fp8_paged_mqa_logits(
        q_fp8,
        cache,
        weights,
        lens,
        block_tables,
        sched,
        seq_len,
        clean_logits=False,
    ).view(-1)[:seq_len]

    k_deq = k_fp8.to(torch.float32) * k_scale.view(-1, 1)
    scores = torch.einsum("hd,nd->hn", q_fp8.view(-1, DIM).to(torch.float32), k_deq)
    ref = torch.einsum("h,hn->n", weights.view(-1), torch.relu(scores))

    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits.float(), ref, rtol=1e-3, atol=1e-3)

    top_ref = set(torch.topk(ref, TOPK).indices.tolist())
    top_got = set(torch.topk(logits, TOPK).indices.tolist())
    assert len(top_ref & top_got) / TOPK > 0.999
