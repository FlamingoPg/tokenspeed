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

"""Round-trip contract for the GLM DSA sparse-decode FP8 pack/dequant pair.

The single-FP8 KV path packs the BF16 MLA latent (NoPE 512 + RoPE 64) into the
656-byte sparse-decode row (NoPE -> FP8 with one FP32 scale per 128 elements,
RoPE -> BF16 passthrough) and later dequantizes gathered rows back to BF16 to
feed the BF16-only flash_mla_sparse_fwd prefill kernel. This pins that the
dequant is the exact inverse of the pack within FP8 quantization error, and that
RoPE survives bit-exact.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton.dsa import (
    GLM_DSA_SPARSE_DECODE_ROW_BYTES,
    glm_dsa_dequant_sparse_decode_kv,
    glm_dsa_pack_sparse_decode_kv,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

NOPE_DIM = 512
ROPE_DIM = 64


@requires_cuda
@pytest.mark.parametrize("n", [1, 17, 256])
def test_pack_dequant_roundtrip(n: int) -> None:
    torch.manual_seed(7)
    dev = "cuda"
    # Mixed magnitudes so the per-128-group scales differ across rows/blocks.
    nope = torch.randn(n, NOPE_DIM, device=dev, dtype=torch.bfloat16)
    nope[:, :128] *= 0.01
    nope[:, 128:256] *= 4.0
    rope = torch.randn(n, ROPE_DIM, device=dev, dtype=torch.bfloat16)

    out = torch.zeros(n, GLM_DSA_SPARSE_DECODE_ROW_BYTES, dtype=torch.uint8, device=dev)
    loc = torch.arange(n, device=dev, dtype=torch.int64)
    glm_dsa_pack_sparse_decode_kv(
        out=out, loc=loc, cache_k_nope=nope, cache_k_rope=rope
    )

    deq = glm_dsa_dequant_sparse_decode_kv(out)
    assert deq.dtype == torch.bfloat16
    assert deq.shape == (n, NOPE_DIM + ROPE_DIM)

    deq_nope = deq[:, :NOPE_DIM]
    deq_rope = deq[:, NOPE_DIM:]

    # RoPE is stored as BF16 and must come back bit-exact.
    torch.testing.assert_close(deq_rope, rope, rtol=0, atol=0)

    # NoPE is FP8-e4m3 quantized with per-128 scales: a few % relative error.
    ref = nope.float()
    err = (deq_nope.float() - ref).abs()
    denom = ref.abs().clamp_min(1e-3)
    assert (err / denom).mean().item() < 0.05


@requires_cuda
def test_dequant_gather_matches_per_row() -> None:
    """Gather-then-dequant must equal dequant of the same rows (prefill path)."""
    torch.manual_seed(1)
    dev = "cuda"
    n = 200
    nope = torch.randn(n, NOPE_DIM, device=dev, dtype=torch.bfloat16) * 0.5
    rope = torch.randn(n, ROPE_DIM, device=dev, dtype=torch.bfloat16) * 0.5
    out = torch.zeros(n, GLM_DSA_SPARSE_DECODE_ROW_BYTES, dtype=torch.uint8, device=dev)
    loc = torch.arange(n, device=dev, dtype=torch.int64)
    glm_dsa_pack_sparse_decode_kv(
        out=out, loc=loc, cache_k_nope=nope, cache_k_rope=rope
    )

    full = glm_dsa_dequant_sparse_decode_kv(out)
    slots = torch.tensor([3, 3, 199, 0, 128, 77], device=dev, dtype=torch.int64)
    gathered = glm_dsa_dequant_sparse_decode_kv(out.index_select(0, slots))
    torch.testing.assert_close(gathered, full.index_select(0, slots), rtol=0, atol=0)
