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

"""Square activation-scale layout contract for online-quantized fp8 mm.

The NVIDIA TRT-LLM helper behind ``per_token_group_quant_fp8`` returns
activation scales as ``[num_k_groups, M]`` even when row-major is requested.
``_online_quantize_mxfp8``'s layout fixup detected that via shape -- which is
ambiguous when ``M == num_k_groups``: the square scale matrix passed through
untransposed and every GEMM row dequantized with another row's group scales.

GLM5's shared-expert down GEMM hits exactly this square at decode bs=2
(M=2 rows, per-rank K=256 -> 2 groups), so every MoE layer's shared-expert
contribution was cross-row corrupted for 2-request batches while bs=1 and
bs>=3 stayed clean. This test pins row independence at the square shape.
"""

from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

BLK = [128, 128]


def _block_quant_weight(n: int, k: int, device: str):
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.03
    wt = w.view(n // 128, 128, k // 128, 128)
    amax = wt.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-6)
    scale = (amax / 448.0).float()
    w_fp8 = (wt / scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(n, k)
    return w_fp8, scale.squeeze(1).squeeze(-1).contiguous()


def _mm(a, w_fp8, w_scale_inv, override=None):
    return tokenspeed_kernel.mm(
        a,
        w_fp8,
        A_scales=None,
        B_scales=w_scale_inv,
        bias=None,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=BLK,
        override=override,
    )


@requires_cuda
@pytest.mark.parametrize(
    "override", [None, "triton_mm_fp8_blockscale", "flashinfer_mm_fp8_blockscale"]
)
@pytest.mark.parametrize("m,k", [(2, 256), (3, 384), (4, 512)])
def test_square_scale_rows_are_independent(override, m: int, k: int) -> None:
    """Row 0's output must not change when other rows' contents change.

    (m, k) pairs all satisfy m == k // 128, the shape-ambiguous square.
    """
    torch.manual_seed(7)
    dev = "cuda"
    n = 6144
    w_fp8, w_scale_inv = _block_quant_weight(n, k, dev)

    h0 = torch.randn(1, k, device=dev, dtype=torch.bfloat16)
    others = [
        torch.randn(1, k, device=dev, dtype=torch.bfloat16) * (i + 2)
        for i in range(m - 1)
    ]
    mixed = torch.cat([h0] + others).contiguous()
    duped = torch.cat([h0] * m).contiguous()

    try:
        out_mixed = _mm(mixed, w_fp8, w_scale_inv, override)
        out_duped = _mm(duped, w_fp8, w_scale_inv, override)
    except Exception as exc:  # pragma: no cover - kernel unavailable
        pytest.skip(f"kernel unavailable: {exc}")

    torch.testing.assert_close(out_mixed[0], out_duped[0], rtol=0, atol=0)


@requires_cuda
def test_square_scale_matches_dequant_reference() -> None:
    """M == K-groups square output must match a bf16 dequant reference."""
    torch.manual_seed(7)
    dev = "cuda"
    m, k, n = 2, 256, 6144
    w_fp8, w_scale_inv = _block_quant_weight(n, k, dev)

    a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    # rows with very different magnitudes make transposed scales obvious
    a[1] *= 37.0

    try:
        out = _mm(a.contiguous(), w_fp8, w_scale_inv)
    except Exception as exc:  # pragma: no cover - kernel unavailable
        pytest.skip(f"kernel unavailable: {exc}")

    w_deq = (
        w_fp8.view(n // 128, 128, k // 128, 128).float()
        * w_scale_inv.view(n // 128, 1, k // 128, 1)
    ).view(n, k)
    ref = a.float() @ w_deq.t()
    rel = (out.float() - ref).abs().max() / ref.abs().max()
    assert rel < 0.05, f"square-scale mm deviates from reference: rel={rel:.3e}"
