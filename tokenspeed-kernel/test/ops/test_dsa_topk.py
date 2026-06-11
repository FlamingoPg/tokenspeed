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

"""Tests for the deterministic DSA decode indexer top-k wrapper.

These exercise the guarantees that make long-context DSA decode usable under
CUDA graph (the reason this wrapper replaced the trtllm ``fast_topk``):

  * **Set parity with ``torch.topk``** — zero accuracy loss: the selected index
    *set* is exactly the mathematically-correct top-k (only tie-break/order
    differ from torch).
  * **Determinism** — repeated runs on identical logits return bit-identical
    indices (this is what trtllm ``indexer_topk_decode`` violated, breaking
    eager-vs-graph parity).
  * **``tie_break=SMALL``** — equal logits resolve to the smallest indices.
  * **Masking** — ``-inf`` rows (beyond a request's valid length) are never
    selected.
  * **In-place int32 output** + a clear error when the flashinfer path is
    unavailable.

The kernel runs on the GPU via flashinfer, so the numerical tests are gated on
CUDA + an importable deterministic ``top_k``; on CPU-only / non-NVIDIA hosts
they skip. ``test_unavailable_raises`` runs everywhere.
"""

from __future__ import annotations

import pytest
import torch

dsa_topk = pytest.importorskip("tokenspeed_kernel.ops.attention.flashinfer.dsa_topk")
glm_dsa_decode_topk_deterministic = dsa_topk.glm_dsa_decode_topk_deterministic
has_deterministic_decode_topk = dsa_topk.has_deterministic_decode_topk

requires_det_topk = pytest.mark.skipif(
    not (torch.cuda.is_available() and has_deterministic_decode_topk()),
    reason="needs CUDA + flashinfer deterministic top_k",
)


def _row_sets(t: torch.Tensor) -> list[set[int]]:
    return [set(row.tolist()) for row in t]


@requires_det_topk
@pytest.mark.parametrize(
    "rows,cols,topk",
    [(4, 2048, 256), (2, 512, 128), (8, 4096, 512), (1, 64, 8)],
)
def test_set_parity_with_torch_topk(rows: int, cols: int, topk: int) -> None:
    """Selected index set is exactly torch.topk's (no accuracy loss)."""
    torch.manual_seed(0)
    logits = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    out = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    glm_dsa_decode_topk_deterministic(logits, out, topk)
    ref = torch.topk(logits, topk, dim=-1).indices
    assert _row_sets(out) == _row_sets(ref)


@requires_det_topk
def test_deterministic_across_repeated_runs() -> None:
    """Repeated runs on identical logits are bit-identical (incl. order)."""
    torch.manual_seed(1)
    logits = torch.randn(4, 2048, device="cuda", dtype=torch.float32)
    topk = 256
    first = torch.empty(4, topk, device="cuda", dtype=torch.int32)
    glm_dsa_decode_topk_deterministic(logits, first, topk)
    for _ in range(7):
        out = torch.empty(4, topk, device="cuda", dtype=torch.int32)
        glm_dsa_decode_topk_deterministic(logits, out, topk)
        assert torch.equal(out, first)


@requires_det_topk
def test_tie_break_picks_smallest_indices() -> None:
    """All-equal logits -> the k smallest indices (tie_break=SMALL)."""
    cols, topk = 64, 8
    logits = torch.zeros(1, cols, device="cuda", dtype=torch.float32)
    out = torch.empty(1, topk, device="cuda", dtype=torch.int32)
    glm_dsa_decode_topk_deterministic(logits, out, topk)
    assert set(out[0].tolist()) == set(range(topk))


@requires_det_topk
def test_masked_inf_never_selected() -> None:
    """-inf beyond the valid length must never be selected."""
    cols, topk, valid = 512, 128, 200
    logits = torch.randn(2, cols, device="cuda", dtype=torch.float32)
    logits[:, valid:] = float("-inf")
    out = torch.empty(2, topk, device="cuda", dtype=torch.int32)
    glm_dsa_decode_topk_deterministic(logits, out, topk)
    assert int(out.max()) < valid


@requires_det_topk
def test_writes_int32_in_place() -> None:
    """Output is written in place as int32 valid offsets; returns None."""
    logits = torch.randn(2, 512, device="cuda", dtype=torch.float32)
    out = torch.full((2, 128), -1, device="cuda", dtype=torch.int32)
    ret = glm_dsa_decode_topk_deterministic(logits, out, 128)
    assert ret is None
    assert out.dtype == torch.int32
    assert (out >= 0).all()


def test_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flashinfer path disabled, raise a clear RuntimeError."""
    monkeypatch.setattr(dsa_topk, "top_k", dsa_topk.error_fn, raising=False)
    monkeypatch.setattr(dsa_topk, "TopKTieBreak", None, raising=False)
    if has_deterministic_decode_topk():
        pytest.skip("could not disable the flashinfer path for this test")
    logits = torch.zeros(1, 4)
    out = torch.zeros(1, 2, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="flashinfer deterministic top_k"):
        glm_dsa_decode_topk_deterministic(logits, out, 2)
