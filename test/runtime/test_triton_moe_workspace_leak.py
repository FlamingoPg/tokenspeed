"""Regression: TritonMoEWorkspace must not leak on eager workspace regrows.

``TritonMoEWorkspace.empty`` caches one scratch buffer per name and regrows it
whenever a call needs a larger ``shape[0]`` (the post-dispatch token count, which
oscillates wildly across steps under high-concurrency / expert-parallel load).
The old buffer is retired (kept alive) so a captured CUDA graph can still replay
into it. Retiring on *every* regrow parked a full per-layer buffer set every time
the token count hit a new high, growing ``_retired`` without bound until the GPU
hit its memory ceiling and a downstream MoE kernel took an illegal memory access.

Only buffers allocated while a CUDA graph is capturing are graph-referenced and
must be retired; eager-path buffers are freed directly. (Capture runs
largest-batch-first, so capture itself never regrows -- the retire path is only
ever reached from the eager prefill / dispatch path.)
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.moe.backends.triton_common import TritonMoEWorkspace


def _get(ws: TritonMoEWorkspace, rows: int) -> torch.Tensor:
    return ws.empty(
        "intermediate_cache",
        (rows, 64),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_eager_regrows_do_not_retire(monkeypatch) -> None:
    """Growing token counts on the eager path must not accumulate buffers."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ws = TritonMoEWorkspace()
    for rows in range(8, 8 + 64):  # strictly increasing -> a regrow every call
        out = _get(ws, rows)
        assert out.shape == (rows, 64)
    assert len(ws._retired) == 0, f"eager regrows leaked {len(ws._retired)} buffers"


def test_capture_buffer_retired_but_eager_regrows_do_not_leak(monkeypatch) -> None:
    """A capture-time buffer is kept alive; later eager regrows are not."""
    ws = TritonMoEWorkspace()

    # Allocate while a graph is capturing -> graph-referenced buffer.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _get(ws, 16)
    assert ws._captured["intermediate_cache"] is True

    # First eager regrow must retain the captured buffer (a replay may still
    # write to it).
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    _get(ws, 32)
    assert len(ws._retired) == 1
    assert ws._captured["intermediate_cache"] is False

    # Every subsequent eager regrow frees its own buffer instead of parking it --
    # the retired list stays bounded at the single captured buffer.
    for rows in range(33, 33 + 64):
        _get(ws, rows)
    assert (
        len(ws._retired) == 1
    ), f"eager regrows after capture leaked {len(ws._retired) - 1} extra buffers"
