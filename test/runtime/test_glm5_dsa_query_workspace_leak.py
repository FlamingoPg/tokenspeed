"""Regression: DSA decode query workspace must not leak on eager q_len flips.

``DSABackend._get_decode_query_workspace`` reallocates whenever the per-request
query length (``shape[1]``) changes -- which flips between plain decode (1) and
spec-verify (``spec_num_tokens``) on every step under MTP. The old buffer is
retired (kept alive) so a captured CUDA graph can still replay into it. Retiring
on *every* eager regrow grew ``_retired_decode_query_workspaces`` without bound
(~one ``[bs, q_len, num_heads, kv_cache_dim]`` query tensor per decode step) and
OOM'd long high-concurrency MTP runs. Only buffers allocated while a CUDA graph
is capturing are graph-referenced and must be retired; eager-path buffers are
freed directly.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.attention.backends.dsa import DSABackend


def _make_backend_shell() -> DSABackend:
    be = object.__new__(DSABackend)
    be._decode_query_workspace = None
    be._decode_query_workspace_num_heads = None
    be._decode_query_workspace_captured = False
    return be


def _get(be: DSABackend, *, seq_len: int) -> torch.Tensor:
    return be._get_decode_query_workspace(
        num_tokens=8,
        seq_len=seq_len,
        padded_heads=4,
        head_dim=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_eager_qlen_flips_do_not_retire_workspaces() -> None:
    """Plain eager decode/verify alternation must not accumulate buffers."""
    be = _make_backend_shell()
    for i in range(64):
        seq_len = 6 if i % 2 == 0 else 1  # spec-verify (6) vs plain decode (1)
        ws = _get(be, seq_len=seq_len)
        assert ws.shape == (8, seq_len, 4, 16)

    retired = getattr(be, "_retired_decode_query_workspaces", [])
    assert (
        len(retired) == 0
    ), f"eager q_len flips leaked {len(retired)} retired workspaces"


def test_capture_buffer_retired_but_eager_regrows_do_not_leak(monkeypatch) -> None:
    """A capture-time buffer is kept alive; later eager regrows are not."""
    be = _make_backend_shell()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    # Allocate while a graph is capturing -> graph-referenced buffer.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _get(be, seq_len=6)
    assert be._decode_query_workspace_captured is True

    # First eager regrow must retain the captured buffer (a replay may still
    # write to it).
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    _get(be, seq_len=1)
    retired = getattr(be, "_retired_decode_query_workspaces", [])
    assert len(retired) == 1
    assert be._decode_query_workspace_captured is False

    # Every subsequent eager q_len flip frees its own buffer instead of parking
    # it -- the retired list stays bounded.
    for i in range(32):
        _get(be, seq_len=6 if i % 2 == 0 else 1)
    assert (
        len(retired) == 1
    ), f"eager regrows after capture leaked {len(retired) - 1} extra workspaces"
