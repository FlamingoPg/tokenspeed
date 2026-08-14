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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends import dsa as dsa_backend
from tokenspeed.runtime.layers.attention.dsa.metadata import DSADecodePlan


def _backend(
    *,
    num_extends: int = 1,
    extend_seq_lens: torch.Tensor | None = None,
) -> dsa_backend.DSABackend:
    backend = dsa_backend.DSABackend.__new__(dsa_backend.DSABackend)
    decode_metadata = SimpleNamespace(
        num_extends=num_extends,
        seq_lens_k=torch.tensor([9, 20, 30], dtype=torch.int32),
        block_kv_indices=torch.tensor(
            [[1, 2], [3, 4], [5, 6]], dtype=torch.int32
        ),
        max_seq_len_k=256,
    )
    backend._dense_backend = SimpleNamespace(
        forward_decode_metadata=decode_metadata,
        chunked_prefill_metadata=SimpleNamespace(
            extend_seq_lens=(
                torch.tensor([3], dtype=torch.int32)
                if extend_seq_lens is None
                else extend_seq_lens
            )
        ),
    )
    backend.page_size = 64
    backend.max_context_len = 512
    backend.spec_num_tokens = 2
    backend.index_topk = 2
    backend.kv_lora_rank = 2
    backend.qk_nope_head_dim = 2
    backend.qk_rope_head_dim = 2
    backend.data_type = torch.bfloat16
    backend._decode_plan = None
    backend._decode_plan_key = None
    backend._decode_cuda_graph_plans = {}
    return backend


def test_mixed_decode_plan_is_the_single_packed_batch_contract(monkeypatch) -> None:
    backend = _backend()
    opaque_plan = object()
    calls = []

    def fake_dsa_plan(*, seq_lens_2d, page_size, out=None):
        calls.append((seq_lens_2d.clone(), page_size, out))
        return opaque_plan if out is None else out

    monkeypatch.setattr(dsa_backend, "dsa_plan", fake_dsa_plan)

    plan = backend.build_dsa_decode_plan(
        total_tokens=7,
        batch_size=3,
        num_extends=1,
    )

    assert plan is not None
    assert (plan.token_start, plan.token_end) == (3, 7)
    assert plan.num_requests == 2
    assert plan.num_tokens == 4
    assert plan.q_len_per_req == 2
    assert plan.seq_lens.tolist() == [20, 30]
    assert plan.block_tables.tolist() == [[3, 4], [5, 6]]
    assert plan.seq_lens_2d.flatten().tolist() == [20, 20, 30, 30]
    assert plan.max_seq_len == 256
    assert plan.kernel_plan is opaque_plan

    # All model layers in the same forward reuse one plan and one kernel plan.
    assert (
        backend.build_dsa_decode_plan(
            total_tokens=7,
            batch_size=3,
            num_extends=1,
        )
        is plan
    )
    assert len(calls) == 1

    dense_metadata = backend.forward_decode_metadata
    assert not hasattr(dense_metadata, "_dsa_plan")
    assert not hasattr(dense_metadata, "_dsa_seq_lens_2d")


def test_decode_plan_rejects_non_uniform_query_width(monkeypatch) -> None:
    backend = _backend()
    monkeypatch.setattr(dsa_backend, "dsa_plan", lambda **_: object())

    with pytest.raises(RuntimeError, match="decode token metadata mismatch"):
        backend.build_dsa_decode_plan(
            total_tokens=6,
            batch_size=3,
            num_extends=1,
        )


def test_decode_plan_rejects_request_metadata_disagreement(monkeypatch) -> None:
    backend = _backend()
    backend.forward_decode_metadata.block_kv_indices = torch.zeros(
        (2, 2), dtype=torch.int32
    )
    monkeypatch.setattr(dsa_backend, "dsa_plan", lambda **_: object())

    with pytest.raises(RuntimeError, match="request metadata mismatch"):
        backend.build_dsa_decode_plan(
            total_tokens=7,
            batch_size=3,
            num_extends=1,
        )


def test_decode_plan_refresh_keeps_graph_owned_storage(monkeypatch) -> None:
    backend = _backend(num_extends=0)
    backend.forward_decode_metadata.seq_lens_k = torch.tensor(
        [20, 30], dtype=torch.int32
    )
    backend.forward_decode_metadata.block_kv_indices = torch.tensor(
        [[3, 4], [5, 6]], dtype=torch.int32
    )
    opaque_plan = object()
    refreshed_out = []

    def fake_dsa_plan(*, seq_lens_2d, page_size, out=None):
        if out is not None:
            refreshed_out.append(out)
        return opaque_plan if out is None else out

    monkeypatch.setattr(dsa_backend, "dsa_plan", fake_dsa_plan)
    plan = backend.build_dsa_decode_plan(
        total_tokens=4,
        batch_size=2,
        num_extends=0,
    )
    assert plan is not None
    seq_lens_2d_ptr = plan.seq_lens_2d.data_ptr()

    plan.seq_lens.copy_(torch.tensor([40, 50], dtype=torch.int32))
    backend._refresh_decode_plan(plan)

    assert plan.seq_lens_2d.data_ptr() == seq_lens_2d_ptr
    assert plan.seq_lens_2d.flatten().tolist() == [40, 40, 50, 50]
    assert refreshed_out == [opaque_plan]


def test_cuda_graph_replay_refreshes_the_captured_plan_in_place(monkeypatch) -> None:
    backend = _backend(num_extends=0)
    dense = backend._dense_backend
    dense.forward_decode_metadata.seq_lens_k = torch.tensor(
        [20, 30], dtype=torch.int32
    )
    dense.forward_decode_metadata.block_kv_indices = torch.tensor(
        [[3, 4], [5, 6]], dtype=torch.int32
    )
    # Draft dense metadata describes the later one-token steps. The outer DSA
    # capture still starts with the two-token verify window.
    dense.forward_decode_metadata.group_q_len_per_req = 1

    def capture_metadata(**_kwargs):
        return None

    def replay_metadata(*, seq_lens, **_kwargs):
        dense.forward_decode_metadata.seq_lens_k.copy_(seq_lens)

    dense.init_forward_metadata_capture_cuda_graph = capture_metadata
    dense.init_forward_metadata_replay_cuda_graph = replay_metadata
    opaque_plan = object()
    monkeypatch.setattr(
        dsa_backend,
        "dsa_plan",
        lambda **kwargs: opaque_plan if kwargs.get("out") is None else kwargs["out"],
    )

    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([20, 30], dtype=torch.int32),
        forward_mode=None,
    )
    captured_plan = backend._decode_cuda_graph_plans[2]
    captured_ptr = captured_plan.seq_lens_2d.data_ptr()

    backend.init_forward_metadata_replay_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([40, 50], dtype=torch.int32),
    )

    assert backend._decode_plan is captured_plan
    assert captured_plan.seq_lens_2d.data_ptr() == captured_ptr
    assert captured_plan.seq_lens_2d.flatten().tolist() == [40, 40, 50, 50]
    assert (
        backend.build_dsa_decode_plan(
            total_tokens=4,
            batch_size=2,
            num_extends=0,
        )
        is captured_plan
    )


def test_sparse_decode_consumes_plan_without_reinferring_shape(monkeypatch) -> None:
    backend = _backend(num_extends=0)
    captured = {}

    def fake_dsa_decode(**kwargs):
        captured.update(kwargs)
        return torch.zeros((4, 1, 2), dtype=torch.bfloat16)

    monkeypatch.setattr(dsa_backend, "dsa_decode", fake_dsa_decode)
    plan = DSADecodePlan(
        token_start=0,
        token_end=4,
        num_requests=2,
        q_len_per_req=2,
        seq_lens=torch.tensor([20, 30], dtype=torch.int32),
        block_tables=torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
        seq_lens_2d=torch.tensor([[20], [20], [30], [30]], dtype=torch.int32),
        max_seq_len=256,
        kernel_plan=object(),
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=1,
        head_dim=4,
        v_head_dim=2,
        scaling=1.0,
        logit_cap=0.0,
        k_scale_float=None,
    )
    pool = SimpleNamespace(
        quant_method="",
        get_key_buffer=lambda _layer_id: torch.zeros(
            (8, 4), dtype=torch.bfloat16
        ),
    )

    output = backend.forward_sparse_decode(
        q=torch.zeros((4, 4), dtype=torch.bfloat16),
        k=None,
        v=None,
        layer=layer,
        out_cache_loc=torch.empty(0, dtype=torch.int32),
        token_to_kv_pool=pool,
        bs=2,
        save_kv_cache=False,
        topk_indices=torch.zeros((4, 2), dtype=torch.int32),
        topk_lens=torch.full((4,), 2, dtype=torch.int32),
        decode_plan=plan,
    )

    assert output.shape == (4, 2)
    assert captured["q_len_per_req"] == 2
    assert captured["max_seqlen_k"] == 256


def test_sparse_decode_rejects_a_different_request_shape(monkeypatch) -> None:
    backend = _backend(num_extends=0)
    monkeypatch.setattr(dsa_backend, "dsa_decode", lambda **_: None)
    plan = DSADecodePlan(
        token_start=0,
        token_end=4,
        num_requests=2,
        q_len_per_req=2,
        seq_lens=torch.tensor([20, 30], dtype=torch.int32),
        block_tables=torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
        seq_lens_2d=torch.tensor([[20], [20], [30], [30]], dtype=torch.int32),
        max_seq_len=256,
        kernel_plan=None,
    )
    layer = SimpleNamespace(logit_cap=0.0)
    pool = SimpleNamespace(quant_method="")

    with pytest.raises(RuntimeError, match="request count differs"):
        backend.forward_sparse_decode(
            q=torch.zeros((4, 4), dtype=torch.bfloat16),
            k=None,
            v=None,
            layer=layer,
            out_cache_loc=torch.empty(0, dtype=torch.int32),
            token_to_kv_pool=pool,
            bs=1,
            save_kv_cache=False,
            topk_indices=torch.zeros((4, 2), dtype=torch.int32),
            topk_lens=torch.full((4,), 2, dtype=torch.int32),
            decode_plan=plan,
        )
