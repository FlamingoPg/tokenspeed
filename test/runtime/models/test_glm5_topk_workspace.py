from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.models import glm5 as glm5_module
from tokenspeed.runtime.models.glm5 import (
    GlmDsaIndexer,
    GlmDsaIndexerOutput,
    GlmMoeDsaAttention,
    _build_prefill_kv_workspace_slots,
    _glm_dsa_hadamard_rotate,
    _glm_dsa_hadamard_rotate_pair,
)


def _make_attention_shell() -> GlmMoeDsaAttention:
    attn = object.__new__(GlmMoeDsaAttention)
    nn.Module.__init__(attn)
    return attn


def test_build_prefill_kv_workspace_slots_maps_pages_and_masks_padding() -> None:
    block_tables = torch.tensor(
        [
            [10, 11],
            [20, 21],
        ],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([6, 3], dtype=torch.int32)

    slots, bases = _build_prefill_kv_workspace_slots(
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=6,
        page_size=4,
        device=torch.device("cpu"),
    )

    assert slots.dtype == torch.int64
    assert slots.is_contiguous()
    assert slots.tolist() == [40, 41, 42, 43, 44, 45, 80, 81, 82]
    assert bases.tolist() == [0, 6]


def test_prefill_topk_fallback_uses_packed_workspace_indices() -> None:
    class FakePool:
        page_size = 4

        def get_index_k_buffer(self, layer_id):
            return torch.empty(128, 128)

    attn = _make_attention_shell()
    attn.indexer = SimpleNamespace(index_topk=4)
    attn.attn_mqa = SimpleNamespace(layer_id=0)
    attn._compute_prefill_topk_indices_deepgemm = lambda **kwargs: None

    chunk_meta = SimpleNamespace(
        extend_prefix_lens=torch.tensor([1, 0], dtype=torch.int32),
        extend_seq_lens=torch.tensor([2, 2], dtype=torch.int32),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
    )
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        num_extends=2,
        token_to_kv_pool=FakePool(),
        req_to_page=torch.tensor([[10], [20]], dtype=torch.int32),
        attn_backend=SimpleNamespace(chunked_prefill_metadata=chunk_meta),
    )
    indexer_output = GlmDsaIndexerOutput(
        query=torch.empty(4, 1, 128),
        key=torch.empty(4, 128),
        weights=torch.empty(4, 1),
    )

    topk = attn._compute_prefill_topk_indices(
        indexer_output,
        ctx,
        num_prefill_tokens=4,
    )

    assert topk is not None
    assert topk.kv_workspace_slots.tolist() == [40, 41, 42, 80, 81]
    assert topk.workspace_indices.tolist() == [
        [0, 1, -1, -1],
        [0, 1, 2, -1],
        [3, -1, -1, -1],
        [3, 4, -1, -1],
    ]
    assert topk.topk_lens.tolist() == [2, 3, 1, 2]


def test_indexer_fused_wk_weights_projection_matches_unfused(monkeypatch) -> None:
    torch.manual_seed(0)
    config = SimpleNamespace(
        index_topk=4,
        index_n_heads=3,
        index_head_dim=8,
        indexer_rope_interleave=False,
    )
    indexer = GlmDsaIndexer(
        config=config,
        hidden_size=7,
        q_lora_rank=5,
        qk_rope_head_dim=4,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=16,
        quant_config=None,
        prefix="test.indexer",
    )
    with torch.no_grad():
        indexer.wk.weight.copy_(torch.randn_like(indexer.wk.weight))
        indexer.weights_proj.weight.copy_(torch.randn_like(indexer.weights_proj.weight))
        indexer.wk_weights_proj.weight[: indexer.index_head_dim].copy_(
            indexer.wk.weight
        )
        indexer.wk_weights_proj.weight[indexer.index_head_dim :].copy_(
            indexer.weights_proj.weight
        )
    hidden_states = torch.randn(11, 7)

    indexer.set_wk_weights_proj_loaded(False)
    expected_key, expected_weights = indexer._compute_index_k_and_weights(hidden_states)
    indexer.set_wk_weights_proj_loaded(True)
    actual_key, actual_weights = indexer._compute_index_k_and_weights(hidden_states)

    torch.testing.assert_close(actual_key, expected_key)
    torch.testing.assert_close(actual_weights, expected_weights)


def test_indexer_fused_key_only_projection_matches_unfused(monkeypatch) -> None:
    torch.manual_seed(0)
    config = SimpleNamespace(
        index_topk=4,
        index_n_heads=3,
        index_head_dim=8,
        indexer_rope_interleave=False,
    )
    indexer = GlmDsaIndexer(
        config=config,
        hidden_size=7,
        q_lora_rank=5,
        qk_rope_head_dim=4,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=16,
        quant_config=None,
        prefix="test.indexer",
    )
    with torch.no_grad():
        indexer.wk.weight.copy_(torch.randn_like(indexer.wk.weight))
        indexer.weights_proj.weight.copy_(torch.randn_like(indexer.weights_proj.weight))
        indexer.wk_weights_proj.weight[: indexer.index_head_dim].copy_(
            indexer.wk.weight
        )
        indexer.wk_weights_proj.weight[indexer.index_head_dim :].copy_(
            indexer.weights_proj.weight
        )
    hidden_states = torch.randn(11, 7)

    indexer.set_wk_weights_proj_loaded(False)
    expected_key = indexer._compute_index_k_only(hidden_states)
    indexer.set_wk_weights_proj_loaded(True)
    actual_key = indexer._compute_index_k_only(hidden_states)

    torch.testing.assert_close(actual_key, expected_key)


def test_decode_topk_workspace_reuses_capacity() -> None:
    attn = _make_attention_shell()

    first = attn._get_decode_topk_workspace("_workspace", 2, 4, torch.device("cpu"))
    first.fill_(7)
    second = attn._get_decode_topk_workspace("_workspace", 1, 4, torch.device("cpu"))

    assert second.data_ptr() == first.data_ptr()
    assert second.shape == (1, 4)
    assert second.tolist() == [[-1, -1, -1, -1]]


def test_decode_topk_workspace_reallocates_for_wider_topk() -> None:
    attn = _make_attention_shell()

    first = attn._get_decode_topk_workspace("_workspace", 2, 4, torch.device("cpu"))
    second = attn._get_decode_topk_workspace("_workspace", 2, 8, torch.device("cpu"))

    assert second.data_ptr() != first.data_ptr()
    assert second.shape == (2, 8)
    assert second.tolist() == [[-1] * 8, [-1] * 8]


def test_try_decode_full_context_topk_does_not_need_indexer_output() -> None:
    class FakePool:
        page_size = 4

        def get_index_k_buffer(self, layer_id):
            raise AssertionError("full-context top-k should not read index K cache")

    attn = _make_attention_shell()
    attn.indexer = SimpleNamespace(index_topk=4)
    attn.attn_mqa = SimpleNamespace(layer_id=0)

    metadata = SimpleNamespace(
        num_extends=0,
        seq_lens_k=torch.tensor([2], dtype=torch.int32),
        block_kv_indices=torch.tensor([[9]], dtype=torch.int32),
    )
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        token_to_kv_pool=FakePool(),
        attn_backend=SimpleNamespace(
            forward_decode_metadata=metadata,
            spec_num_tokens=1,
        ),
        bs=1,
        num_extends=0,
    )

    topk_indices = attn._try_compute_decode_full_context_topk_indices(
        ctx,
        num_tokens=1,
        device=torch.device("cpu"),
    )

    assert topk_indices is not None
    assert topk_indices.topk_indices.tolist() == [[36, 37, -1, -1]]


def test_decode_deepgemm_topk_sanitizes_nonfinite_logits(monkeypatch) -> None:
    # This exercises the deepgemm + fast_topk sanitize path; force the
    # fast_topk fallback (on GPU hosts flashinfer's deterministic top-k
    # imports fine and would take the other branch).
    monkeypatch.setattr(glm5_module, "glm_dsa_decode_topk_deterministic", None)
    attn = _make_attention_shell()
    attn.indexer = SimpleNamespace(
        index_topk=512,
        index_head_dim=128,
        index_n_heads=1,
        softmax_scale=1.0,
    )
    attn.attn_mqa = SimpleNamespace(layer_id=0)

    class FakePool:
        page_size = 64

        def has_index_k_with_scale_buffer(self):
            return True

        def get_index_k_with_scale_buffer(self, layer_id):
            return torch.empty(64 * 80, 132, dtype=torch.uint8)

    class FakeDeepGemm:
        @staticmethod
        def get_num_sms():
            return 1

        @staticmethod
        def get_paged_mqa_logits_metadata(seq_lens, page_size, num_sms):
            return torch.zeros(1, 2, dtype=torch.int32)

        @staticmethod
        def fp8_paged_mqa_logits(*args, **kwargs):
            logits = torch.zeros(1, 5120, dtype=torch.float32)
            logits[0, 1] = float("inf")
            logits[0, 2] = float("nan")
            logits[0, 3] = float("-inf")
            return logits

    captured = {}

    def fake_quantize_fp8_with_scale(x, **kwargs):
        return x, torch.ones(x.shape[0], 1, dtype=torch.float32, device=x.device)

    def fake_fast_topk_v2(logits, seq_lens, out, topk, *args):
        captured["logits"] = logits.clone()
        out.fill_(0)

    monkeypatch.setattr(glm5_module, "deep_gemm", FakeDeepGemm)
    monkeypatch.setattr(glm5_module, "fast_topk_v2", fake_fast_topk_v2)
    monkeypatch.setattr(
        glm5_module, "quantize_fp8_with_scale", fake_quantize_fp8_with_scale
    )

    ctx = SimpleNamespace(
        token_to_kv_pool=FakePool(),
        attn_backend=SimpleNamespace(forward_decode_metadata=SimpleNamespace()),
    )
    indexer_output = GlmDsaIndexerOutput(
        query=torch.ones(1, 1, 128),
        key=torch.empty(1, 128),
        weights=torch.ones(1, 1),
    )

    result = attn._compute_decode_topk_indices_deepgemm(
        indexer_output=indexer_output,
        ctx=ctx,
        seq_lens_per_token=torch.tensor([4097], dtype=torch.int32),
        block_tables=torch.arange(80, dtype=torch.int32).view(1, 80),
        block_tables_per_token=torch.arange(80, dtype=torch.int32).view(1, 80),
        q_len_per_req=1,
        decode_start=0,
        num_tokens=1,
        num_decode_tokens=1,
        topk=512,
    )

    assert result is not None
    sanitized = captured["logits"]
    assert torch.isneginf(sanitized[0, 1])
    assert torch.isneginf(sanitized[0, 2])
    assert torch.isneginf(sanitized[0, 3])


def test_forward_dsa_indexer_key_only_writes_cache_without_full_indexer() -> None:
    class FakeIndexer:
        def forward_key_only(self, hidden_states, positions):
            return hidden_states + positions.float().unsqueeze(-1)

        def __call__(self, *args, **kwargs):
            raise AssertionError("key-only path should skip full indexer")

    class FakePool:
        def __init__(self):
            self.saved = None

        def set_index_k_buffer(self, layer_id, out_cache_loc, index_k):
            self.saved = (layer_id, out_cache_loc.clone(), index_k.clone())

    attn = _make_attention_shell()
    attn.indexer = FakeIndexer()
    attn.attn_mqa = SimpleNamespace(layer_id=3)
    pool = FakePool()
    ctx = SimpleNamespace(token_to_kv_pool=pool)
    comm_manager = SimpleNamespace(
        pre_attn_comm=lambda hidden_states, ctx: hidden_states
    )
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    positions = torch.tensor([10, 20])
    out_cache_loc = torch.tensor([5, 6], dtype=torch.int32)

    output = attn._forward_dsa_indexer(
        positions=positions,
        hidden_states=hidden_states,
        q_lora=torch.empty(2, 2),
        ctx=ctx,
        out_cache_loc=out_cache_loc,
        comm_manager=comm_manager,
        key_only=True,
    )

    assert output is None
    assert pool.saved is not None
    layer_id, saved_locs, saved_key = pool.saved
    assert layer_id == 3
    torch.testing.assert_close(saved_locs, out_cache_loc)
    torch.testing.assert_close(
        saved_key,
        torch.tensor([[11.0, 12.0], [23.0, 24.0]]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_paired_hadamard_matches_separate_calls() -> None:
    pytest.importorskip("tokenspeed_kernel.thirdparty.fast_hadamard_transform")
    query = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)

    query_ref = _glm_dsa_hadamard_rotate(query)
    key_ref = _glm_dsa_hadamard_rotate(key)
    query_actual, key_actual = _glm_dsa_hadamard_rotate_pair(query, key)

    torch.cuda.synchronize()
    assert torch.equal(query_actual, query_ref)
    assert torch.equal(key_actual, key_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_key_only_matches_full_indexer_key() -> None:
    pytest.importorskip("tokenspeed_kernel.thirdparty.fast_hadamard_transform")
    torch.manual_seed(0)
    config = SimpleNamespace(
        index_topk=4,
        index_n_heads=3,
        index_head_dim=128,
        indexer_rope_interleave=False,
    )
    indexer = (
        GlmDsaIndexer(
            config=config,
            hidden_size=7,
            q_lora_rank=5,
            qk_rope_head_dim=4,
            rope_theta=10000.0,
            rope_scaling=None,
            max_position_embeddings=16,
            quant_config=None,
            prefix="test.indexer",
        )
        .cuda()
        .bfloat16()
    )
    hidden_states = torch.randn(2, 7, device="cuda", dtype=torch.bfloat16)
    q_lora = torch.randn(2, 5, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([1, 2], device="cuda")
    with torch.no_grad():
        indexer.wq_b.weight.copy_(torch.randn_like(indexer.wq_b.weight))
        indexer.wk.weight.copy_(torch.randn_like(indexer.wk.weight))
        indexer.weights_proj.weight.copy_(torch.randn_like(indexer.weights_proj.weight))

    full_key = indexer(hidden_states, q_lora, positions).key
    key_only = indexer.forward_key_only(hidden_states, positions)

    torch.cuda.synchronize()
    torch.testing.assert_close(key_only, full_key)
