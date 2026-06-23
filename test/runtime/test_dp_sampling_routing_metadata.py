from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper
from tokenspeed.runtime.execution.drafter.eagle import (
    Eagle,
    should_reduce_draft_first_step,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.models.extensible import ExtensibleLM
from tokenspeed.runtime.models.glm5 import GlmDsaDecodeTopK
from tokenspeed.runtime.models.glm5_nextn import (
    GlmMoeDsaDraftDecoderLayer,
    GlmMoeDsaForCausalLMNextN,
)
from tokenspeed.runtime.sampling.dp_sampling_config import (
    DpSamplingRuntimeConfig,
    DpSamplingRuntimeLimits,
    DpSamplingSupport,
    DpSamplingTopology,
    resolve_dp_sampling_runtime,
    resolve_dp_sampling_support,
    validate_dp_sampling_lm_head_vocab,
)
from tokenspeed.runtime.sampling.logits_layout import LogitsLayoutPlan


def _graph_route(
    bs: int,
    ctx: ForwardContext,
    *,
    disable: bool = False,
    dp_size: int = 1,
    disable_padding: bool = False,
    max_bs: int,
    capture_bs: list[int],
    max_tokens_per_req: int = 1,
) -> tuple[bool, int]:
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper.disable = disable
    wrapper.dp_size = dp_size
    wrapper.disable_padding = disable_padding
    wrapper.max_bs = max_bs
    wrapper.capture_bs = capture_bs
    wrapper.graphs = set(capture_bs)
    wrapper.max_tokens_per_req = max_tokens_per_req
    use_graph = wrapper.can_run(bs, ctx)
    return use_graph, wrapper.padded_bs(bs, ctx) if use_graph else bs


def _dp_runtime_config(
    *,
    tp_rank: int = 0,
    tp_size: int = 4,
    tp_group: tuple[int, ...] = (0, 1, 2, 3),
    num_tokens_per_req: int = 6,
    min_bs: int = 8,
    max_bucket_bs: int = 8,
    vocab_size: int = 8,
    device: torch.device | str = "cpu",
    skip_all_gather: bool = False,
) -> DpSamplingRuntimeConfig:
    return DpSamplingRuntimeConfig(
        enabled=True,
        vocab_size=vocab_size,
        max_bucket_bs=max_bucket_bs,
        min_bs=min_bs,
        num_tokens_per_req=num_tokens_per_req,
        topology=DpSamplingTopology(
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            skip_all_gather=skip_all_gather,
        ),
        device=device,
    )


def test_extensible_lm_exposes_base_sampling_setup_handles():
    base = SimpleNamespace(logits_processor=object(), lm_head=object())
    ext = ExtensibleLM.__new__(ExtensibleLM)
    torch.nn.Module.__init__(ext)
    ext.base_lm = base

    assert ext.logits_processor is base.logits_processor
    assert ext.lm_head is base.lm_head


def test_logits_processor_dp_layout_threshold_and_modes():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    processor.configure_dp_logits_layout(_dp_runtime_config(min_bs=16))

    assert (
        processor._resolve_logits_layout_plan(
            torch.empty(15 * 6, 3),
            LogitsMetadata(forward_mode=ForwardMode.DECODE),
        )
        is None
    )

    decode_plan = processor._resolve_logits_layout_plan(
        torch.empty(16 * 6, 3),
        LogitsMetadata(forward_mode=ForwardMode.DECODE),
    )
    assert decode_plan is not None

    verify_plan = processor._resolve_logits_layout_plan(
        torch.empty(16 * 6, 3),
        LogitsMetadata(forward_mode=ForwardMode.TARGET_VERIFY),
    )
    assert verify_plan is not None

    assert (
        processor._resolve_logits_layout_plan(
            torch.empty(32 * 6, 3),
            LogitsMetadata(forward_mode=ForwardMode.EXTEND),
        )
        is None
    )


def test_cuda_graph_wrapper_uses_existing_route_for_padding():
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper.disable = False
    wrapper.dp_size = 1
    wrapper.disable_padding = False
    wrapper.max_bs = 32
    wrapper.capture_bs = [24, 32]
    wrapper.graphs = {24, 32}
    wrapper.max_tokens_per_req = 1
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=30,
        num_extends=0,
        input_num_tokens=30,
        forward_mode=ForwardMode.DECODE,
    )

    assert wrapper.can_run(30, ctx)
    assert wrapper.padded_bs(30, ctx) == 32


def test_cuda_graph_req_pool_padding_uses_reserved_sink_row():
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper.config = SimpleNamespace(max_req_pool_size=99)
    active_indices = torch.tensor([7, 8], dtype=torch.int64)

    padded_indices = wrapper._pad_graph_req_pool_indices(active_indices, 4)

    assert padded_indices.tolist() == [7, 8, 99, 99]


def test_cuda_graph_state_write_padding_uses_reserved_sink_row():
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper.config = SimpleNamespace(max_req_pool_size=99)
    wrapper.input_buffers = SimpleNamespace(
        state_write_req_pool_indices_buf=torch.full((4,), -1, dtype=torch.int64)
    )
    active_indices = torch.tensor([7, 8], dtype=torch.int64)

    wrapper._set_graph_state_write_indices(active_indices, 4)

    assert wrapper.input_buffers.state_write_req_pool_indices_buf.tolist() == [
        7,
        8,
        99,
        99,
    ]


def test_cuda_graph_replay_syncs_draft_seq_lens_before_draft_metadata():
    class Backend:
        uses_paged_cache_groups = False
        uses_padded_decode_token_mask = False

        def __init__(self):
            self.calls = []

        def init_forward_metadata_replay_cuda_graph(
            self,
            bs,
            req_pool_indices,
            seq_lens,
            *,
            req_to_page,
            forward_mode,
            **kwargs,
        ):
            self.calls.append(
                SimpleNamespace(
                    bs=bs,
                    seq_lens=seq_lens.clone(),
                    seq_lens_ptr=seq_lens.data_ptr(),
                    req_to_page=req_to_page,
                    forward_mode=forward_mode,
                )
            )

    target_backend = Backend()
    draft_backend = Backend()
    draft_seq_lens_buf = torch.full((4,), -1, dtype=torch.int32)
    draft_req_to_page = torch.zeros((4, 1), dtype=torch.int32)
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper.attn_backend = target_backend
    wrapper.draft_attn_backend = draft_backend
    wrapper.drafter = SimpleNamespace(
        draft_seq_lens_buf=draft_seq_lens_buf,
        req_to_page=draft_req_to_page,
    )
    wrapper.max_tokens_per_req = 6
    wrapper.use_target_verify_forward_mode = True

    seq_lens = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    wrapper._init_replay_metadata(
        padded_bs=4,
        actual_bs=4,
        req_pool_indices=torch.arange(4, dtype=torch.int64),
        seq_lens=seq_lens,
        req_to_page=torch.zeros((4, 1), dtype=torch.int32),
        forward_mode=ForwardMode.TARGET_VERIFY,
    )

    assert draft_seq_lens_buf.tolist() == [10, 11, 12, 13]
    assert target_backend.calls[0].seq_lens_ptr == seq_lens.data_ptr()
    assert draft_backend.calls[0].seq_lens_ptr == draft_seq_lens_buf.data_ptr()
    assert draft_backend.calls[0].seq_lens.tolist() == [10, 11, 12, 13]
    assert draft_backend.calls[0].req_to_page is draft_req_to_page
    assert draft_backend.calls[0].forward_mode is ForwardMode.DRAFT_EXTEND


def test_glm_nextn_draft_first_step_uses_reduced_collectives():
    model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)

    assert should_reduce_draft_first_step(model, ForwardMode.TARGET_VERIFY)
    assert not should_reduce_draft_first_step(model, ForwardMode.IDLE)
    assert should_reduce_draft_first_step(object(), ForwardMode.DECODE)
    assert not should_reduce_draft_first_step(object(), ForwardMode.TARGET_VERIFY)


def test_glm_nextn_first_step_correction_refreshes_dsa_metadata():
    layer = GlmMoeDsaDraftDecoderLayer.__new__(GlmMoeDsaDraftDecoderLayer)
    seq_lens = torch.tensor([100, 100, 100], dtype=torch.int32)
    accept_lengths = torch.tensor([4, 2, 1], dtype=torch.int32)
    refreshed_seq_lens = []

    class Backend:
        spec_num_tokens = 4

        def advance_draft_forward_metadata(self, lens):
            refreshed_seq_lens.append(lens.clone())

    ctx = ForwardContext(
        attn_backend=Backend(),
        token_to_kv_pool=None,
        bs=3,
        num_extends=1,
        input_num_tokens=9,
        forward_mode=ForwardMode.DRAFT_EXTEND,
        draft_seq_lens_buf=seq_lens,
        accept_lengths=accept_lengths,
    )

    layer._apply_correction(ctx)

    assert seq_lens.tolist() == [100, 98, 97]
    assert len(refreshed_seq_lens) == 1
    assert refreshed_seq_lens[0].tolist() == [100, 98, 97]


def test_glm_nextn_decode_first_step_narrows_to_accepted_rows():
    model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
    torch.nn.Module.__init__(model)
    corrected_contexts = []

    class Decoder:
        def _apply_correction(self, ctx):
            corrected_contexts.append(ctx)

    model.model = SimpleNamespace(decoder=Decoder())

    input_ids = torch.arange(12, dtype=torch.int32)
    positions = torch.arange(100, 112, dtype=torch.int64)
    out_cache_loc = torch.arange(200, 212, dtype=torch.int32)
    hidden_states = torch.arange(12 * 3, dtype=torch.float32).view(12, 3)
    gather_ids = torch.tensor([3, 10], dtype=torch.int64)
    ctx = ForwardContext(
        attn_backend=SimpleNamespace(spec_num_tokens=6),
        token_to_kv_pool=None,
        bs=2,
        num_extends=0,
        input_num_tokens=12,
        forward_mode=ForwardMode.DRAFT_EXTEND,
        gather_ids=gather_ids,
        global_num_tokens=[12],
        global_bs=[2],
        draft_first_step_reduce=True,
        accept_lengths=torch.tensor([4, 5], dtype=torch.int32),
    )

    narrowed_ctx, narrowed_ids, narrowed_positions, narrowed_cache, narrowed_hidden = (
        model._narrow_decode_first_step(
            ctx,
            input_ids,
            positions,
            out_cache_loc,
            hidden_states,
        )
    )

    assert len(corrected_contexts) == 1
    assert corrected_contexts[0] is ctx
    assert narrowed_ctx is not ctx
    assert narrowed_ctx.bs == 2
    assert narrowed_ctx.input_num_tokens == 2
    assert narrowed_ctx.forward_mode is ForwardMode.DRAFT_EXTEND
    assert narrowed_ctx.global_num_tokens == [2]
    assert narrowed_ctx.draft_first_step_reduce is False
    assert narrowed_ctx.gather_ids is None
    assert narrowed_ctx.accept_lengths is None
    assert narrowed_ids.tolist() == [3, 10]
    assert narrowed_positions.tolist() == [103, 110]
    assert narrowed_cache.tolist() == [203, 210]
    assert torch.equal(narrowed_hidden, hidden_states.index_select(0, gather_ids))


def test_glm_nextn_decode_first_step_keeps_topk_on_original_context():
    model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
    torch.nn.Module.__init__(model)

    class Decoder:
        def _apply_correction(self, ctx):
            return None

    class InnerModel:
        decoder = Decoder()

        def __call__(self, input_ids, positions, ctx, out_cache_loc, **kwargs):
            ctx.dsa_prefill_topk = "prefill-topk"
            ctx.dsa_decode_topk = "decode-topk"
            return torch.ones((input_ids.numel(), 4)), None

    class Processor:
        def __call__(self, input_ids, hidden_states, lm_head, logits_metadata):
            self.input_ids = input_ids
            self.logits_metadata = logits_metadata
            return SimpleNamespace(hidden_states=hidden_states, next_token_logits=None)

    processor = Processor()
    model.model = InnerModel()
    model.logits_processor = processor
    model.lm_head = object()
    ctx = ForwardContext(
        attn_backend=SimpleNamespace(spec_num_tokens=6),
        token_to_kv_pool=None,
        bs=2,
        num_extends=0,
        input_num_tokens=12,
        forward_mode=ForwardMode.DRAFT_EXTEND,
        gather_ids=torch.tensor([3, 10], dtype=torch.int64),
        global_num_tokens=[12],
        global_bs=[2],
        draft_first_step_reduce=True,
        accept_lengths=torch.tensor([4, 5], dtype=torch.int32),
    )

    output = model.forward(
        ctx=ctx,
        input_ids=torch.arange(12, dtype=torch.int32),
        positions=torch.arange(12, dtype=torch.int64),
        out_cache_loc=torch.arange(12, dtype=torch.int32),
        captured_hidden_states=torch.arange(12 * 4, dtype=torch.float32).view(12, 4),
    )

    assert ctx.dsa_prefill_topk == "prefill-topk"
    assert ctx.dsa_decode_topk == "decode-topk"
    assert processor.input_ids.tolist() == [3, 10]
    assert processor.logits_metadata.forward_mode is ForwardMode.DRAFT_EXTEND
    assert output.hidden_states.shape == (2, 4)


def test_eagle_run_carries_dsa_topk_from_target_context():
    eagle = Eagle.__new__(Eagle)
    seen = {}

    def fake_draft(draft_input):
        seen["dsa_topk"] = draft_input.dsa_topk
        return torch.empty((0,), dtype=torch.int32)

    eagle.draft = fake_draft
    prefill_topk = object()
    decode_topk = object()
    base_ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=2,
        num_extends=0,
        input_num_tokens=12,
        forward_mode=ForwardMode.TARGET_VERIFY,
        dsa_prefill_topk=prefill_topk,
        dsa_decode_topk=decode_topk,
    )

    result = Eagle.run(
        eagle,
        base_ctx,
        SimpleNamespace(hidden_states=torch.empty((2, 4))),
        torch.empty((12,), dtype=torch.int32),
        torch.ones((2,), dtype=torch.int32),
    )

    assert result.numel() == 0
    assert seen["dsa_topk"] == (prefill_topk, decode_topk)


def test_eagle_first_step_reuses_selected_target_dsa_decode_topk():
    eagle = Eagle.__new__(Eagle)
    eagle.spec_num_tokens = 6
    eagle.input_buffers = SimpleNamespace(
        positions_buf=torch.arange(12, dtype=torch.int64),
        out_cache_loc_buf=torch.arange(100, 112, dtype=torch.int32),
    )
    eagle.mm_pad_substitute_id = None
    eagle.padded_gather_ids_offsets_buf = torch.arange(2, dtype=torch.int64) * 6 - 1
    eagle.attn_backend = SimpleNamespace()
    eagle.token_to_kv_pool = None
    eagle.req_to_page = None
    eagle._dsa_reuse_mtp_topk = True
    eagle.draft_seq_lens_buf = torch.zeros((2,), dtype=torch.int32)
    draft_model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
    seen = {}

    class Runner:
        model = draft_model

        def forward(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                hidden_states=torch.empty((2, 4)),
                next_token_logits=torch.empty((2, 8)),
            )

    eagle.draft_model_runner = Runner()
    full_decode_topk = GlmDsaDecodeTopK(
        topk_indices=torch.arange(12 * 3, dtype=torch.int32).view(12, 3),
        topk_lens=torch.arange(12, dtype=torch.int32),
    )
    draft_input = SimpleNamespace(
        input_num_tokens=12,
        num_extends=0,
        forward_mode=ForwardMode.TARGET_VERIFY,
        base_model_output=torch.arange(12, dtype=torch.int32),
        accept_lengths=torch.tensor([1, 4], dtype=torch.int64),
        base_out_hidden_states=torch.empty((12, 4)),
        global_num_tokens=[12],
        global_bs=[2],
        all_decode_or_idle=True,
        dsa_topk=(None, full_decode_topk),
    )

    logits_output, dsa_topk = eagle._run_first_step(2, draft_input)

    selected = seen["ctx"].dsa_decode_topk
    assert selected.topk_lens.tolist() == [0, 9]
    assert selected.topk_indices.tolist() == [
        full_decode_topk.topk_indices[0].tolist(),
        full_decode_topk.topk_indices[9].tolist(),
    ]
    assert dsa_topk[1] is selected
    assert logits_output.next_token_logits.shape == (2, 8)


def test_cuda_graph_route_uses_global_batch_for_dp_idle_rank():
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=0,
        num_extends=0,
        input_num_tokens=0,
        forward_mode=ForwardMode.DECODE,
        global_num_tokens=[0, 17],
        all_decode_or_idle=True,
    )

    assert _graph_route(
        0,
        ctx,
        dp_size=2,
        max_bs=32,
        capture_bs=[16, 32],
        max_tokens_per_req=1,
    ) == (True, 32)


def test_cuda_graph_route_respects_disable_padding_with_global_batch():
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=0,
        num_extends=0,
        input_num_tokens=0,
        forward_mode=ForwardMode.DECODE,
        global_num_tokens=[0, 17],
        all_decode_or_idle=True,
    )

    assert _graph_route(
        0,
        ctx,
        dp_size=2,
        disable_padding=True,
        max_bs=32,
        capture_bs=[16, 32],
        max_tokens_per_req=1,
    ) == (False, 0)


def test_configure_dp_sampling_sets_state():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )

    processor.configure_dp_logits_layout(_dp_runtime_config())
    assert processor.dp_sampling_enabled
    assert processor.dp_num_tokens_per_req == 6


def test_resolve_dp_sampling_runtime_uses_grouped_metadata():
    support = DpSamplingSupport(
        requested=True,
        enabled=True,
        infra_supports=True,
        drafter_available=True,
        backend_supports_verify=True,
        tp_size=4,
        tp_group_set=True,
    )

    runtime_config = resolve_dp_sampling_runtime(
        support=support,
        lm_head_rows=7,
        topology=DpSamplingTopology(
            tp_rank=0,
            tp_size=4,
            tp_group=(0, 1, 2, 3),
            skip_all_gather=False,
        ),
        limits=DpSamplingRuntimeLimits(
            runtime_vocab_size=7,
            max_num_seqs=17,
            data_parallel_size=1,
            num_tokens_per_req=6,
            configured_min_bs=None,
            device="cpu",
        ),
    )

    assert runtime_config.enabled
    assert runtime_config.vocab_size == 28
    assert runtime_config.max_bucket_bs == 20
    assert runtime_config.min_bs == 8
    assert runtime_config.num_tokens_per_req == 6


@pytest.mark.parametrize(
    "forward_mode",
    [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY],
)
def test_logits_processor_derives_dp_layout_from_effective_hidden_states(
    forward_mode,
):
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    processor.configure_dp_logits_layout(_dp_runtime_config(min_bs=5))

    plan = processor._resolve_logits_layout_plan(
        torch.empty(5 * 6, 3),
        LogitsMetadata(forward_mode=forward_mode),
    )

    assert plan is not None
    assert plan.effective_bs == 5
    assert plan.bucket_bs == 8


def test_dp_sampling_skip_all_gather_rejects_sharded_lm_head_vocab():
    with pytest.raises(RuntimeError, match="replicated/full-vocab LM head"):
        validate_dp_sampling_lm_head_vocab(
            lm_head_rows=4,
            vocab_size=7,
            tp_size=2,
            skip_all_gather=True,
            tie_word_embeddings=True,
        )


def test_resolve_dp_sampling_support_rejects_missing_preconditions():
    with pytest.raises(RuntimeError, match="backend_supports_dp_verify=False"):
        resolve_dp_sampling_support(
            requested=True,
            drafter_available=True,
            backend_supports_verify=False,
            topology=DpSamplingTopology(
                tp_rank=0,
                tp_size=4,
                tp_group=(0, 1, 2, 3),
                skip_all_gather=False,
            ),
        )


def test_skip_all_gather_dp_sampling_slices_hidden_states_before_lm_head():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        skip_all_gather=True,
        tp_rank=1,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    processor.configure_dp_logits_layout(
        _dp_runtime_config(tp_rank=1, skip_all_gather=True, device="cpu")
    )
    hidden_states = torch.arange(5 * 6 * 3, dtype=torch.float32).view(5 * 6, 3)
    lm_head = SimpleNamespace(weight=torch.ones(7, 3))
    plan = LogitsLayoutPlan(
        effective_bs=5,
        bucket_bs=8,
        tp_size=4,
        num_tokens_per_req=6,
    )

    logits = processor._get_logits(
        hidden_states,
        lm_head,
        LogitsMetadata(forward_mode=ForwardMode.DECODE),
        plan=plan,
    )

    assert logits.shape == (12, 7)
    expected_rows = hidden_states[12:24].sum(dim=1)
    assert torch.equal(logits[:, 0], expected_rows)


def test_dp_sampling_slices_graph_effective_hidden_states_before_lm_head():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        skip_all_gather=True,
        tp_rank=2,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    processor.configure_dp_logits_layout(
        _dp_runtime_config(tp_rank=2, skip_all_gather=True, device="cpu")
    )
    hidden_states = torch.arange(5 * 6 * 3, dtype=torch.float32).view(5 * 6, 3)
    lm_head = SimpleNamespace(weight=torch.ones(7, 3))
    plan = LogitsLayoutPlan(
        effective_bs=5,
        bucket_bs=8,
        tp_size=4,
        num_tokens_per_req=6,
    )

    logits = processor._get_logits(
        hidden_states,
        lm_head,
        LogitsMetadata(forward_mode=ForwardMode.DECODE),
        plan=plan,
    )

    assert logits.shape == (12, 7)
    expected_rows = torch.cat(
        [hidden_states[24:30].sum(dim=1), torch.zeros(6, dtype=torch.float32)]
    )
    assert torch.equal(logits[:, 0], expected_rows)
