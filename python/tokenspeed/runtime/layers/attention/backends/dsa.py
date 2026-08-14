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

from contextlib import contextmanager

import torch
from tokenspeed_kernel.ops.attention import (
    dsa_decode,
    dsa_plan,
    dsa_prefill,
)
from tokenspeed_kernel.ops.attention.triton.dsa_topk import (
    workspace_topk_to_global_slots,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.backends.trtllm_mla import TRTLLMMLABackend
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.dsa.metadata import DSADecodePlan
from tokenspeed.runtime.layers.attention.registry import register_backend


def _make_dense_backend(config: DSAConfig, platform) -> AttentionBackend:
    if platform.is_nvidia:
        return TRTLLMMLABackend(config)
    if platform.is_amd:
        return MLAAttnBackend(config)
    raise RuntimeError(f"DSA backend does not support platform {platform.vendor!r}.")


class DSABackend(AttentionBackend):
    """DSA backend for sparse MLA attention.

    Dense MLA metadata and dense attention calls are delegated to a platform
    backend. DSA-specific decode plans remain owned here and are shared by the
    model's indexer and this backend's sparse attention path.
    """

    # DSA reads the history (full-attention) family: the dense sub-backend holds
    # the group tables, and the sparse path maps its top-k slots through the same
    # block_kv_indices. Declared here because the scheduler validates the
    # outermost backend, not the delegate.
    cache_consumer_families = frozenset({"history"})

    def __init__(self, config: DSAConfig):
        super().__init__(config)
        platform = current_platform()
        self._dense_backend = _make_dense_backend(config, platform)
        self.index_topk = config.index_topk
        self.max_context_len = config.context_len
        self.page_size = config.page_size
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_cache_dim
        self.scaling = config.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.num_local_heads = config.num_attention_heads // config.attn_tp_size
        self._prefill_block_tables: torch.Tensor | None = None
        self._decode_plan: DSADecodePlan | None = None
        self._decode_plan_key: tuple[int, ...] | None = None
        self._decode_cuda_graph_plans: dict[int, DSADecodePlan] = {}

    @property
    def forward_decode_metadata(self):
        return self._dense_backend.forward_decode_metadata

    @property
    def forward_prefill_metadata(self):
        return self._dense_backend.forward_prefill_metadata

    @property
    def chunked_prefill_metadata(self):
        return self._dense_backend.chunked_prefill_metadata

    @property
    def decode_cuda_graph_metadata(self):
        return self._dense_backend.decode_cuda_graph_metadata

    @property
    def decode_cuda_graph_kv_indices(self):
        return getattr(self._dense_backend, "decode_cuda_graph_kv_indices", None)

    @decode_cuda_graph_kv_indices.setter
    def decode_cuda_graph_kv_indices(self, value):
        if not hasattr(self._dense_backend, "decode_cuda_graph_kv_indices"):
            raise RuntimeError(
                "DSA dense backend does not expose decode CUDA graph KV indices."
            )
        self._dense_backend.decode_cuda_graph_kv_indices = value

    @property
    def trtllm_workspace(self):
        return self._dense_backend.trtllm_workspace

    @property
    def _block_table_aliased(self):
        return getattr(self._dense_backend, "_block_table_aliased", False)

    @_block_table_aliased.setter
    def _block_table_aliased(self, value):
        if hasattr(self, "_dense_backend"):
            self._dense_backend._block_table_aliased = value

    def register_step_counter(self, step_counter):
        super().register_step_counter(step_counter)
        self._dense_backend.register_step_counter(step_counter)

    @contextmanager
    def override_num_extends(self, num_extends: int):
        self._clear_decode_plan()
        with self._dense_backend.override_num_extends(num_extends):
            try:
                yield
            finally:
                self._clear_decode_plan()

    def mark_cache_contract(self) -> None:
        """Forward the contract mark to the dense sub-backend, which owns the
        group tables and the graph write-location buffer."""
        self._dense_backend.mark_cache_contract()

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        return self._dense_backend.select_out_cache_loc(
            layer, out_cache_loc, forward_mode
        )

    def init_cuda_graph_state(self, max_bs: int):
        self._dense_backend.init_cuda_graph_state(max_bs)
        self._decode_cuda_graph_plans.clear()
        self._clear_decode_plan()

    def _clear_decode_plan(self) -> None:
        self._decode_plan = None
        self._decode_plan_key = None

    def _prefill_token_count(self, num_extends: int) -> int:
        if num_extends <= 0:
            return 0
        metadata = self.chunked_prefill_metadata
        extend_seq_lens = getattr(metadata, "extend_seq_lens", None)
        if extend_seq_lens is None or extend_seq_lens.numel() < num_extends:
            available = 0 if extend_seq_lens is None else extend_seq_lens.numel()
            raise RuntimeError(
                "DSA decode plan requires one prefill query length per extend "
                f"request; requests={num_extends}, lengths={available}."
            )
        return int(extend_seq_lens[:num_extends].sum().item())

    def build_dsa_decode_plan(
        self,
        *,
        total_tokens: int,
        batch_size: int,
        num_extends: int,
    ) -> DSADecodePlan | None:
        """Build the decode contract shared by DSA top-k and attention.

        ``total_tokens`` is the actual packed model input after graph padding is
        removed.  Scheduler request counts determine the request split, while
        dense MLA metadata supplies the corresponding sequence lengths and page
        tables.  Inconsistencies are rejected here instead of being repaired by
        independent model/backend fallbacks.
        """

        total_tokens = int(total_tokens)
        batch_size = int(batch_size)
        num_extends = int(num_extends)
        if total_tokens < 0 or batch_size < 0:
            raise ValueError(
                "DSA decode plan sizes must be non-negative; "
                f"tokens={total_tokens}, batch_size={batch_size}."
            )
        if not 0 <= num_extends <= batch_size:
            raise ValueError(
                "DSA decode plan requires 0 <= num_extends <= batch_size; "
                f"num_extends={num_extends}, batch_size={batch_size}."
            )

        num_requests = batch_size - num_extends
        if num_requests == 0:
            return None

        metadata = self.forward_decode_metadata
        seq_lens_k = getattr(metadata, "seq_lens_k", None)
        block_kv_indices = getattr(metadata, "block_kv_indices", None)
        if metadata is None or seq_lens_k is None or block_kv_indices is None:
            raise RuntimeError(
                "DSA sparse decode requires initialized sequence lengths and "
                "block tables."
            )

        metadata_num_extends = int(getattr(metadata, "num_extends", 0) or 0)
        plan_key = (
            id(metadata),
            total_tokens,
            batch_size,
            num_extends,
            metadata_num_extends,
        )
        if self._decode_plan is not None and self._decode_plan_key == plan_key:
            return self._decode_plan

        available_seq_lens = max(0, int(seq_lens_k.shape[0]) - metadata_num_extends)
        available_block_tables = max(
            0, int(block_kv_indices.shape[0]) - metadata_num_extends
        )
        if available_seq_lens != num_requests or available_block_tables != num_requests:
            raise RuntimeError(
                "DSA decode request metadata mismatch: "
                f"scheduled={num_requests}, seq_lens={available_seq_lens}, "
                f"block_tables={available_block_tables}, "
                f"metadata_num_extends={metadata_num_extends}."
            )

        prefill_tokens = self._prefill_token_count(num_extends)
        if prefill_tokens > total_tokens:
            raise RuntimeError(
                "DSA packed token split is invalid: "
                f"tokens={total_tokens}, prefill_tokens={prefill_tokens}."
            )
        num_decode_tokens = total_tokens - prefill_tokens
        q_len_per_req, remainder = divmod(num_decode_tokens, num_requests)
        if remainder or q_len_per_req <= 0:
            raise RuntimeError(
                "DSA decode token metadata mismatch: "
                f"decode_tokens={num_decode_tokens}, requests={num_requests}."
            )
        if not 1 <= q_len_per_req <= 6:
            raise NotImplementedError(
                "DSA sparse decode supports 1-6 query tokens per request "
                f"(verified next_n <= 6), got {q_len_per_req}."
            )

        seq_lens = seq_lens_k[
            metadata_num_extends : metadata_num_extends + num_requests
        ]
        block_tables = block_kv_indices[
            metadata_num_extends : metadata_num_extends + num_requests
        ]
        seq_lens_2d = (
            seq_lens.unsqueeze(1).expand(-1, q_len_per_req).reshape(-1, 1).contiguous()
        )
        kernel_plan = dsa_plan(
            seq_lens_2d=seq_lens_2d,
            page_size=self.page_size,
        )
        max_seq_len = int(getattr(metadata, "max_seq_len_k", 0) or self.max_context_len)
        plan = DSADecodePlan(
            token_start=prefill_tokens,
            token_end=total_tokens,
            num_requests=num_requests,
            q_len_per_req=q_len_per_req,
            seq_lens=seq_lens,
            block_tables=block_tables,
            seq_lens_2d=seq_lens_2d,
            max_seq_len=max_seq_len,
            kernel_plan=kernel_plan,
        )
        self._decode_plan = plan
        self._decode_plan_key = plan_key
        return plan

    def _refresh_decode_plan(self, plan: DSADecodePlan) -> None:
        refreshed_seq_lens = (
            plan.seq_lens.unsqueeze(1).expand(-1, plan.q_len_per_req).reshape(-1, 1)
        )
        if refreshed_seq_lens.shape != plan.seq_lens_2d.shape:
            raise RuntimeError(
                "DSA decode plan shape changed during CUDA graph replay: "
                f"captured={tuple(plan.seq_lens_2d.shape)}, "
                f"replayed={tuple(refreshed_seq_lens.shape)}."
            )
        plan.seq_lens_2d.copy_(refreshed_seq_lens)
        dsa_plan(
            seq_lens_2d=plan.seq_lens_2d,
            page_size=self.page_size,
            out=plan.kernel_plan,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        self._dense_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=forward_mode,
        )
        self._clear_decode_plan()
        # A draft's dense metadata advertises one row per request for chained
        # draft steps, but the first model call in the captured graph still
        # consumes the target-shaped verify window.  Capture that outer width;
        # later one-row draft calls build their own actual-shape plan.
        q_len_per_req = max(1, int(self.spec_num_tokens))
        plan = self.build_dsa_decode_plan(
            total_tokens=bs * q_len_per_req,
            batch_size=bs,
            num_extends=0,
        )
        if plan is None:
            raise RuntimeError("DSA CUDA graph capture requires a decode plan.")
        self._decode_cuda_graph_plans[bs] = plan

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        page_table: torch.Tensor = None,
        **kwargs,
    ):
        self._dense_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=forward_mode,
            page_table=page_table,
            **kwargs,
        )
        plan = self._decode_cuda_graph_plans.get(bs)
        if plan is None:
            raise RuntimeError(
                f"DSA CUDA graph replay has no captured decode plan for bs={bs}."
            )
        metadata = self.forward_decode_metadata
        metadata_num_extends = int(getattr(metadata, "num_extends", 0) or 0)
        self._decode_plan = plan
        self._decode_plan_key = (
            id(metadata),
            plan.token_end,
            bs,
            0,
            metadata_num_extends,
        )
        self._refresh_decode_plan(plan)

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor | None = None):
        metadata = self.forward_decode_metadata
        if metadata is None or metadata.seq_lens_k is None:
            raise RuntimeError("DSA draft decode metadata was not initialized")
        if seq_lens is None:
            metadata.seq_lens_k.add_(1)
        else:
            metadata.seq_lens_k.copy_(seq_lens[: metadata.seq_lens_k.numel()])
        plan = self._decode_plan
        if (
            plan is not None
            and plan.q_len_per_req == 1
            and plan.num_requests == plan.seq_lens.numel()
        ):
            self._refresh_decode_plan(plan)
        else:
            self._clear_decode_plan()

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        page_table: torch.Tensor,
        **kwargs,
    ):
        self._dense_backend.init_forward_metadata(
            bs=bs,
            num_extends=num_extends,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=forward_mode,
            page_table=page_table,
            **kwargs,
        )
        self._clear_decode_plan()

        self._prefill_block_tables = None
        if num_extends > 0 and forward_mode.is_extend_or_mixed():
            cache_metadata = kwargs.get("cache_metadata")
            cmeta = getattr(self._dense_backend, "chunked_prefill_metadata", None)
            if cmeta is not None:
                # Extend requests are the first num_extends batch rows. The
                # target carries the full-history table in cache_metadata; a
                # draft is handed the batch-ordered draft page table directly.
                table = None
                if cache_metadata is not None:
                    table = cache_metadata.require_full_attention_table(
                        active_forward_op=kwargs.get("forward_batch")
                    )
                elif page_table is not None:
                    table = page_table
                if table is not None:
                    self._prefill_block_tables = table[:num_extends]
                    cmeta.block_tables = self._prefill_block_tables

    def _validate_logit_cap(self, logits_soft_cap: float) -> None:
        if logits_soft_cap and logits_soft_cap > 0:
            raise NotImplementedError(
                "TokenSpeed DSA fused dense attention does not support "
                f"logits_soft_cap={logits_soft_cap}. Sparse DSA kernels must "
                "preserve the capped-score semantics before enabling this model."
            )

    def _validate_dense_context(self, seq_lens: torch.Tensor, bs: int) -> None:
        if seq_lens is None or bs <= 0:
            return
        active_seq_lens = seq_lens[:bs]
        if active_seq_lens.numel() == 0:
            return
        max_seq_len = int(active_seq_lens.max().item())
        if max_seq_len > self.index_topk:
            raise NotImplementedError(
                "TokenSpeed DSA dense attention is exact only when every "
                f"request has seq_len <= index_topk ({self.index_topk}); got "
                f"max seq_len {max_seq_len}. Sparse DSA top-k indices are "
                "required for longer contexts."
            )

    def _metadata_seq_lens(self, metadata) -> torch.Tensor | None:
        seq_lens = getattr(metadata, "seq_lens_k", None)
        if seq_lens is not None:
            return seq_lens
        return getattr(metadata, "seq_lens", None)

    def forward_extend_chunked(
        self,
        q,
        k,
        v,
        scaling,
        logits_soft_cap,
        *,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        max_q_len,
        max_kv_len,
        seq_lens,
        batch_size,
        causal,
        out: torch.Tensor | None = None,
    ):
        self._validate_logit_cap(logits_soft_cap)
        self._validate_dense_context(seq_lens, batch_size)
        return self._dense_backend.forward_extend_chunked(
            q,
            k,
            v,
            scaling,
            logits_soft_cap,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            seq_lens=seq_lens,
            batch_size=batch_size,
            causal=causal,
            out=out,
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        topk_indices: torch.Tensor | None = None,
        topk_lens: torch.Tensor | None = None,
        decode_plan: DSADecodePlan | None = None,
        **kwargs,
    ) -> torch.Tensor:
        self._validate_logit_cap(layer.logit_cap)
        if topk_indices is not None:
            if decode_plan is None:
                raise RuntimeError(
                    "DSA sparse decode requires the plan used to compute top-k."
                )
            return self.forward_sparse_decode(
                q=q,
                k=k,
                v=v,
                layer=layer,
                out_cache_loc=out_cache_loc,
                token_to_kv_pool=token_to_kv_pool,
                bs=bs,
                save_kv_cache=save_kv_cache,
                topk_indices=topk_indices,
                topk_lens=topk_lens,
                decode_plan=decode_plan,
            )
        metadata = getattr(self, "forward_decode_metadata", None)
        seq_lens = self._metadata_seq_lens(metadata) if metadata is not None else None
        if seq_lens is not None:
            num_extends = int(metadata.num_extends or 0)
            self._validate_dense_context(seq_lens[num_extends:], bs)
        return self._dense_backend.forward_decode(
            q=q,
            k=k,
            v=v,
            layer=layer,
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=token_to_kv_pool,
            bs=bs,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

    def forward_sparse_prefill(
        self,
        *,
        q: torch.Tensor,
        layer,
        token_to_kv_pool,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace_indices: torch.Tensor,
        topk_lens: torch.Tensor,
        kv_workspace_slots: torch.Tensor | None = None,
        max_seq_len: int,
    ) -> torch.Tensor:
        if layer.logit_cap and layer.logit_cap > 0:
            self._validate_logit_cap(layer.logit_cap)
        if getattr(token_to_kv_pool, "quant_method", None) == "per_token_head":
            raise RuntimeError(
                "DSA sparse prefill does not support "
                "kv_cache_quant_method='per_token_head' yet."
            )
        if workspace_indices.shape[0] != q.shape[0]:
            raise RuntimeError(
                "DSA sparse prefill metadata token mismatch: "
                f"indices={workspace_indices.shape[0]}, q_tokens={q.shape[0]}"
            )
        if topk_lens.shape[0] != q.shape[0]:
            raise RuntimeError(
                "DSA sparse prefill top-k length mismatch: "
                f"lens={topk_lens.shape[0]}, q_tokens={q.shape[0]}"
            )
        if q.shape[0] == 0:
            return q.new_empty((0, layer.tp_q_head_num * layer.v_head_dim))
        if workspace_indices.shape != (q.shape[0], self.index_topk):
            raise RuntimeError(
                "DSA sparse prefill top-k shape mismatch: "
                f"indices={tuple(workspace_indices.shape)}, "
                f"expected={(q.shape[0], self.index_topk)}"
            )
        if kv_workspace_slots is None:
            raise RuntimeError(
                "DSA sparse prefill requires kv_workspace_slots to "
                "map workspace-local top-k rows back to KV cache slots."
            )
        topk_slots = workspace_topk_to_global_slots(
            workspace_indices=workspace_indices,
            kv_workspace_slots=kv_workspace_slots,
        )
        q_view = q.view(q.shape[0], layer.tp_q_head_num, layer.head_dim)
        if self.data_type == torch.float8_e4m3fn and q_view.dtype != self.data_type:
            q_view = q_view.to(self.data_type)
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        sparse_kv_cache = None
        if hasattr(token_to_kv_pool, "get_sparse_decode_kv_buffer"):
            sparse_kv_cache = token_to_kv_pool.get_sparse_decode_kv_buffer(
                layer.layer_id
            )

        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        out = dsa_prefill(
            q=q_view,
            kv_cache=kv_cache,
            sparse_kv_cache=sparse_kv_cache,
            topk_slots=topk_slots,
            topk_lens=topk_lens.to(device=q.device, dtype=torch.int32).contiguous(),
            max_seqlen_k=max_seq_len,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            softmax_scale=layer.scaling,
            page_size=self.page_size,
            logit_cap=layer.logit_cap,
            k_scale=k_scale,
        )
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_sparse_decode(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor | None,
        decode_plan: DSADecodePlan,
    ) -> torch.Tensor:
        if self.page_size != 64:
            raise RuntimeError(
                "DSA sparse decode currently requires page_size=64 for "
                f"sparse KV layout, got {self.page_size}."
            )
        if getattr(token_to_kv_pool, "quant_method", None) == "per_token_head":
            raise RuntimeError(
                "DSA sparse decode does not support "
                "kv_cache_quant_method='per_token_head' yet."
            )
        allow_fp8_query = (
            getattr(self, "data_type", torch.bfloat16) == torch.float8_e4m3fn
            and q.dtype == torch.float8_e4m3fn
        )
        if q.dtype != torch.bfloat16 and not allow_fp8_query:
            raise RuntimeError(
                "DSA sparse decode requires BF16 query tensors, or FP8 query "
                f"tensors on FP8 KV sparse paths, got {q.dtype}."
            )
        if save_kv_cache:
            assert k is not None
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : self.kv_lora_rank],
                k[..., self.kv_lora_rank :],
            )

        if topk_indices.dtype != torch.int32:
            topk_indices = topk_indices.to(torch.int32)
        if topk_indices.shape[-1] != self.index_topk:
            raise RuntimeError(
                "DSA sparse decode top-k width mismatch: "
                f"indices={topk_indices.shape[-1]}, expected={self.index_topk}"
            )
        num_tokens = q.shape[0]
        if bs != decode_plan.num_requests:
            raise RuntimeError(
                "DSA sparse decode request count differs from its plan: "
                f"bs={bs}, planned={decode_plan.num_requests}."
            )
        if num_tokens != decode_plan.num_tokens:
            raise RuntimeError(
                "DSA sparse decode token count differs from its plan: "
                f"q_tokens={num_tokens}, planned={decode_plan.num_tokens}."
            )
        if decode_plan.seq_lens.numel() != decode_plan.num_requests:
            raise RuntimeError(
                "DSA sparse decode plan has inconsistent sequence lengths: "
                f"seq_lens={decode_plan.seq_lens.numel()}, "
                f"requests={decode_plan.num_requests}."
            )
        if topk_indices.shape[0] != num_tokens:
            raise RuntimeError(
                "DSA sparse decode top-k token mismatch: "
                f"indices={topk_indices.shape[0]}, q_tokens={num_tokens}."
            )
        if topk_lens is not None:
            if topk_lens.dim() != 1 or topk_lens.numel() != num_tokens:
                raise RuntimeError(
                    "DSA sparse decode top-k length mismatch: "
                    f"lens={tuple(topk_lens.shape)}, q_tokens={num_tokens}."
                )
            topk_lens = topk_lens.to(device=q.device, dtype=torch.int32).contiguous()

        q_view = q.view(num_tokens, layer.tp_q_head_num, layer.head_dim)
        if self.data_type == torch.float8_e4m3fn:
            q_view = q_view.to(self.data_type)
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        sparse_kv_cache = None
        if hasattr(token_to_kv_pool, "get_sparse_decode_kv_buffer"):
            sparse_kv_cache = token_to_kv_pool.get_sparse_decode_kv_buffer(
                layer.layer_id
            )

        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        out = dsa_decode(
            q=q_view,
            kv_cache=kv_cache,
            sparse_kv_cache=sparse_kv_cache,
            topk_slots=topk_indices.view(num_tokens, -1),
            topk_lens=topk_lens,
            max_seqlen_k=decode_plan.max_seq_len,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            softmax_scale=layer.scaling,
            page_size=self.page_size,
            q_len_per_req=decode_plan.q_len_per_req,
            logit_cap=layer.logit_cap,
            k_scale=k_scale,
        )
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)


register_backend("dsa", {AttentionArch.DSA}, DSABackend)
