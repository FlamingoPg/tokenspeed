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

import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8
from torch import nn

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.triton_common import (
    TritonMoEWorkspace,
    build_triton_gemms,
    triton_forward,
)
from tokenspeed.runtime.layers.moe.backends.triton_weights import (
    attach_dense_weight_pair,
    register_block_scale_inverses,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.quantization import Fp8Config

try:
    from tokenspeed_kernel.ops.gemm.deep_gemm import (
        m_grouped_fp8_gemm_nt_contiguous,
        m_grouped_fp8_gemm_nt_masked,
    )
except ImportError:
    m_grouped_fp8_gemm_nt_contiguous = None  # type: ignore[assignment]
    m_grouped_fp8_gemm_nt_masked = None  # type: ignore[assignment]

_DEEP_GEMM_FP8_GROUPED_AVAILABLE = (
    m_grouped_fp8_gemm_nt_contiguous is not None
    and m_grouped_fp8_gemm_nt_masked is not None
)
_DEEP_GEMM_ACTIVATION_BLOCK_SIZE = 128
_DEEP_GEMM_EXPERT_ALIGNMENT = 128
_DEEP_GEMM_GATHER_BLOCK_D = 1024
_DEEP_GEMM_MASKED_MAX_TOKENS = 4
_DEFAULT_TRITON_DECODE_THRESHOLD = 4


def _align_count(value: int, alignment: int = _DEEP_GEMM_EXPERT_ALIGNMENT) -> int:
    if value <= 0:
        return 0
    return ((value + alignment - 1) // alignment) * alignment


def _localize_topk_for_ep(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    ep_rank: int,
    ep_size: int,
    num_local_experts: int,
    workspace: TritonMoEWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    if ep_size <= 1:
        return topk_ids, topk_weights

    local_topk_ids = None
    local_topk_weights = None
    if workspace is not None:
        local_topk_ids = workspace.empty(
            "deep_gemm_local_topk_ids",
            topk_ids.shape,
            device=topk_ids.device,
            dtype=topk_ids.dtype,
        )
        local_topk_weights = workspace.empty(
            "deep_gemm_local_topk_weights",
            topk_weights.shape,
            device=topk_weights.device,
            dtype=topk_weights.dtype,
        )
    return tokenspeed_kernel.moe_localize_topk(
        topk_ids,
        topk_weights,
        ep_rank,
        num_local_experts,
        nonlocal_expert_id=-1,
        local_topk_ids=local_topk_ids,
        local_topk_weights=local_topk_weights,
        dtype=topk_ids.dtype,
        expected_kernel_name="triton_moe_localize_topk",
    )


def _aligned_local_expert_token_counts(
    topk_ids: torch.Tensor,
    *,
    num_local_experts: int,
    alignment: int = _DEEP_GEMM_EXPERT_ALIGNMENT,
) -> list[int]:
    if num_local_experts <= 0:
        raise ValueError(f"num_local_experts must be positive, got {num_local_experts}")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    if topk_ids.numel() == 0:
        return [0] * num_local_experts

    valid_topk_ids = topk_ids[topk_ids >= 0]
    if valid_topk_ids.numel() == 0:
        return [0] * num_local_experts

    max_expert_id = int(valid_topk_ids.max().item())
    if max_expert_id >= num_local_experts:
        raise ValueError(
            f"local top-k ids must be smaller than {num_local_experts}, "
            f"got {max_expert_id}"
        )

    counts = torch.bincount(
        valid_topk_ids.to(torch.long),
        minlength=num_local_experts,
    )[:num_local_experts]
    return [_align_count(int(count), alignment) for count in counts.cpu().tolist()]


def _masked_local_route_metadata(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be [tokens, topk], got {topk_ids.shape}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights must have the same shape as topk_ids, got "
            f"{topk_weights.shape} and {topk_ids.shape}"
        )

    num_tokens, top_k = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).to(torch.long)
    flat_weights = topk_weights.reshape(-1).to(torch.float32)
    flat_token_ids = torch.arange(
        num_tokens,
        device=topk_ids.device,
        dtype=torch.long,
    ).repeat_interleave(top_k)

    active_mask = flat_ids >= 0
    expert_ids = flat_ids[active_mask]
    token_ids = flat_token_ids[active_mask]
    route_weights = flat_weights[active_mask]

    if expert_ids.numel() > 0:
        order = torch.argsort(expert_ids)
        expert_ids = expert_ids[order]
        token_ids = token_ids[order]
        route_weights = route_weights[order]

    counts = torch.bincount(
        expert_ids,
        minlength=num_local_experts,
    )[
        :num_local_experts
    ].to(torch.int32)
    expert_starts = torch.cumsum(counts.to(torch.long), dim=0) - counts.to(torch.long)
    route_positions = torch.arange(
        expert_ids.numel(), device=topk_ids.device, dtype=torch.long
    ) - expert_starts.index_select(0, expert_ids)
    return expert_ids, token_ids, route_positions, route_weights, counts


def _quantize_fp8_token_major(
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states_fp8, input_scales = per_token_group_quant_fp8(
        hidden_states,
        _DEEP_GEMM_ACTIVATION_BLOCK_SIZE,
        column_major_scales=True,
        scale_tma_aligned=True,
    )
    return hidden_states_fp8, input_scales.contiguous()


class Fp8DeepGemmBackend(MoEBackend):
    supported_arches = frozenset({"sm90", "sm100"})

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._triton_workspace = TritonMoEWorkspace()

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        return (
            _DEEP_GEMM_FP8_GROUPED_AVAILABLE
            and isinstance(quant_config, Fp8Config)
            and tuple(quant_config.weight_block_size or ()) == (128, 128)
            and spec.activation == "silu"
            and spec.tp_size == 1
            and spec.top_k > 0
            and spec.hidden_size % _DEEP_GEMM_GATHER_BLOCK_D == 0
            and spec.intermediate_size % _DEEP_GEMM_ACTIVATION_BLOCK_SIZE == 0
        )

    def create_layer_weights(
        self, layer: nn.Module, *, with_bias: bool = False
    ) -> None:
        ispp = attach_dense_weight_pair(
            self,
            layer,
            with_bias=with_bias,
            params_dtype=torch.float8_e4m3fn,
        )
        register_block_scale_inverses(
            self,
            layer,
            num_local_experts=self.spec.num_local_experts,
            hidden_size=self.spec.hidden_size,
            intermediate_size_per_partition=ispp,
            block_shape=self.quant_config.weight_block_size,
        )
        self._gate_up_gemm, self._down_gemm, self._get_config_func = build_triton_gemms(
            layer,
            self.spec,
            use_fp8_w8a8=True,
            block_shape=self.quant_config.weight_block_size,
            dtype_tag="fp8_w8a8",
            gate_up_B_scale=layer.w13_weight_scale_inv,
            down_B_scale=layer.w2_weight_scale_inv,
        )
        self._executor = None

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self._executor = layer.construct_executor()

    def _get_executor(self, layer: nn.Module):
        if self._executor is None:
            self._executor = layer.construct_executor()
        return self._executor

    def _forward_triton_decode(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        topk_output: object,
    ) -> torch.Tensor:
        return triton_forward(
            self._gate_up_gemm,
            self._down_gemm,
            self._get_config_func,
            layer.activation,
            layer,
            hidden_states,
            topk_output,
            workspace=self._triton_workspace,
        )

    def _forward_masked_local(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        top_k = topk_ids.shape[1]
        num_local_experts = layer.w13_weight.shape[0]
        max_tokens_per_expert = max(1, num_tokens * top_k)

        (
            expert_ids,
            token_ids,
            route_positions,
            route_weights,
            masked_m,
        ) = _masked_local_route_metadata(
            topk_ids,
            topk_weights,
            num_local_experts=num_local_experts,
        )
        hidden_states_fp8, input_scales = _quantize_fp8_token_major(hidden_states)
        packed_hidden = hidden_states_fp8.new_zeros(
            num_local_experts,
            max_tokens_per_expert,
            hidden_size,
        )
        packed_scales = input_scales.new_zeros(
            num_local_experts,
            max_tokens_per_expert,
            input_scales.shape[-1],
        )
        packed_hidden[expert_ids, route_positions] = hidden_states_fp8[token_ids]
        packed_scales[expert_ids, route_positions] = input_scales[token_ids]

        expert_output = self._get_executor(layer).forward_low_latency(
            packed_hidden,
            packed_scales,
            masked_m,
            expected_m=max_tokens_per_expert,
        )
        output = torch.zeros_like(hidden_states)
        routed_output = expert_output[expert_ids, route_positions].to(
            hidden_states.dtype
        ) * route_weights.to(hidden_states.dtype).unsqueeze(-1)
        output.index_add_(0, token_ids, routed_output)
        return output

    def forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        topk_output: object,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        del num_global_tokens, max_num_tokens_per_gpu

        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)
        assert hidden_states.is_contiguous(), "hidden_states must be contiguous"

        if hidden_states.shape[0] <= _DEFAULT_TRITON_DECODE_THRESHOLD:
            return self._forward_triton_decode(
                layer,
                hidden_states,
                topk_output,
            )

        topk_ids, topk_weights = _localize_topk_for_ep(
            topk_output.topk_ids,
            topk_output.topk_weights,
            ep_rank=getattr(layer, "ep_rank", 0),
            ep_size=getattr(layer, "ep_size", 1),
            num_local_experts=getattr(
                layer,
                "num_local_experts",
                layer.w13_weight.shape[0],
            ),
            workspace=self._triton_workspace,
        )
        if hidden_states.shape[0] <= _DEEP_GEMM_MASKED_MAX_TOKENS:
            return self._forward_masked_local(
                layer,
                hidden_states,
                topk_ids,
                topk_weights,
            )

        num_tokens_per_expert = _aligned_local_expert_token_counts(
            topk_ids,
            num_local_experts=layer.w13_weight.shape[0],
        )
        if sum(num_tokens_per_expert) <= 0:
            return torch.zeros_like(hidden_states)

        hidden_states_fp8, input_scales = _quantize_fp8_token_major(hidden_states)
        output = self._get_executor(layer).forward_normal(
            hidden_states_fp8,
            input_scales,
            topk_ids,
            topk_weights,
            num_tokens_per_expert,
        )
        if output.dtype != hidden_states.dtype:
            output = output.to(hidden_states.dtype)
        return output


__all__ = [
    "Fp8DeepGemmBackend",
    "_aligned_local_expert_token_counts",
    "_localize_topk_for_ep",
    "_masked_local_route_metadata",
]
