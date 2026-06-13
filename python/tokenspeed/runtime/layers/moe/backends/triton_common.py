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

import functools
from functools import partial

import tokenspeed_kernel
import torch
import triton.language as tl
from torch import nn

from tokenspeed.runtime.layers.moe.backends.triton_config import (
    try_get_optimal_moe_config,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.utils.env import envs

__all__ = [
    "TritonMoEWorkspace",
    "support_tensor_descriptor",
    "triton_forward",
]

padding_size = 128 if envs.TOKENSPEED_MOE_PADDING.get() else 0
_DEFAULT_TRITON_MOE_WORKSPACE_CACHE_BYTES = 64 * 1024 * 1024


def _workspace_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    numel = functools.reduce(lambda lhs, rhs: lhs * rhs, shape, 1)
    return numel * torch.empty((), dtype=dtype).element_size()


class TritonMoEWorkspace:
    """Per-layer scratch buffers for the Triton MoE path."""

    __slots__ = ("_buffers", "_max_cache_bytes", "_retired")

    def __init__(
        self,
        max_cache_bytes: int = _DEFAULT_TRITON_MOE_WORKSPACE_CACHE_BYTES,
    ) -> None:
        self._buffers: dict[str, torch.Tensor] = {}
        self._max_cache_bytes = int(max_cache_bytes)
        self._retired: list[torch.Tensor] = []

    def empty(
        self,
        name: str,
        shape,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shape = tuple(int(dim) for dim in shape)
        if not shape:
            raise ValueError("workspace shape must have at least one dimension")

        capacity_shape = (max(1, shape[0]), *shape[1:])
        if _workspace_nbytes(capacity_shape, dtype) > self._max_cache_bytes:
            return torch.empty(capacity_shape, device=device, dtype=dtype)[: shape[0]]

        buffer = self._buffers.get(name)
        if (
            buffer is None
            or buffer.device != device
            or buffer.dtype != dtype
            or buffer.dim() != len(shape)
            or tuple(buffer.shape[1:]) != shape[1:]
            or buffer.shape[0] < shape[0]
        ):
            if buffer is not None:
                # A captured CUDA graph may still reference the old buffer
                # (decode/verify graphs capture MoE scratch at their batch
                # size while eager prefill regrows it much larger). Keep the
                # old allocation alive so replays never write into memory the
                # allocator has handed to someone else.
                self._retired.append(buffer)
            buffer = torch.empty(capacity_shape, device=device, dtype=dtype)
            self._buffers[name] = buffer
        return buffer[: shape[0]]


def build_triton_gemms(
    layer: nn.Module,
    spec: MoELayerSpec,
    *,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    block_shape=None,
    dtype_tag: str = "bf16",
    gate_up_B_scale=None,
    down_B_scale=None,
):
    num_local_experts, intermediate_size_x2, hidden_size = layer.w13_weight.shape
    intermediate_size = intermediate_size_x2 // 2

    common = dict(
        compute_type=tl.bfloat16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        filter_expert=True,
    )
    _experts_common = dict(
        **common,
        dtype=torch.bfloat16,
        features={"dispatch_sorted"},
        expected_kernel_name="triton_moe_fused_experts",
    )
    gemm = partial(tokenspeed_kernel.moe_experts, **_experts_common)
    gate_up_gemm = partial(
        gemm,
        A_scale=None,
        B_scale=gate_up_B_scale,
        mul_routed_weight=False,
        top_k=spec.top_k,
    )
    down_gemm = partial(
        gemm,
        A_scale=None,
        B_scale=down_B_scale,
        mul_routed_weight=True,
        top_k=1,
    )
    get_config_func = partial(
        try_get_optimal_moe_config,
        (num_local_experts, intermediate_size * 2, hidden_size),
        (num_local_experts, hidden_size, intermediate_size),
        spec.top_k,
        dtype_tag,
        block_shape=None,
        return_down_config=True,
    )
    return gate_up_gemm, down_gemm, get_config_func


def _should_skip_zero_weight_tiny_routes(
    *,
    ep_size: int,
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> bool:
    return (
        ep_size > 1
        and topk_ids.is_cuda
        and topk_ids.numel() > 0
        and topk_ids.numel() <= 32
        and num_experts <= 32
        and block_size <= 64
    )


def _should_use_tiny_dispatch_workspace(
    *,
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> bool:
    return (
        topk_ids.is_cuda
        and topk_ids.numel() > 0
        and topk_ids.numel() <= 32
        and num_experts <= 32
        and block_size <= 64
    )


def _dispatch_workspace_sizes(
    *,
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[int, int]:
    if _should_use_tiny_dispatch_workspace(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
    ):
        max_num_tokens_padded = min(num_experts, topk_ids.numel()) * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    return max_num_tokens_padded, max_num_m_blocks


def _localize_topk_for_ep(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    ep_rank: int,
    num_local_experts: int,
    skip_zero_weight_tiny_routes: bool,
    workspace: TritonMoEWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    nonlocal_expert_id = -1 if skip_zero_weight_tiny_routes else 0
    local_topk_ids = None
    local_topk_weights = None
    if workspace is not None:
        local_topk_ids = workspace.empty(
            "local_topk_ids",
            topk_ids.shape,
            device=topk_ids.device,
            dtype=topk_ids.dtype,
        )
        local_topk_weights = workspace.empty(
            "local_topk_weights",
            topk_weights.shape,
            device=topk_weights.device,
            dtype=topk_weights.dtype,
        )
    if topk_ids.is_cuda and topk_weights.is_cuda:
        return tokenspeed_kernel.moe_localize_topk(
            topk_ids,
            topk_weights,
            ep_rank,
            num_local_experts,
            nonlocal_expert_id=nonlocal_expert_id,
            local_topk_ids=local_topk_ids,
            local_topk_weights=local_topk_weights,
            dtype=topk_ids.dtype,
            expected_kernel_name="triton_moe_localize_topk",
        )

    local_expert_start = ep_rank * num_local_experts
    local_expert_end = local_expert_start + num_local_experts
    local_expert_mask = (topk_ids >= local_expert_start) & (topk_ids < local_expert_end)
    computed_topk_ids = torch.where(
        local_expert_mask,
        topk_ids - local_expert_start,
        torch.full_like(topk_ids, nonlocal_expert_id),
    )
    computed_topk_weights = torch.where(
        local_expert_mask,
        topk_weights,
        torch.zeros_like(topk_weights),
    )
    if local_topk_ids is not None and local_topk_weights is not None:
        local_topk_ids.copy_(computed_topk_ids)
        local_topk_weights.copy_(computed_topk_weights)
        return local_topk_ids, local_topk_weights
    return computed_topk_ids, computed_topk_weights


def triton_forward(
    gate_up_gemm,
    down_gemm,
    get_config_func,
    activation: str,
    layer: nn.Module,
    hidden_states: torch.Tensor,
    topk_output: object,
    *,
    workspace: TritonMoEWorkspace | None = None,
) -> torch.Tensor:
    from tokenspeed.runtime.layers.activation import silu_and_mul

    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    m_tokens = hidden_states.shape[0]
    num_experts, intermediate_size_x2, hidden_size = layer.w13_weight.shape
    top_k = topk_ids.shape[1]
    dtype = hidden_states.dtype
    device = hidden_states.device

    config, (down_config, _max_block_m) = get_config_func(M=m_tokens)
    config = dict(config)
    down_config = dict(down_config)

    gate_up_moe_use_tma = config.pop("USE_TMA", False)
    down_moe_use_tma = down_config.pop("USE_TMA", False)

    scratch = workspace or TritonMoEWorkspace()
    dispatch_block_size = config["BLOCK_SIZE_M"]
    skip_zero_weight_tiny_routes = _should_skip_zero_weight_tiny_routes(
        ep_size=getattr(layer, "ep_size", 1),
        topk_ids=topk_ids,
        block_size=dispatch_block_size,
        num_experts=num_experts,
    )

    ep_size = getattr(layer, "ep_size", 1)
    if ep_size > 1:
        num_local_experts_for_ep = getattr(
            layer, "num_local_experts", layer.w13_weight.shape[0]
        )
        topk_ids, topk_weights = _localize_topk_for_ep(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            ep_rank=getattr(layer, "ep_rank", 0),
            num_local_experts=num_local_experts_for_ep,
            skip_zero_weight_tiny_routes=skip_zero_weight_tiny_routes,
            workspace=scratch,
        )

    max_num_tokens_padded, max_num_m_blocks = _dispatch_workspace_sizes(
        topk_ids=topk_ids,
        block_size=dispatch_block_size,
        num_experts=num_experts,
    )
    dispatch_sorted_ids = scratch.empty(
        "dispatch_sorted_ids",
        (max_num_tokens_padded,),
        device=device,
        dtype=torch.int32,
    )
    dispatch_expert_ids = scratch.empty(
        "dispatch_expert_ids",
        (max_num_m_blocks,),
        device=device,
        dtype=torch.int32,
    )
    dispatch_num_tokens_post_pad = scratch.empty(
        "dispatch_num_tokens_post_pad",
        (1,),
        device=device,
        dtype=torch.int32,
    )
    dispatch_cumsum_buffer = scratch.empty(
        "dispatch_cumsum_buffer",
        (num_experts + 2,),
        device=device,
        dtype=torch.int32,
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        tokenspeed_kernel.moe_dispatch(
            topk_ids,
            dispatch_block_size,
            num_experts,
            dtype=torch.int32,
            topk_weights=topk_weights if skip_zero_weight_tiny_routes else None,
            sorted_ids=dispatch_sorted_ids,
            expert_ids=dispatch_expert_ids,
            num_tokens_post_pad=dispatch_num_tokens_post_pad,
            cumsum_buffer=dispatch_cumsum_buffer,
            expected_kernel_name="triton_moe_align_block_size",
        )
    )

    max_num_active_experts = min(m_tokens * top_k, num_experts + 1)
    padded_tokens = (
        max_num_active_experts * (config["BLOCK_SIZE_M"] - 1) if down_moe_use_tma else 0
    )
    intermediate_cache1 = scratch.empty(
        "intermediate_cache1",
        (m_tokens * top_k + padded_tokens, intermediate_size_x2),
        device=device,
        dtype=dtype,
    )
    if skip_zero_weight_tiny_routes:
        # The skip-zero-weight tiny-dispatch path localizes non-local routes to
        # expert_id=-1, and the fused gate-up GEMM (filter_expert=True) does not
        # write those rows. silu_and_mul below reads the full cache, so the
        # skipped rows would otherwise be read uninitialized (compute-sanitizer
        # initcheck flags this; the garbage -- e.g. 3.4e38/NaN -- can propagate
        # downstream and corrupt routing into an out-of-range expert index ->
        # illegal memory access). Zero the scratch first; this only runs on the
        # tiny path (<=32 rows) so the memset is negligible.
        intermediate_cache1.zero_()
    intermediate_cache2 = scratch.empty(
        "intermediate_cache2",
        (m_tokens * top_k + padded_tokens, intermediate_size_x2 // 2),
        device=device,
        dtype=dtype,
    )
    intermediate_cache3 = scratch.empty(
        "intermediate_cache3",
        (m_tokens, top_k, hidden_size),
        device=device,
        dtype=dtype,
    )
    out_hidden_states = scratch.empty(
        "out_hidden_states",
        hidden_states.shape,
        device=device,
        dtype=dtype,
    )

    gate_up_gemm(
        A=hidden_states,
        B=layer.w13_weight,
        bias=None,
        C=intermediate_cache1,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        config=config,
        a_use_tma=False,
        b_use_tma=gate_up_moe_use_tma,
        c_sorted=down_moe_use_tma,
    )

    if activation == "silu":
        silu_and_mul(
            intermediate_cache1.view(-1, intermediate_size_x2),
            intermediate_cache2,
        )
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    down_gemm(
        A=intermediate_cache2,
        B=layer.w2_weight,
        bias=None,
        C=intermediate_cache3,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        config=down_config,
        a_use_tma=down_moe_use_tma,
        b_use_tma=down_moe_use_tma,
    )

    # Current limitation: Should avoid using runtime shapes as traits
    expected_combine_kernel = (
        "torch_compile_moe_sum_reduce" if m_tokens <= 32 else "triton_moe_sum_reduce"
    )
    routed_scaling_factor = 1.0
    if skip_zero_weight_tiny_routes:
        tokenspeed_kernel.moe_combine(
            intermediate_cache3,
            out_hidden_states,
            routed_scaling_factor,
            topk_weights,
            dtype=dtype,
            traits={
                "num_tokens": m_tokens,
                "comm_strategy": None,
                "skip_zero_weights": True,
            },
            expected_kernel_name="triton_moe_sum_reduce_skip_zero_weights",
        )
    else:
        tokenspeed_kernel.moe_combine(
            intermediate_cache3,
            out_hidden_states,
            routed_scaling_factor,
            dtype=dtype,
            traits={
                "num_tokens": m_tokens,
                "comm_strategy": None,
                "skip_zero_weights": False,
            },
            expected_kernel_name=expected_combine_kernel,
        )
    return out_hidden_states
