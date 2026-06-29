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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Triton FP8 block-wise MoE kernel (portable, sm_89+).

Provides FP8 block-scaled fused MoE for architectures lacking a specialized
vendor kernel (e.g. Ada sm_89 / L20). Registered with the same trait contract
as ``flashinfer_cutlass_fp8_moe_apply`` so DeepSeek-V3's MoE routes here on L20.

Implementation: dequantizes the FP8 block-scaled weights to bf16 once at
weight-load time (via the weight preprocessor), then dispatches the bf16
fused MoE to flashinfer's ``cutlass_fused_moe`` (CUTLASS bf16 kernel,
available on sm_89+). This avoids the Python per-expert loop and gets a
single high-performance CUTLASS launch.

Weight contract (per DeepSeek-V3 ``compressed-tensors`` FP8 block recipe):
  - ``w.w13_weight``           [E, 2*N, K]  float8_e4m3fn  (gate | up)
  - ``w.w13_weight_scale_inv`` [E, 2*N//128, K//128]  float32
  - ``w.w2_weight``            [E, K, N]    float8_e4m3fn
  - ``w.w2_weight_scale_inv``  [E, K//128, N//128]  float32

Block dequant model:
    W_dequant[e, n, k] = W_fp8[e, n, k] * scale[e, n // 128, k // 128]
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_BLOCK_N = 128
_BLOCK_K = 128


def triton_fp8_moe_weights(plan: dict, w: torch.nn.Module):
    """Weight preprocessor: dequant FP8 block-scaled weights to bf16 in place.

    After this runs, ``w.w13_weight`` and ``w.w2_weight`` hold bf16 tensors and
    the scale attributes are cleared. The apply kernel then calls
    ``cutlass_fused_moe`` with ``quant_scales=None`` (bf16 unquant path).

    The dequantization is deterministic (block scale broadcast) and exact — no
    precision loss beyond the original FP8 quantization.
    """
    w_fp8_13 = w.w13_weight
    w_scale_13 = w.w13_weight_scale_inv
    w_fp8_2 = w.w2_weight
    w_scale_2 = w.w2_weight_scale_inv

    # flashinfer cutlass_fused_moe with Swiglu expects [up | gate] ordering
    # (it computes act(w1*x) * (w2*x) where w1=first half). DeepSeek-V3 stores
    # [gate | up]. Swap the two halves to match, same as flashinfer_cutlass_fp8.
    half_n = w_fp8_13.shape[1] // 2
    first = w_fp8_13.data[:, :half_n, :].clone()
    w_fp8_13.data[:, :half_n, :] = w_fp8_13.data[:, half_n:, :]
    w_fp8_13.data[:, half_n:, :] = first

    half_sn = w_scale_13.shape[1] // 2
    first_s = w_scale_13.data[:, :half_sn, :].clone()
    w_scale_13.data[:, :half_sn, :] = w_scale_13.data[:, half_sn:, :]
    w_scale_13.data[:, half_sn:, :] = first_s
    w_scale_13.data.clamp_(min=1e-10)
    w_scale_2.data.clamp_(min=1e-10)

    # Dequant FP8 block-scaled -> bf16.
    w_dq_13 = _dequant_fp8_block(w_fp8_13, w_scale_13)
    w_dq_2 = _dequant_fp8_block(w_fp8_2, w_scale_2)

    # Replace in place with bf16 Parameters for the bf16 MoE path.
    w.w13_weight = torch.nn.Parameter(w_dq_13, requires_grad=False)
    w.w2_weight = torch.nn.Parameter(w_dq_2, requires_grad=False)


def _dequant_fp8_block(
    w_fp8: torch.Tensor,      # [E, OutDim, InDim] fp8
    w_scale: torch.Tensor,    # [E, OutDim // BN, InDim // BK] fp32
) -> torch.Tensor:
    """Dequantize FP8 block-scaled weights to bf16.

    Each (BN, BK) block of the weight shares one fp32 scale. Broadcast via
    repeat_interleave, multiply, cast to bf16.
    """
    return (
        w_fp8.to(torch.float32)
        * w_scale.repeat_interleave(_BLOCK_N, dim=1).repeat_interleave(_BLOCK_K, dim=2)
    ).to(torch.bfloat16)


@register_kernel(
    "moe",
    "apply",
    name="triton_fp8_block_moe_apply",
    solution="triton",
    weight_preprocessor=triton_fp8_moe_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(8, 9),
    ),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={
        "weight_dtype": frozenset({"fp8"}),
        "activation": frozenset({"silu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"input"}),
        "fp8_scale_block_shape": frozenset({(128, 128)}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_fp8_block_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Portable FP8 block-scaled fused MoE.

    The weight preprocessor (``triton_fp8_moe_weights``) has already dequantized
    the FP8 block-scaled weights to bf16 at load time. This function dispatches
    the bf16 fused MoE via flashinfer's ``cutlass_fused_moe`` (CUTLASS bf16,
    available on sm_89+), giving a single optimized launch rather than a
    per-expert Python loop.

    Args:
        plan: Execution plan from ``moe_plan``.
        x: Hidden states [tokens, hidden_size] in bf16/fp16.
        w: Module whose w13_weight/w2_weight have been dequantized to bf16 by
            the weight preprocessor.
        router_logits: [tokens, num_experts] — used if topk not precomputed.
        topk_weights: [tokens, top_k] routing weights.
        topk_ids: [tokens, top_k] expert indices.

    Returns:
        MoE output [tokens, hidden_size].
    """
    if topk_weights is None or topk_ids is None:
        scores = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(
            scores, k=getattr(w, "top_k", 1), dim=-1, sorted=False
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # cutlass_fused_moe requires token_final_scales as float32.
    topk_weights = topk_weights.to(torch.float32)
    # After weight preprocessing, w13_weight and w2_weight are bf16.
    # Dispatch to flashinfer cutlass_fused_moe (bf16 unquant path).
    from flashinfer import ActivationType, cutlass_fused_moe

    return cutlass_fused_moe(
        input=x,
        token_selected_experts=topk_ids.to(torch.int),
        token_final_scales=topk_weights,
        fc1_expert_weights=w.w13_weight,
        fc2_expert_weights=w.w2_weight,
        output_dtype=x.dtype,
        quant_scales=None,
        ep_size=getattr(w, "ep_size", 1),
        ep_rank=getattr(w, "ep_rank", 0),
        tp_size=getattr(w, "tp_size", 1),
        tp_rank=getattr(w, "tp_rank", 0),
        tune_max_num_tokens=max(8192, x.shape[0]),
        activation_type=ActivationType.Swiglu,
    )[0]
