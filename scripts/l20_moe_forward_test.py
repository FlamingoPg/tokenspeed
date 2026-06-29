#!/usr/bin/env python3
"""End-to-end MoE forward verification on L20.

Constructs a tiny DeepSeek-V3 model config + random FP8 weights, runs one
forward pass through the MoE layer, and verifies the triton_fp8_block_moe_apply
kernel produces correct output vs an eager reference.

This bypasses the full server (tokenizer/HTTP/scheduler) to isolate the MoE
kernel's correctness in a real model forward context.
"""
from __future__ import annotations

import sys
import os

# Assume run from workspace root with tokenspeed-kernel on PYTHONPATH.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tokenspeed-kernel", "python"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenspeed_kernel.ops.moe.triton.fp8 import (
    triton_fp8_block_moe_apply,
    triton_fp8_moe_weights,
    _dequant_fp8_block,
)


def main():
    torch.manual_seed(42)
    device = "cuda"

    # Tiny DeepSeek-V3-like MoE dimensions
    E = 8           # experts (small for fast test)
    M = 32          # tokens
    H = 512         # hidden size
    N = 1024        # intermediate size (per gate/up)
    top_k = 2
    BN, BK = 128, 128

    print(f"=== Tiny MoE: E={E} M={M} H={H} N={N} top_k={top_k} ===")

    class FakeMoE(nn.Module):
        def __init__(self):
            super().__init__()
            # FP8 block-scaled weights (DeepSeek-V3 format)
            self.register_parameter(
                "w13_weight",
                nn.Parameter(
                    torch.randn(E, 2 * N, H, device=device, dtype=torch.bfloat16).to(
                        torch.float8_e4m3fn
                    ),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "w13_weight_scale_inv",
                nn.Parameter(
                    torch.ones(E, 2 * N // BN, H // BK, device=device, dtype=torch.float32) * 0.1,
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "w2_weight",
                nn.Parameter(
                    torch.randn(E, H, N, device=device, dtype=torch.bfloat16).to(
                        torch.float8_e4m3fn
                    ),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "w2_weight_scale_inv",
                nn.Parameter(
                    torch.ones(E, H // BN, N // BK, device=device, dtype=torch.float32) * 0.1,
                    requires_grad=False,
                ),
            )
            self.top_k = top_k
            self.ep_size = 1
            self.ep_rank = 0
            self.tp_size = 1
            self.tp_rank = 0

    w = FakeMoE()

    # --- Build reference from original FP8 weights (before dequant) ---
    w13_dq_ref = _dequant_fp8_block(w.w13_weight, w.w13_weight_scale_inv)
    w2_dq_ref = _dequant_fp8_block(w.w2_weight, w.w2_weight_scale_inv)

    x = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    topk_ids = torch.randint(0, E, (M, top_k), device=device)
    topk_weights = F.softmax(
        torch.randn(M, top_k, device=device, dtype=torch.float32), dim=-1
    )

    print("Computing eager reference...")
    out_ref = torch.zeros(M, H, device=device, dtype=torch.bfloat16)
    for t in range(M):
        for k in range(top_k):
            e = int(topk_ids[t, k])
            wt = float(topk_weights[t, k])
            gate = x[t].float() @ w13_dq_ref[e, :N, :].T.float()
            up = x[t].float() @ w13_dq_ref[e, N:, :].T.float()
            inter = F.silu(gate) * up
            fc2 = inter @ w2_dq_ref[e].T.float()
            out_ref[t] += (fc2 * wt).to(torch.bfloat16)

    # --- Dequant weights via preprocessor ---
    print("Running weight preprocessor (FP8 dequant)...")
    triton_fp8_moe_weights(plan={}, w=w)
    assert w.w13_weight.dtype == torch.bfloat16, "dequant failed"

    # --- Run kernel ---
    print("Running triton_fp8_block_moe_apply (flashinfer cutlass bf16)...")
    out_kernel = triton_fp8_block_moe_apply(
        plan={},
        x=x,
        w=w,
        router_logits=torch.randn(M, E, device=device, dtype=torch.float32),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    # --- Compare ---
    diff = (out_kernel.float() - out_ref.float()).abs()
    max_abs = diff.max().item()
    rel = max_abs / (out_ref.float().abs().max().item() + 1e-9)
    print(f"\nResults:")
    print(f"  max abs diff:  {max_abs:.4f}")
    print(f"  max rel diff:  {rel:.4f}")
    print(f"  kernel[0,:4]:  {[round(v, 3) for v in out_kernel[0, :4].float().tolist()]}")
    print(f"  ref[0,:4]:     {[round(v, 3) for v in out_ref[0, :4].float().tolist()]}")

    if rel < 0.05:
        print(f"\nPASS: MoE forward correct on L20 (rel diff {rel:.2%} < 5%)")
        return 0
    else:
        print(f"\nFAIL: numerical mismatch (rel diff {rel:.2%})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
