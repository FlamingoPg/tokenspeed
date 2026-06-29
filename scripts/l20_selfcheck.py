"""L20 (Ada sm_89) self-check for portable MLA stack.

Run on an L20 machine AFTER installing tokenspeed-kernel:
    python scripts/l20_selfcheck.py

This script verifies three things in order:
  1. Environment: platform detects as sm_89, flash_mla/deep_gemm are unavailable.
  2. Kernel routing: MLA decode/prefill select the triton portable kernel,
     NOT FlashMLA/CuteDSL/DeepGEMM (which require sm_90+).
  3. Numerics: the portable MLA kernels produce correct results on sm_89 by
     reusing the existing test_attention.py numerics harness.

Exit code 0 = L20 portable MLA stack is viable. Non-zero = blocked, see output.
"""

from __future__ import annotations

import sys
import traceback


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"  [..]  {msg}")


def check_environment() -> bool:
    """Stage 1: confirm L20 is detected and Blackwell-only deps are absent."""
    _section("STAGE 1 — Environment detection")
    ok = True

    import torch
    from tokenspeed_kernel.platform import current_platform

    platform = current_platform()

    print(f"  device_name    = {platform.device_name}")
    print(f"  vendor         = {platform.vendor}")
    print(f"  arch_version   = {platform.arch_version}")
    print(f"  sm_count       = {platform.sm_count}")
    print(f"  total_memory   = {platform.total_memory / (1024**3):.1f} GB")
    print(f"  sm_features    = {sorted(platform.sm_features)}")
    print(f"  runtime_feats  = {sorted(platform.runtime_features)}")
    print(f"  is_blackwell   = {platform.is_blackwell}")
    print(f"  is_hopper_plus = {platform.is_hopper_plus}")
    print(f"  torch          = {torch.__version__}")
    print(f"  cuda available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        print(f"  cuda cap       = sm_{cap[0]}_{cap[1]}")

    # --- L20 assertions ---
    if not platform.is_nvidia:
        _fail("not NVIDIA vendor")
        ok = False
    if platform.arch_version.major != 8 or platform.arch_version.minor != 9:
        _fail(
            f"expected Ada sm_89, got {platform.arch_version} — "
            "this script is for L20 only"
        )
        ok = False
    else:
        _ok("detected Ada sm_89 (L20 class)")

    # FP8 tensor core must be present (Ada has it).
    if "tensor_core:f8" not in platform.sm_features:
        _fail("missing tensor_core:f8 — FP8 path will not work")
        ok = False
    else:
        _ok("FP8 tensor core available")

    # TMA / cluster / FP4 must be absent (these are sm_90+).
    for feat in ("memory:tma", "compute:cluster", "tensor_core:f4"):
        if feat in platform.sm_features:
            _fail(f"unexpected feature {feat!r} on Ada — platform bug?")
            ok = False
    _ok("no Blackwell-only features (TMA/cluster/FP4) correctly absent")

    # --- Blackwell-only deps must fall back to error_fn ---
    _info("checking Blackwell-only kernel deps fall back gracefully...")
    from tokenspeed_kernel.registry import error_fn

    from tokenspeed_kernel.ops.attention.flash_mla import (
        flash_mla_with_kvcache as fm_decode,
        flash_mla_sparse_fwd as fm_sparse,
        get_mla_metadata as fm_meta,
    )
    from tokenspeed_kernel.ops.attention.tokenspeed_mla import (
        tokenspeed_mla_decode as cutedsl_decode,
    )

    for name, fn in [
        ("flash_mla_with_kvcache", fm_decode),
        ("flash_mla_sparse_fwd", fm_sparse),
        ("get_mla_metadata", fm_meta),
        ("tokenspeed_mla_decode", cutedsl_decode),
    ]:
        if fn is error_fn:
            _ok(f"{name} -> error_fn (Blackwell kernel correctly unavailable)")
        else:
            _fail(
                f"{name} is unexpectedly available on sm_89 — "
                "it will crash at call time"
            )
            ok = False

    # deep_gemm (Hopper+) must be None.
    try:
        from tokenspeed_kernel.thirdparty import deep_gemm

        if deep_gemm is None:
            _ok("deep_gemm is None (Hopper+ GEMM correctly unavailable)")
        else:
            _fail(
                "deep_gemm imported on sm_89 — weights may route to a kernel "
                "that emits sm_90+ instructions"
            )
            ok = False
    except Exception:
        _ok("deep_gemm import failed (acceptable on Ada)")

    return ok


def check_kernel_routing() -> bool:
    """Stage 2: confirm kernel selection lands on triton portable MLA."""
    _section("STAGE 2 — MLA kernel selection routing")
    ok = True

    import torch
    from tokenspeed_kernel import load_builtin_kernels
    from tokenspeed_kernel.platform import current_platform
    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    load_builtin_kernels()
    platform = current_platform()
    bf16 = dense_tensor_format(torch.bfloat16)

    # --- mla_decode_with_kvcache ---
    sig = format_signature(q=bf16, kv_cache=bf16)
    kernel = select_kernel(
        "attention",
        "mla_decode_with_kvcache",
        sig,
        traits={
            "page_size": 4,
            "q_len": 1,
            "qk_nope_head_dim": 128,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "support_logit_cap": False,
            "return_lse": True,
        },
    )
    print(f"  mla_decode_with_kvcache -> {kernel.name}")
    if kernel.name == "triton_mla_decode_with_kvcache":
        _ok("decode routes to triton portable kernel")
    else:
        _fail(f"decode routed to {kernel.name!r}, expected triton_mla_decode_with_kvcache")
        ok = False

    # --- mla_prefill ---
    sig = format_signature(q=bf16, k=bf16, v=bf16)
    kernel = select_kernel(
        "attention",
        "mla_prefill",
        sig,
        traits={
            "qk_head_dim": 192,
            "v_head_dim": 128,
            "is_causal": True,
            "support_logit_cap": False,
            "return_lse": True,
        },
    )
    print(f"  mla_prefill              -> {kernel.name}")
    if kernel.name == "triton_mla_prefill":
        _ok("prefill routes to triton portable kernel")
    else:
        _fail(f"prefill routed to {kernel.name!r}, expected triton_mla_prefill")
        ok = False

    # --- enumerate ALL MLA candidates on this platform (sanity) ---
    _info("all mla_decode_with_kvcache candidates on this platform:")
    for spec in KernelRegistry.get().get_for_operator(
        "attention",
        "mla_decode_with_kvcache",
        platform=platform,
    ):
        print(f"    - {spec.name:40s} priority={spec.priority} solution={spec.solution}")

    return ok


def check_numerics() -> bool:
    """Stage 3: run existing MLA numerics tests on real L20 hardware."""
    _section("STAGE 3 — MLA numerics (triton portable on sm_89)")
    ok = True

    import pytest

    # Reuse the existing, reviewed numerics tests. They parametrize
    # solution=["triton"] only and carry torch-eager reference impls.
    test_file = "tokenspeed-kernel/test/ops/test_attention.py"
    ret = pytest.main(
        [
            test_file + "::test_mla_decode_with_kvcache",
            test_file + "::test_mla_prefill",
            "-v",
            "--tb=short",
            "-x",  # stop at first failure
        ]
    )

    if ret == 0:
        _ok("MLA decode + prefill numerics passed on sm_89")
    else:
        _fail(f"numerics failed (pytest exit {ret})")
        ok = False

    return ok


def main() -> int:
    print(
        "\n  TokenSpeed L20 (Ada sm_89) portable MLA self-check\n"
        "  Verifies the fallback MLA stack works without Blackwell features."
    )

    stages = [
        ("Environment detection", check_environment),
        ("MLA kernel selection routing", check_kernel_routing),
        ("MLA numerics on sm_89", check_numerics),
    ]

    results: list[tuple[str, bool]] = []
    for name, fn in stages:
        try:
            passed = fn()
        except Exception:
            traceback.print_exc()
            passed = False
        results.append((name, passed))
        if not passed:
            print(f"\n  !! Stage {name!r} failed — stopping early.")
            break

    _section("SUMMARY")
    all_ok = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_ok = all_ok and passed

    if all_ok:
        print("\n  L20 portable MLA stack is VIABLE. Proceed to server smoke test.\n")
        return 0
    print("\n  L20 portable MLA stack is BLOCKED. See failures above.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
