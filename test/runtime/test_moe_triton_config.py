import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.layers.moe.backends.triton_config import (
    _is_amd,
    get_default_config,
)


class TestMoeTritonConfigDecodeTile(unittest.TestCase):
    """Decode (M<=E) on the block-wise FP8 path must shrink the M tile.

    GLM5.1 uses block-wise FP8 (weight_block_size=(128,128)), which routes
    through the block-wise branch of ``get_default_config``. That branch used a
    fixed ``BLOCK_SIZE_M=64`` for both prefill and decode, so bs=1 decode padded
    a single token up to 64 rows and wasted 63/64 of every expert MMA tile. A
    B200 microbench (CUDA-graph replay, which is how decode actually runs) showed
    shrinking ``BLOCK_SIZE_M`` 64->16 (``GROUP_SIZE_M`` 32->1) cuts the gate_up
    fused_moe kernel ~28% and the down kernel ~20%. ``BLOCK_SIZE_K`` stays tied
    to ``block_shape[1]`` for quant alignment; only the free M dim shrinks.
    """

    COMMON = dict(
        E=32,
        inter_size=2048,
        hidden_size=6144,
        topk=8,
        dtype="fp8_w8a8",
        is_marlin=False,
    )

    def test_blockwise_decode_shrinks_m_tile(self):
        cfg = get_default_config(M=1, block_shape=[128, 128], **self.COMMON)
        self.assertEqual(cfg["BLOCK_SIZE_M"], 16)
        self.assertEqual(cfg["GROUP_SIZE_M"], 1)
        # K stays tied to block_shape[1] for quant alignment; N unchanged.
        self.assertEqual(cfg["BLOCK_SIZE_K"], 128)
        self.assertEqual(cfg["BLOCK_SIZE_N"], 128)
        # Deeper pipelining for the tiny-M gate_up GEMM (sweep: ~19%); deeper
        # than the prefill stages (3 on NVIDIA).
        self.assertEqual(cfg["num_stages"], 2 if _is_amd else 4)

    def test_blockwise_prefill_keeps_large_m_tile(self):
        cfg = get_default_config(M=4096, block_shape=[128, 128], **self.COMMON)
        self.assertEqual(cfg["BLOCK_SIZE_M"], 64)
        self.assertEqual(cfg["GROUP_SIZE_M"], 32)
        # Prefill keeps the shallower pipeline (decode bumps it to 4).
        self.assertEqual(cfg["num_stages"], 2 if _is_amd else 3)

    def test_blockwise_decode_boundary_at_E(self):
        # M == E is still decode-ish (M <= E) -> small tile.
        cfg_eq = get_default_config(M=32, block_shape=[128, 128], **self.COMMON)
        self.assertEqual(cfg_eq["BLOCK_SIZE_M"], 16)
        # One token past E uses the prefill tile.
        cfg_gt = get_default_config(M=33, block_shape=[128, 128], **self.COMMON)
        self.assertEqual(cfg_gt["BLOCK_SIZE_M"], 64)


if __name__ == "__main__":
    unittest.main()
