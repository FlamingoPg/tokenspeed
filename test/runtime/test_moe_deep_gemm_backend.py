import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import torch

from tokenspeed.runtime.layers.moe.backends import triton_common
from tokenspeed.runtime.layers.moe.backends.fp8 import deep_gemm as deep_gemm_backend
from tokenspeed.runtime.layers.moe.backends.fp8.triton import Fp8TritonBackend
from tokenspeed.runtime.layers.moe.backends.triton_common import TritonMoEWorkspace
from tokenspeed.runtime.layers.moe.core import selector
from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
from tokenspeed.runtime.layers.moe.utils import MoeBackend
from tokenspeed.runtime.layers.quantization import Fp8Config


class TestFp8DeepGemmBackend(unittest.TestCase):
    def test_triton_moe_workspace_reuses_leading_dimension_capacity(self):
        workspace = TritonMoEWorkspace()

        first = workspace.empty(
            "cache",
            (2, 4),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        same_shape = workspace.empty(
            "cache",
            (2, 4),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        bigger = workspace.empty(
            "cache",
            (4, 4),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        smaller = workspace.empty(
            "cache",
            (1, 4),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        self.assertEqual(first.shape, (2, 4))
        self.assertEqual(bigger.shape, (4, 4))
        self.assertEqual(smaller.shape, (1, 4))
        self.assertEqual(first.data_ptr(), same_shape.data_ptr())
        self.assertEqual(bigger.data_ptr(), smaller.data_ptr())

    def test_triton_moe_workspace_reallocates_when_tail_shape_changes(self):
        workspace = TritonMoEWorkspace()

        first = workspace.empty(
            "cache",
            (2, 4),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        changed_tail = workspace.empty(
            "cache",
            (2, 8),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        self.assertEqual(changed_tail.shape, (2, 8))
        self.assertNotEqual(first.data_ptr(), changed_tail.data_ptr())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dispatch_workspace_sizes_use_tiny_bound_on_cuda(self):
        topk_ids = torch.empty((1, 8), device="cuda", dtype=torch.int32)

        max_tokens, max_blocks = triton_common._dispatch_workspace_sizes(
            topk_ids=topk_ids,
            block_size=32,
            num_experts=32,
        )

        self.assertEqual(max_tokens, 256)
        self.assertEqual(max_blocks, 8)

    def test_triton_forward_uses_masked_combine_for_tiny_ep_zero_routes(self):
        config = {
            "BLOCK_SIZE_M": 1,
            "BLOCK_SIZE_N": 1,
            "BLOCK_SIZE_K": 1,
            "GROUP_SIZE_M": 1,
        }
        hidden_states = torch.ones(1, 4, dtype=torch.bfloat16)
        topk_output = SimpleNamespace(
            topk_ids=torch.tensor([[0, 2]], dtype=torch.int64),
            topk_weights=torch.tensor([[0.75, 0.25]], dtype=torch.float32),
        )
        layer = SimpleNamespace(
            w13_weight=torch.empty(2, 8, 4, dtype=torch.bfloat16),
            w2_weight=torch.empty(2, 4, 4, dtype=torch.bfloat16),
            activation="silu",
            ep_rank=0,
            ep_size=2,
            num_local_experts=2,
        )

        def get_config_func(M):
            self.assertEqual(M, 1)
            return dict(config), (dict(config), 1)

        def gate_up_gemm(**kwargs):
            torch.testing.assert_close(
                kwargs["topk_ids"],
                torch.tensor([[0, -1]], dtype=torch.int64),
            )
            torch.testing.assert_close(
                kwargs["topk_weights"],
                torch.tensor([[0.75, 0.0]], dtype=torch.float32),
            )
            kwargs["C"].fill_(1)

        def down_gemm(**kwargs):
            torch.testing.assert_close(
                kwargs["topk_ids"],
                torch.tensor([[0, -1]], dtype=torch.int64),
            )
            torch.testing.assert_close(
                kwargs["topk_weights"],
                torch.tensor([[0.75, 0.0]], dtype=torch.float32),
            )
            kwargs["C"].fill_(1)

        def moe_dispatch(*args, **kwargs):
            torch.testing.assert_close(
                args[0],
                torch.tensor([[0, -1]], dtype=torch.int64),
            )
            torch.testing.assert_close(
                kwargs["topk_weights"],
                torch.tensor([[0.75, 0.0]], dtype=torch.float32),
            )
            return (
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                torch.tensor(1, dtype=torch.int32),
            )

        def silu_and_mul(input_tensor, output_tensor):
            del input_tensor
            output_tensor.fill_(1)

        def moe_combine(input_tensor, output_tensor, *args, **kwargs):
            del input_tensor
            self.assertEqual(args[0], 1.0)
            torch.testing.assert_close(
                args[1],
                torch.tensor([[0.75, 0.0]], dtype=torch.float32),
            )
            self.assertEqual(
                kwargs["expected_kernel_name"],
                "triton_moe_sum_reduce_skip_zero_weights",
            )
            self.assertEqual(kwargs["traits"]["skip_zero_weights"], True)
            output_tensor.fill_(2)

        with (
            mock.patch.object(
                triton_common,
                "_should_skip_zero_weight_tiny_routes",
                return_value=True,
            ),
            mock.patch.object(
                triton_common.tokenspeed_kernel,
                "moe_dispatch",
                side_effect=moe_dispatch,
            ),
            mock.patch.object(
                triton_common.tokenspeed_kernel,
                "moe_combine",
                side_effect=moe_combine,
            ),
            mock.patch(
                "tokenspeed.runtime.layers.activation.silu_and_mul",
                side_effect=silu_and_mul,
            ),
        ):
            output = triton_common.triton_forward(
                gate_up_gemm,
                down_gemm,
                get_config_func,
                "silu",
                layer,
                hidden_states,
                topk_output,
            )

        torch.testing.assert_close(output, torch.full_like(hidden_states, 2))

    def test_localize_topk_for_ep_marks_nonlocal_experts(self):
        topk_ids = torch.tensor([[0, 3, 4, 7], [5, 2, -1, 6]], dtype=torch.int32)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)

        local_ids, local_weights = deep_gemm_backend._localize_topk_for_ep(
            topk_ids,
            topk_weights,
            ep_rank=1,
            ep_size=2,
            num_local_experts=4,
        )

        torch.testing.assert_close(
            local_ids,
            torch.tensor([[-1, -1, 0, 3], [1, -1, -1, 2]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            local_weights,
            torch.tensor([[0, 0, 1, 1], [1, 0, 0, 1]], dtype=torch.float32),
        )

    def test_aligned_local_expert_token_counts(self):
        topk_ids = torch.tensor([[0, 0, 1], [-1, 2, 2]], dtype=torch.int64)

        counts = deep_gemm_backend._aligned_local_expert_token_counts(
            topk_ids,
            num_local_experts=4,
            alignment=8,
        )

        self.assertEqual(counts, [8, 8, 8, 0])

    def test_masked_local_route_metadata_sorts_and_positions_routes(self):
        topk_ids = torch.tensor([[2, -1, 1], [2, 0, 1]], dtype=torch.int64)
        topk_weights = torch.tensor(
            [[0.2, 0.0, 0.1], [0.5, 0.7, 0.3]],
            dtype=torch.float32,
        )

        (
            expert_ids,
            token_ids,
            route_positions,
            route_weights,
            masked_m,
        ) = deep_gemm_backend._masked_local_route_metadata(
            topk_ids,
            topk_weights,
            num_local_experts=4,
        )

        torch.testing.assert_close(
            expert_ids,
            torch.tensor([0, 1, 1, 2, 2], dtype=torch.long),
        )
        torch.testing.assert_close(
            token_ids,
            torch.tensor([1, 0, 1, 0, 1], dtype=torch.long),
        )
        torch.testing.assert_close(
            route_positions,
            torch.tensor([0, 0, 1, 0, 1], dtype=torch.long),
        )
        torch.testing.assert_close(
            route_weights,
            torch.tensor([0.7, 0.1, 0.3, 0.2, 0.5], dtype=torch.float32),
        )
        torch.testing.assert_close(
            masked_m,
            torch.tensor([1, 2, 2, 0], dtype=torch.int32),
        )

    def test_deep_gemm_backend_delegates_decode_microbatch_to_triton(self):
        backend = deep_gemm_backend.Fp8DeepGemmBackend(
            key=BackendKey("sm100", "fp8", "deep_gemm"),
            spec=self._make_glm5_spec(),
            quant_config=self._make_fp8_config(),
        )
        backend._gate_up_gemm = object()
        backend._down_gemm = object()
        backend._get_config_func = object()
        hidden_states = torch.randn(2, 6144, dtype=torch.bfloat16)
        topk_output = SimpleNamespace(
            topk_ids=torch.zeros(2, 8, dtype=torch.int64),
            topk_weights=torch.ones(2, 8, dtype=torch.float32),
        )
        layer = SimpleNamespace(activation="silu")
        expected = torch.randn_like(hidden_states)

        with (
            mock.patch.object(
                deep_gemm_backend,
                "triton_forward",
                return_value=expected,
            ) as triton_forward,
        ):
            actual = backend.forward(layer, hidden_states, topk_output, 2, 2)

        self.assertIs(actual, expected)
        triton_forward.assert_called_once_with(
            backend._gate_up_gemm,
            backend._down_gemm,
            backend._get_config_func,
            "silu",
            layer,
            hidden_states,
            topk_output,
            workspace=backend._triton_workspace,
        )

    def test_forced_deep_gemm_backend_selection_on_sm100(self):
        backend = self._select_backend(MoeBackend.DEEP_GEMM)
        self.assertIsInstance(backend, deep_gemm_backend.Fp8DeepGemmBackend)

    def test_forced_deep_gemm_backend_selection_for_glm5_shape(self):
        backend = self._select_backend(
            MoeBackend.DEEP_GEMM,
            hidden_size=6144,
            intermediate_size=2048,
            num_experts=32,
            num_local_experts=4,
            ep_size=8,
            top_k=8,
        )
        self.assertIsInstance(backend, deep_gemm_backend.Fp8DeepGemmBackend)

    def test_auto_fp8_backend_keeps_triton_for_non_glm5_shape(self):
        backend = self._select_backend(MoeBackend.AUTO)
        self.assertIsInstance(backend, Fp8TritonBackend)

    def test_auto_fp8_backend_selects_deep_gemm_for_glm5_sm100_shape(self):
        backend = self._select_backend(
            MoeBackend.AUTO,
            hidden_size=6144,
            intermediate_size=2048,
            num_experts=32,
            num_local_experts=4,
            ep_size=8,
            top_k=8,
        )
        self.assertIsInstance(backend, deep_gemm_backend.Fp8DeepGemmBackend)

    def _select_backend(
        self,
        moe_backend: MoeBackend,
        *,
        hidden_size: int = 1024,
        intermediate_size: int = 2048,
        num_experts: int = 32,
        num_local_experts: int = 4,
        ep_size: int = 8,
        top_k: int = 8,
    ):
        spec = self._make_glm5_spec(
            top_k=top_k,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ep_size=ep_size,
        )
        quant_config = self._make_fp8_config()
        fake_platform = SimpleNamespace(
            is_amd=False,
            is_nvidia=True,
            arch_version=SimpleNamespace(major=10, minor=0),
        )

        with (
            mock.patch.object(selector, "current_platform", return_value=fake_platform),
            mock.patch(
                "tokenspeed.runtime.layers.moe.utils.MOE_BACKEND",
                moe_backend,
            ),
            mock.patch.object(
                deep_gemm_backend,
                "_DEEP_GEMM_FP8_GROUPED_AVAILABLE",
                True,
            ),
        ):
            return selector.select_backend(spec, quant_config)

    def _make_glm5_spec(
        self,
        *,
        top_k: int = 8,
        num_experts: int = 32,
        num_local_experts: int = 4,
        hidden_size: int = 6144,
        intermediate_size: int = 2048,
        ep_size: int = 8,
    ):
        return MoELayerSpec(
            top_k=top_k,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation="silu",
            tp_rank=0,
            tp_size=1,
            ep_rank=0,
            ep_size=ep_size,
        )

    def _make_fp8_config(self):
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )


if __name__ == "__main__":
    unittest.main()
