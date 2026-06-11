"""Regression tests for --attention-backend / --drafter-attention-backend choices.

Guards against the bug where --drafter-attention-backend rejected valid main-model
backends (e.g. trtllm_mla) because its argparse `choices` was a narrower subset
of --attention-backend's.
"""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import argparse
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.layers.attention import registry
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.utils.server_args import ServerArgs, prepare_server_args


class TestAttentionBackendChoices(unittest.TestCase):
    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        return parser

    def _action(self, parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
        for action in parser._actions:
            if action.dest == dest:
                return action
        raise AssertionError(f"no action with dest={dest!r}")

    def test_attention_backend_accepts_trtllm_mla(self):
        args = self._build_parser().parse_args(
            ["--model", "x", "--attention-backend", "trtllm_mla"]
        )
        self.assertEqual(args.attention_backend, "trtllm_mla")

    def test_attention_backend_accepts_mla(self):
        args = self._build_parser().parse_args(
            ["--model", "x", "--attention-backend", "mla"]
        )
        self.assertEqual(args.attention_backend, "mla")

    def test_attention_backend_accepts_mha_kernel_solutions(self):
        for backend in ("fa3", "fa4", "triton", "flashinfer"):
            args = self._build_parser().parse_args(
                ["--model", "x", "--attention-backend", backend]
            )
            self.assertEqual(args.attention_backend, backend)

    def test_drafter_attention_backend_accepts_trtllm_mla(self):
        """Regression: trtllm_mla must be accepted here too."""
        args = self._build_parser().parse_args(
            ["--model", "x", "--drafter-attention-backend", "trtllm_mla"]
        )
        self.assertEqual(args.drafter_attention_backend, "trtllm_mla")

    def test_drafter_choices_match_main_choices(self):
        parser = self._build_parser()
        main = set(self._action(parser, "attention_backend").choices)
        drafter = set(self._action(parser, "drafter_attention_backend").choices)
        self.assertEqual(main, drafter)

    def test_invalid_backend_rejected_on_both_flags(self):
        for flag in ("--attention-backend", "--drafter-attention-backend"):
            parser = self._build_parser()
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--model", "x", flag, "bogus"])

    def test_inline_detokenizer_flag_removed_from_cli(self):
        parser = self._build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--model", "x", "--enable-inline-detokenizer"])

    def test_inline_detokenizer_is_forced_on(self):
        args = prepare_server_args(["--model", "x"])
        self.assertTrue(args.enable_inline_detokenizer)

    def test_model_path_alias_sets_model(self):
        args = self._build_parser().parse_args(["--model-path", "x"])
        self.assertEqual(args.model, "x")

    def test_prepare_server_args_accepts_model_path_alias(self):
        args = prepare_server_args(["--model-path", "x"])
        self.assertEqual(args.model, "x")

    def test_defaults_to_mha_for_mha(self):
        self.assertEqual(registry._get_default_backend_name(AttentionArch.MHA), "mha")

    def test_mha_kernel_solution_backends_use_mha_backend(self):
        from tokenspeed.runtime.layers.attention.backends.mha import MHAAttnBackend

        for backend in ("mha", "fa3", "fa4", "triton", "flashinfer"):
            self.assertIs(
                registry._get_backend_cls(backend, AttentionArch.MHA),
                MHAAttnBackend,
            )

    def test_mla_backend_registered_for_mla(self):
        from tokenspeed.runtime.layers.attention.backends.mla import MLAAttnBackend

        self.assertIs(
            registry._get_backend_cls("mla", AttentionArch.MLA),
            MLAAttnBackend,
        )

    def test_dsa_backend_wraps_dense_mla_backend(self):
        from tokenspeed.runtime.layers.attention.backends import dsa as dsa_backend
        from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
        from tokenspeed.runtime.layers.attention.backends.trtllm_mla import (
            TRTLLMMLABackend,
        )

        class FakeDenseMLABackend:
            def __init__(self, config):
                self.config = config
                self.forward_decode_metadata = SimpleNamespace(
                    num_extends=0,
                    seq_lens_k=torch.tensor([16], dtype=torch.int32),
                )
                self.forward_prefill_metadata = object()
                self.chunked_prefill_metadata = object()
                self.decode_cuda_graph_metadata = {}
                self.decode_cuda_graph_kv_indices = None

            def register_step_counter(self, step_counter):
                self.step_counter = step_counter

            def init_forward_metadata(self, **kwargs):
                self.init_forward_metadata_kwargs = kwargs

            def forward_decode(self, **kwargs):
                self.forward_decode_kwargs = kwargs
                return "dense-decode"

            def forward_extend_chunked(self, *args, **kwargs):
                self.forward_extend_chunked_args = args
                self.forward_extend_chunked_kwargs = kwargs
                return "dense-prefill"

        config = SimpleNamespace(
            device="cpu",
            backend_name="dsa",
            num_attention_heads=16,
            num_kv_heads=1,
            head_dim=640,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            page_size=64,
            context_len=4096,
            max_bs=8,
            max_graph_bs=8,
            kv_cache_quant_method="none",
            speculative_num_draft_tokens=1,
            is_draft=False,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=512,
            scaling=1.0,
            kv_cache_dim=576,
            index_topk=512,
        )

        with mock.patch.object(
            dsa_backend,
            "TRTLLMMLABackend",
            FakeDenseMLABackend,
        ):
            backend = dsa_backend.DSABackend(config)

        self.assertIsInstance(backend, AttentionBackend)
        self.assertFalse(issubclass(dsa_backend.DSABackend, TRTLLMMLABackend))
        self.assertIs(
            backend.forward_decode_metadata,
            backend._dense_backend.forward_decode_metadata,
        )

        step_counter = object()
        backend.register_step_counter(step_counter)
        self.assertIs(backend._dense_backend.step_counter, step_counter)

        backend.decode_cuda_graph_kv_indices = "aliased-kv-indices"
        backend._block_table_aliased = True
        self.assertEqual(
            backend._dense_backend.decode_cuda_graph_kv_indices,
            "aliased-kv-indices",
        )
        self.assertTrue(backend._dense_backend._block_table_aliased)

        dense_out = backend.forward_decode(
            q=torch.empty(1, 16, 640),
            k=None,
            v=None,
            layer=SimpleNamespace(logit_cap=0),
            out_cache_loc=torch.empty(1, dtype=torch.int32),
            token_to_kv_pool=SimpleNamespace(),
            bs=1,
        )
        self.assertEqual(dense_out, "dense-decode")

    def test_dsa_flashmla_sparse_prefill_rejects_zero_heads(self):
        from tokenspeed.runtime.layers.attention.backends import dsa as dsa_backend

        with self.assertRaisesRegex(RuntimeError, "positive query head count"):
            dsa_backend._flashmla_sparse_prefill_padded_heads(0, 128)

    def test_dsa_sparse_prefill_workspace_reuses_capacity(self):
        from tokenspeed.runtime.layers.attention.backends import dsa as dsa_backend

        backend = object.__new__(dsa_backend.DSABackend)
        backend.kv_cache_dim = 8
        backend._prefill_workspace_buffer = None
        backend._prefill_workspace_rows = 0
        backend._prefill_workspace_dim = 0

        first = backend._get_prefill_workspace(
            num_reqs=2,
            max_seq_len=4,
            device=torch.device("cpu"),
        )
        second = backend._get_prefill_workspace(
            num_reqs=1,
            max_seq_len=3,
            device=torch.device("cpu"),
        )
        third = backend._get_prefill_workspace(
            num_reqs=3,
            max_seq_len=4,
            device=torch.device("cpu"),
        )

        self.assertEqual(first.shape, (2, 4, 1, 8))
        self.assertEqual(second.shape, (1, 3, 1, 8))
        self.assertEqual(third.shape, (3, 4, 1, 8))
        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(backend._prefill_workspace_rows, 12)

    def test_dsa_sparse_prefill_uses_precomputed_kv_slots(self):
        from tokenspeed.runtime.layers.attention.backends import dsa as dsa_backend

        backend = object.__new__(dsa_backend.DSABackend)
        backend.page_size = 64
        backend.kv_lora_rank = 512
        backend.kv_cache_dim = 4
        backend.data_type = torch.bfloat16
        backend._prefill_workspace_buffer = None
        backend._prefill_workspace_rows = 0
        backend._prefill_workspace_dim = 0
        backend._prefill_query_workspace = None
        backend._prefill_query_workspace_num_heads = None

        k_cache = torch.arange(24 * 4, dtype=torch.bfloat16).view(24, 1, 4)
        pool = SimpleNamespace(
            quant_method=None,
            get_key_buffer=lambda layer_id: k_cache,
        )
        layer = SimpleNamespace(
            layer_id=0,
            logit_cap=0.0,
            tp_q_head_num=1,
            head_dim=2,
            scaling=1.0,
            v_head_dim=2,
        )
        q = torch.ones(2, 2, dtype=torch.bfloat16)
        kv_workspace_slots = torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.int64)
        captured = {}

        def fake_flash_mla_sparse_fwd(**kwargs):
            captured["kv"] = kwargs["kv"].detach().clone()
            return (
                torch.zeros(2, 1, 2, dtype=torch.bfloat16),
                None,
                None,
            )

        with (
            mock.patch.object(
                dsa_backend,
                "_flashmla_sparse_prefill_head_multiple",
                return_value=1,
            ),
            mock.patch.object(
                dsa_backend,
                "flash_mla_sparse_fwd",
                side_effect=fake_flash_mla_sparse_fwd,
            ),
        ):
            out = backend.forward_sparse_prefill(
                q=q,
                layer=layer,
                token_to_kv_pool=pool,
                block_tables=torch.zeros(2, 1, dtype=torch.int32),
                seq_lens=torch.tensor([3, 3], dtype=torch.int32),
                workspace_indices=torch.zeros(2, 1, dtype=torch.int32),
                topk_lens=torch.ones(2, dtype=torch.int32),
                kv_workspace_slots=kv_workspace_slots,
                max_seq_len=3,
            )

        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(backend._prefill_workspace_rows, 6)
        torch.testing.assert_close(
            captured["kv"],
            k_cache.index_select(0, kv_workspace_slots),
        )

    def test_dsa_sparse_prefill_query_workspace_reuses_and_clears_tail(self):
        from tokenspeed.runtime.layers.attention.backends import dsa as dsa_backend

        backend = object.__new__(dsa_backend.DSABackend)
        backend._prefill_query_workspace = None
        backend._prefill_query_workspace_num_heads = None

        q1 = torch.ones(2, 16, 4)
        padded1, actual_heads = backend._pad_sparse_prefill_query_heads(
            q1,
            num_heads=16,
            head_dim=4,
            head_multiple=64,
        )

        q2 = torch.full((1, 8, 4), 2.0)
        padded2, actual_heads2 = backend._pad_sparse_prefill_query_heads(
            q2,
            num_heads=8,
            head_dim=4,
            head_multiple=64,
        )

        self.assertEqual(actual_heads, 16)
        self.assertEqual(actual_heads2, 8)
        self.assertEqual(padded1.data_ptr(), padded2.data_ptr())
        self.assertEqual(padded2.shape, (1, 64, 4))
        torch.testing.assert_close(padded2[:, :8, :], q2)
        self.assertTrue(torch.all(padded2[:, 8:, :] == 0))

    def test_dsa_flashmla_sparse_decode_rejects_zero_heads(self):
        from tokenspeed.runtime.layers.attention.backends import dsa as dsa_backend

        with self.assertRaisesRegex(RuntimeError, "positive query head count"):
            dsa_backend._flashmla_sparse_decode_padded_heads(0)

    def test_dsa_sparse_decode_metadata_is_persistent_per_shape(self):
        # The FlashMLA sched_meta must be reused for a given (num_reqs, q_len)
        # so its lazily-built tile schedule survives CUDA graph capture/replay
        # at fixed buffer addresses. Distinct shapes get distinct metadata.
        from tokenspeed.runtime.layers.attention.backends import dsa as dsa_backend

        backend = object.__new__(dsa_backend.DSABackend)
        calls = []

        def fake_get_mla_metadata():
            metadata = object()
            calls.append(metadata)
            return metadata, None

        with mock.patch.object(
            dsa_backend,
            "get_mla_metadata",
            fake_get_mla_metadata,
        ):
            first = backend._get_sparse_decode_tile_metadata(4, 1)
            second = backend._get_sparse_decode_tile_metadata(4, 1)
            verify = backend._get_sparse_decode_tile_metadata(4, 2)
            other_bs = backend._get_sparse_decode_tile_metadata(8, 1)

        self.assertIs(first, second)
        self.assertIsNot(first, verify)
        self.assertIsNot(first, other_bs)
        self.assertEqual(len(calls), 3)

    def test_dsa_sparse_decode_query_workspace_reuses_and_clears_tail(self):
        from tokenspeed.runtime.layers.attention.backends import dsa as dsa_backend

        backend = object.__new__(dsa_backend.DSABackend)
        backend._decode_query_workspace = None
        backend._decode_query_workspace_num_heads = None

        q1 = torch.ones(2, 1, 16, 4)
        padded1, actual_heads = backend._pad_sparse_decode_query_heads(
            q1,
            num_heads=16,
        )

        q2 = torch.full((1, 1, 8, 4), 2.0)
        padded2, actual_heads2 = backend._pad_sparse_decode_query_heads(
            q2,
            num_heads=8,
        )

        self.assertEqual(actual_heads, 16)
        self.assertEqual(actual_heads2, 8)
        self.assertEqual(padded1.data_ptr(), padded2.data_ptr())
        self.assertEqual(padded2.shape, (1, 1, 64, 4))
        torch.testing.assert_close(padded2[:, :, :8, :], q2)
        self.assertTrue(torch.all(padded2[:, :, 8:, :] == 0))

    def test_sm90_defaults_to_flashmla_for_mla(self):
        platform = SimpleNamespace(is_blackwell=False, is_hopper=True)
        with mock.patch.object(registry, "current_platform", return_value=platform):
            self.assertEqual(
                registry._get_default_backend_name(AttentionArch.MLA), "flashmla"
            )

    def test_mha_config_propagates_speculative_settings(self):
        server_args = SimpleNamespace(
            device="cuda",
            attention_backend=None,
            drafter_attention_backend=None,
            attn_tp_size=None,
            mapping=SimpleNamespace(attn=SimpleNamespace(tp_size=2, dp_size=1)),
            kv_cache_dtype="auto",
            max_num_seqs=8,
            data_parallel_size=None,
            block_size=64,
            max_cudagraph_capture_size=4,
            kv_cache_quant_method="none",
            speculative_algorithm="EAGLE3",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
        )
        model_config = SimpleNamespace(
            context_len=4096,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            dtype="bfloat16",
        )

        config = MHAConfig.generate(server_args, model_config)

        self.assertEqual(config.speculative_num_steps, 3)
        self.assertEqual(config.speculative_num_draft_tokens, 4)


if __name__ == "__main__":
    unittest.main()
