import os
import pathlib
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_REPO = pathlib.Path(__file__).resolve().parents[2]
_GLM5 = _REPO / "python/tokenspeed/runtime/models/glm5.py"
_DSA = _REPO / "python/tokenspeed/runtime/layers/attention/backends/dsa.py"
_CTX = _REPO / "python/tokenspeed/runtime/execution/context.py"
_DRAFTER = _REPO / "python/tokenspeed/runtime/execution/drafter/eagle.py"
_HF_UTILS = _REPO / "python/tokenspeed/runtime/utils/hf_transformers_utils.py"


class TestDsaMultiQueryVerifyWiring(unittest.TestCase):
    """GLM5 DSA sparse decode multi-query (MTP verify) wiring guards.

    Target verify runs q_len_per_req = spec_num_tokens (= 2 for GLM5's single
    MTP layer) query tokens per request through the sparse decode path:

    - DeepGEMM ``fp8_paged_mqa_logits`` takes q ``[reqs, next_n, heads, dim]``
      with 2D ``context_lens [reqs, next_n]`` holding PER-TOKEN visible
      lengths (verified bit-exact vs two next_n=1 calls on B200).
    - FlashMLA sparse takes q ``[reqs, q_len, heads, dim]`` with per-token
      indices rows; causality is carried by the -1 padding inside each row and
      ``topk_length`` stays per-request (verified bit-exact on B200).

    Token ``j`` of a request may only see ``seq_lens - q_len + j + 1``
    positions, where ``seq_lens`` is the full per-request context (draft
    tokens already in the KV cache).
    """

    def test_single_query_raise_removed(self):
        src = _GLM5.read_text()
        self.assertNotIn("supports one query token", src)
        # the q_len guard accepts multi-step MTP verify rows up to next_n = 6
        self.assertIn("_check_decode_q_len_per_req", src)
        self.assertIn("next_n <= 6", src)

    def test_per_token_expansion_wired(self):
        src = _GLM5.read_text()
        self.assertIn("_expand_decode_seq_lens_per_token", src)
        # deep_gemm q is reshaped to [reqs, next_n, heads, dim], not
        # unconditionally unsqueeze(1)
        self.assertNotIn("q_fp8.unsqueeze(1)", src)
        # per-token block table rows feed the local->global slot kernels
        self.assertIn("block_tables_per_token", src)

    def test_flashmla_sparse_multi_query_wired(self):
        src = _DSA.read_text()
        self.assertIn("q_len_per_req", src)
        # indices must be per-token rows shaped [reqs, q_len, topk]
        self.assertIn("topk_indices.view(num_reqs, q_len_per_req, -1)", src)
        # the old fixed seq_len_q=1 view is gone
        self.assertNotIn("q.view(q.shape[0], 1, layer.tp_q_head_num", src)

    def test_glm_dsa_topk_carrier_is_context_scoped(self):
        glm_src = _GLM5.read_text()
        ctx_src = _CTX.read_text()
        drafter_src = _DRAFTER.read_text()
        self.assertNotIn("_GLM_DSA_CARRIED_TOPK", glm_src)
        self.assertIn("glm_dsa_decode_topk", ctx_src)
        self.assertIn("_seed_glm_dsa_topk", drafter_src)

    def test_nextn_attention_receives_nextn_flag(self):
        src = _GLM5.read_text()
        self.assertIn("is_nextn: bool = False", src)
        self.assertIn("is_nextn=is_nextn", src)
        self.assertIn("self.is_nextn", src)

    def test_mtp_iteration_share_config_is_restored(self):
        src = _HF_UTILS.read_text()
        self.assertIn("index_share_for_mtp_iteration", src)


class TestPerTokenSeqLens(unittest.TestCase):
    """Causality formula checks. Requires runtime deps; skipped on CPU box."""

    def _expand(self):
        try:
            from tokenspeed.runtime.models.glm5 import GlmMoeDsaAttention
        except Exception as e:  # pragma: no cover - dep-gated
            self.skipTest(f"runtime deps unavailable: {e}")
        return GlmMoeDsaAttention._expand_decode_seq_lens_per_token

    def test_qlen1_is_identity(self):
        try:
            import torch
        except Exception as e:  # pragma: no cover - dep-gated
            self.skipTest(f"torch unavailable: {e}")

        expand = self._expand()
        seq_lens = torch.tensor([7, 130, 4096], dtype=torch.int32)
        out = expand(seq_lens, 1)
        self.assertIs(out, seq_lens)

    def test_qlen2_per_token_causality(self):
        try:
            import torch
        except Exception as e:  # pragma: no cover - dep-gated
            self.skipTest(f"torch unavailable: {e}")

        expand = self._expand()
        seq_lens = torch.tensor([7, 130], dtype=torch.int32)
        out = expand(seq_lens, 2)
        # token j of request i sees seq_lens[i] - 2 + j + 1 positions
        self.assertEqual(out.tolist(), [6, 7, 129, 130])
        # the last token of each request sees the full context, so the
        # per-request seq_lens upper-bounds every token (fit-topk check
        # stays on the per-request tensor)
        self.assertTrue((out.view(-1, 2).max(dim=1).values == seq_lens).all())

    def test_draft_decode_token_count_uses_actual_rows(self):
        from types import SimpleNamespace

        try:
            from tokenspeed.runtime.models.glm5 import GlmMoeDsaAttention
        except Exception as e:  # pragma: no cover - dep-gated
            self.skipTest(f"runtime deps unavailable: {e}")

        draft_ctx = SimpleNamespace(
            attn_backend=SimpleNamespace(spec_num_tokens=6, is_draft=True)
        )
        self.assertEqual(
            GlmMoeDsaAttention._resolve_num_decode_tokens(
                draft_ctx,
                total_tokens=2,
                num_decode_reqs=2,
            ),
            2,
        )

        target_ctx = SimpleNamespace(
            attn_backend=SimpleNamespace(spec_num_tokens=6, is_draft=False)
        )
        self.assertEqual(
            GlmMoeDsaAttention._resolve_num_decode_tokens(
                target_ctx,
                total_tokens=12,
                num_decode_reqs=2,
            ),
            12,
        )


if __name__ == "__main__":
    unittest.main()
