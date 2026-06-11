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
        # the q_len guard accepts multi-step MTP verify rows up to next_n = 4
        self.assertIn("_check_decode_q_len_per_req", src)
        self.assertIn("next_n <= 4", src)

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


class TestPerTokenSeqLens(unittest.TestCase):
    """Causality formula checks. Requires runtime deps; skipped on CPU box."""

    def _expand(self):
        try:
            from tokenspeed.runtime.models.glm5 import GlmMoeDsaAttention
        except Exception as e:  # pragma: no cover - dep-gated
            self.skipTest(f"runtime deps unavailable: {e}")
        return GlmMoeDsaAttention._expand_decode_seq_lens_per_token

    def test_qlen1_is_identity(self):
        import torch

        expand = self._expand()
        seq_lens = torch.tensor([7, 130, 4096], dtype=torch.int32)
        out = expand(seq_lens, 1)
        self.assertIs(out, seq_lens)

    def test_qlen2_per_token_causality(self):
        import torch

        expand = self._expand()
        seq_lens = torch.tensor([7, 130], dtype=torch.int32)
        out = expand(seq_lens, 2)
        # token j of request i sees seq_lens[i] - 2 + j + 1 positions
        self.assertEqual(out.tolist(), [6, 7, 129, 130])
        # the last token of each request sees the full context, so the
        # per-request seq_lens upper-bounds every token (fit-topk check
        # stays on the per-request tensor)
        self.assertTrue((out.view(-1, 2).max(dim=1).values == seq_lens).all())


if __name__ == "__main__":
    unittest.main()
