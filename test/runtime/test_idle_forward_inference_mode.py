import os
import pathlib
import sys
import unittest

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

# Resolve the source file directly (no heavy module import) so the source-guard
# test stays CPU-only and dependency-light.
_MODEL_EXECUTOR_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python/tokenspeed/runtime/execution/model_executor.py"
)


class TestIdleForwardInferenceMode(unittest.TestCase):
    """Regression for the DP idle-forward MoE crash.

    On a DP rank with no work (e.g. a bs=1 request routed to a single rank),
    ``ModelExecutor.execute_idle_forward`` runs a zero-token forward so the rank
    still participates in the MoE all-to-all / dense NCCL collectives that the
    busy ranks issue. That zero-token batch hits the MoE align-block
    *large-buffer* branch (``sorted_ids.shape[0] > 4096``), which does an
    in-place ``sorted_ids.fill_`` on a buffer that was allocated during warmup
    as an *inference tensor*.

    In-place updates to inference tensors are only legal inside InferenceMode,
    and the model forward is merely ``@torch.no_grad()`` — so the idle rank
    crashed with "Inplace update to inference tensor outside InferenceMode is
    not allowed", taking the whole DP server down.

    Fix: ``execute_idle_forward`` wraps the idle forward in
    ``torch.inference_mode()``.
    """

    def test_idle_forward_source_wraps_inference_mode(self):
        # Guard the fix against accidental revert: the idle model forward must
        # run under inference_mode (stronger than the model's @no_grad) so the
        # cached inference-tensor MoE buffers can be updated in place on idle
        # DP ranks.
        src = _MODEL_EXECUTOR_SRC.read_text()
        self.assertIn(
            "with torch.inference_mode():",
            src,
            "execute_idle_forward must wrap the idle forward in "
            "torch.inference_mode() to avoid 'Inplace update to inference "
            "tensor' crashes on idle DP ranks",
        )

    def test_inference_tensor_inplace_needs_inference_mode(self):
        # Documents and guards the PyTorch invariant the fix relies on: a tensor
        # allocated inside inference_mode (like the warmup-cached MoE buffers)
        # can only be updated in place from inside inference_mode. A plain
        # no_grad context (the model forward's decorator) is NOT sufficient.
        with torch.inference_mode():
            buf = torch.zeros(8, dtype=torch.int32)  # an inference tensor

        with torch.no_grad():
            with self.assertRaises(RuntimeError):
                buf.fill_(3)

        with torch.inference_mode():
            buf.fill_(3)
        self.assertEqual(int(buf[0]), 3)


if __name__ == "__main__":
    unittest.main()
