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

"""Typed hand-off objects shared by DSA models and attention backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DSADecodeTopK:
    """Top-k workspaces in the packed token coordinate space.

    The tensors cover every token row in the model forward.  A decode plan
    selects the trailing decode rows when the batch also contains prefill work.
    """

    topk_indices: torch.Tensor
    topk_lens: torch.Tensor


@dataclass(frozen=True)
class DSADecodePlan:
    """One authoritative sparse-decode layout for a model forward.

    The DSA backend derives this plan from scheduler batch counts, the dense
    MLA metadata it owns, and the actual packed token count.  Both indexer
    top-k and sparse attention consume the same instance, so neither path
    reconstructs request or query geometry independently.

    Tensor storage may be refreshed in place for CUDA graph replay, while the
    structural fields stay fixed for the lifetime of the plan.
    """

    token_start: int
    token_end: int
    num_requests: int
    q_len_per_req: int
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    seq_lens_2d: torch.Tensor
    max_seq_len: int
    kernel_plan: object | None

    @property
    def num_tokens(self) -> int:
        """Number of packed query rows belonging to decode requests."""

        return self.token_end - self.token_start
