"""Regression: DSA prefill must snapshot the page table during metadata prep.

GLM DSA's indexer top-k (``GlmMoeDsaAttention._compute_prefill_topk_indices``)
needs the extend requests' page table. Reading the mutable global ``req_to_page``
deep in the model forward races with the overlap scheduler's previous-step
post-processing, which runs concurrently on the default stream: the indexer uses
the page ids as *gather indices*, so a torn read becomes an out-of-bounds gather
(illegal memory access) instead of a silent numerical error like dense
attention, which uses the page table only as lengths. CUDA graph decode widens
the race window (the CPU runs far ahead of the GPU), so it surfaced as a crash
only at high concurrency.

``DSABackend.init_forward_metadata`` snapshots the page table in the prep phase
-- ordered like the dense backend's own block-table builds -- so the forward
consumes a stable copy. These tests pin that contract.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.dsa import DSABackend


class _DenseStub:
    """Minimal dense backend: records the call and exposes chunked metadata.

    The real dense backend builds ``chunked_prefill_metadata`` (including
    ``req_pool_indices`` in the order the model consumes the extend requests)
    inside ``init_forward_metadata``; the snapshot reuses that field so its rows
    line up with what the model iterates over.
    """

    def __init__(self) -> None:
        self.chunked_prefill_metadata = None

    def init_forward_metadata(self, *, req_pool_indices, **kwargs):
        self.chunked_prefill_metadata = type(
            "_ChunkMeta", (), {"req_pool_indices": req_pool_indices}
        )()
        return None


def _make_backend() -> DSABackend:
    be = object.__new__(DSABackend)
    be._dense_backend = _DenseStub()
    be._prefill_block_tables = None
    return be


def _init(be, *, num_extends, req_pool_indices, req_to_page, mode) -> None:
    be.init_forward_metadata(
        bs=int(req_pool_indices.numel()),
        num_extends=num_extends,
        req_pool_indices=req_pool_indices,
        seq_lens=torch.ones(req_pool_indices.numel(), dtype=torch.int32),
        forward_mode=mode,
        req_to_page=req_to_page,
    )


def test_extend_snapshots_page_table() -> None:
    """Extend batches snapshot req_to_page rows in chunk-metadata order."""
    be = _make_backend()
    req_to_page = torch.arange(5 * 4, dtype=torch.int32).reshape(5, 4)
    req_pool_indices = torch.tensor([2, 0, 3], dtype=torch.int64)
    _init(
        be,
        num_extends=3,
        req_pool_indices=req_pool_indices,
        req_to_page=req_to_page,
        mode=ForwardMode.EXTEND,
    )
    assert be._prefill_block_tables is not None
    assert torch.equal(be._prefill_block_tables, req_to_page[req_pool_indices])


def test_snapshot_is_decoupled_from_mutable_global() -> None:
    """The snapshot must not alias req_to_page.

    The overlap scheduler mutates the global page table concurrently; if the
    snapshot were a view, the forward would still read corrupted ids. A later
    in-place mutation must leave the snapshot untouched.
    """
    be = _make_backend()
    req_to_page = torch.arange(5 * 4, dtype=torch.int32).reshape(5, 4)
    req_pool_indices = torch.tensor([1, 4], dtype=torch.int64)
    _init(
        be,
        num_extends=2,
        req_pool_indices=req_pool_indices,
        req_to_page=req_to_page,
        mode=ForwardMode.EXTEND,
    )
    snapshot = be._prefill_block_tables.clone()
    req_to_page.fill_(-999)  # simulate the racing post-processing write
    assert torch.equal(
        be._prefill_block_tables, snapshot
    ), "snapshot aliased the mutable req_to_page and is still race-exposed"


def test_decode_clears_stale_snapshot() -> None:
    """A decode batch must clear any prefill snapshot from a prior extend step."""
    be = _make_backend()
    req_to_page = torch.arange(5 * 4, dtype=torch.int32).reshape(5, 4)
    _init(
        be,
        num_extends=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        req_to_page=req_to_page,
        mode=ForwardMode.EXTEND,
    )
    assert be._prefill_block_tables is not None
    _init(
        be,
        num_extends=0,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        req_to_page=req_to_page,
        mode=ForwardMode.DECODE,
    )
    assert be._prefill_block_tables is None
