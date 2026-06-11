from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton.dsa import (
    glm_dsa_full_context_topk_to_global_slots,
    glm_dsa_local_topk_to_global_slots,
)


def _expected_local_offsets(seq_lens: torch.Tensor, topk: int) -> torch.Tensor:
    offsets = torch.arange(topk, dtype=torch.int32, device=seq_lens.device)
    return torch.where(
        offsets.view(1, topk) < seq_lens.view(seq_lens.numel(), 1),
        offsets.view(1, topk).expand(seq_lens.numel(), topk),
        torch.full(
            (seq_lens.numel(), topk),
            -1,
            dtype=torch.int32,
            device=seq_lens.device,
        ),
    )


@pytest.mark.parametrize("topk", [1, 4, 9])
def test_glm_dsa_full_context_topk_matches_local_transform_cpu(topk: int) -> None:
    seq_lens = torch.tensor([0, 1, 3, 7], dtype=torch.int32)
    block_table = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
            [40, 41, 42],
        ],
        dtype=torch.int32,
    )
    local_offsets = _expected_local_offsets(seq_lens, topk)

    expected_slots, expected_lens = glm_dsa_local_topk_to_global_slots(
        local_topk_offsets=local_offsets,
        block_table=block_table,
        block_size=4,
        seq_lens=seq_lens,
    )
    actual_slots, actual_lens = glm_dsa_full_context_topk_to_global_slots(
        seq_lens=seq_lens,
        block_table=block_table,
        block_size=4,
        topk=topk,
    )

    assert torch.equal(actual_slots, expected_slots)
    assert torch.equal(actual_lens, expected_lens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("topk", [4, 17])
def test_glm_dsa_full_context_topk_matches_local_transform_cuda(topk: int) -> None:
    seq_lens = torch.tensor([0, 1, 3, 7, 16], dtype=torch.int32, device="cuda")
    block_table = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
            [40, 41, 42, 43],
            [50, 51, 52, 53],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    local_offsets = _expected_local_offsets(seq_lens, topk)

    expected_slots, expected_lens = glm_dsa_local_topk_to_global_slots(
        local_topk_offsets=local_offsets,
        block_table=block_table,
        block_size=4,
        seq_lens=seq_lens,
    )
    actual_slots, actual_lens = glm_dsa_full_context_topk_to_global_slots(
        seq_lens=seq_lens,
        block_table=block_table,
        block_size=4,
        topk=topk,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_slots, expected_slots)
    assert torch.equal(actual_lens, expected_lens)


def test_glm_dsa_local_topk_masks_offsets_beyond_seq_lens_cpu() -> None:
    seq_lens = torch.tensor([2, 5], dtype=torch.int32)
    block_table = torch.tensor(
        [
            [10, 11],
            [20, 21],
        ],
        dtype=torch.int32,
    )
    local_offsets = torch.tensor(
        [
            [0, 1, 2, -1, 99],
            [4, 5, 1, -1, 128],
        ],
        dtype=torch.int32,
    )

    actual_slots, actual_lens = glm_dsa_local_topk_to_global_slots(
        local_topk_offsets=local_offsets,
        block_table=block_table,
        block_size=4,
        seq_lens=seq_lens,
    )

    expected_slots = torch.tensor(
        [
            [40, 41, -1, -1, -1],
            [84, -1, 81, -1, -1],
        ],
        dtype=torch.int32,
    )
    expected_lens = torch.tensor([2, 2], dtype=torch.int32)
    assert torch.equal(actual_slots, expected_slots)
    assert torch.equal(actual_lens, expected_lens)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="requires CUDA",
            ),
        ),
    ],
)
def test_glm_dsa_local_topk_writes_preallocated_outputs(device: str) -> None:
    seq_lens = torch.tensor([2, 5], dtype=torch.int32, device=device)
    block_table = torch.tensor(
        [
            [10, 11],
            [20, 21],
        ],
        dtype=torch.int32,
        device=device,
    )
    local_offsets = torch.tensor(
        [
            [0, 1, 2, -1],
            [4, 1, 5, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    out = torch.full_like(local_offsets, 123)
    lens_out = torch.full((2,), -9, dtype=torch.int32, device=device)

    actual_slots, actual_lens = glm_dsa_local_topk_to_global_slots(
        local_topk_offsets=local_offsets,
        block_table=block_table,
        block_size=4,
        seq_lens=seq_lens,
        out=out,
        lens_out=lens_out,
    )
    if device == "cuda":
        torch.cuda.synchronize()

    expected_slots = torch.tensor(
        [
            [40, 41, -1, -1],
            [84, 81, -1, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    expected_lens = torch.tensor([2, 2], dtype=torch.int32, device=device)
    assert actual_slots.data_ptr() == out.data_ptr()
    assert actual_lens.data_ptr() == lens_out.data_ptr()
    assert torch.equal(out, expected_slots)
    assert torch.equal(lens_out, expected_lens)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="requires CUDA",
            ),
        ),
    ],
)
def test_glm_dsa_full_context_topk_writes_preallocated_outputs(device: str) -> None:
    seq_lens = torch.tensor([3, 6], dtype=torch.int32, device=device)
    block_table = torch.tensor(
        [
            [10, 11],
            [20, 21],
        ],
        dtype=torch.int32,
        device=device,
    )
    out = torch.full((2, 4), 123, dtype=torch.int32, device=device)
    lens_out = torch.full((2,), -9, dtype=torch.int32, device=device)

    actual_slots, actual_lens = glm_dsa_full_context_topk_to_global_slots(
        seq_lens=seq_lens,
        block_table=block_table,
        block_size=4,
        topk=4,
        out=out,
        lens_out=lens_out,
    )
    if device == "cuda":
        torch.cuda.synchronize()

    expected_slots = torch.tensor(
        [
            [40, 41, 42, -1],
            [80, 81, 82, 83],
        ],
        dtype=torch.int32,
        device=device,
    )
    expected_lens = torch.tensor([3, 4], dtype=torch.int32, device=device)
    assert actual_slots.data_ptr() == out.data_ptr()
    assert actual_lens.data_ptr() == lens_out.data_ptr()
    assert torch.equal(out, expected_slots)
    assert torch.equal(lens_out, expected_lens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_glm_dsa_local_topk_masks_offsets_beyond_seq_lens_cuda() -> None:
    seq_lens = torch.tensor([2, 5], dtype=torch.int32, device="cuda")
    block_table = torch.tensor(
        [
            [10, 11],
            [20, 21],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    local_offsets = torch.tensor(
        [
            [0, 1, 2, -1, 99],
            [4, 5, 1, -1, 128],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    actual_slots, actual_lens = glm_dsa_local_topk_to_global_slots(
        local_topk_offsets=local_offsets,
        block_table=block_table,
        block_size=4,
        seq_lens=seq_lens,
    )
    torch.cuda.synchronize()

    expected_slots = torch.tensor(
        [
            [40, 41, -1, -1, -1],
            [84, -1, 81, -1, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    expected_lens = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
    assert torch.equal(actual_slots, expected_slots)
    assert torch.equal(actual_lens, expected_lens)
