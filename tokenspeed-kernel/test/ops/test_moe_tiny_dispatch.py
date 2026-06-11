import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.moe.triton import moe_align_block_size

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="tiny MoE dispatch is a CUDA Triton path",
)


def _assert_dispatch_layout(topk_ids, block_size, num_experts, topk_weights=None):
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        topk_ids,
        block_size,
        num_experts,
        topk_weights=topk_weights,
    )

    flat_ids = topk_ids.reshape(-1).detach().cpu()
    flat_weights = (
        torch.ones_like(flat_ids, dtype=torch.float32)
        if topk_weights is None
        else topk_weights.reshape(-1).detach().cpu()
    )
    total_routes = flat_ids.numel()
    expected_experts = [
        expert
        for expert in range(num_experts)
        if ((flat_ids == expert) & (flat_weights != 0)).any().item()
    ]
    expected_num_tokens = len(expected_experts) * block_size

    assert int(num_tokens_post_pad.cpu().item()) == expected_num_tokens
    torch.testing.assert_close(
        expert_ids[: len(expected_experts)].cpu(),
        torch.tensor(expected_experts, dtype=torch.int32),
    )

    sorted_cpu = sorted_ids[:expected_num_tokens].cpu()
    for block_idx, expert in enumerate(expected_experts):
        block = sorted_cpu[
            block_idx * block_size : (block_idx + 1) * block_size
        ].tolist()
        expected_routes = [
            route_idx
            for route_idx, route_expert in enumerate(flat_ids.tolist())
            if route_expert == expert and float(flat_weights[route_idx]) != 0.0
        ]
        expected_block = expected_routes + [total_routes] * (
            block_size - len(expected_routes)
        )
        assert block == expected_block


def test_tiny_moe_dispatch_groups_active_experts_only():
    topk_ids = torch.tensor(
        [[0, 2, 1, 2], [3, 0, 1, 0]],
        device="cuda",
        dtype=torch.int32,
    )

    _assert_dispatch_layout(topk_ids, block_size=8, num_experts=4)


def test_tiny_moe_dispatch_skips_empty_and_negative_experts():
    topk_ids = torch.tensor(
        [[2, 2, -1, 2]],
        device="cuda",
        dtype=torch.int32,
    )

    _assert_dispatch_layout(topk_ids, block_size=8, num_experts=4)


def test_tiny_moe_dispatch_covers_glm5_local_expert_shape():
    topk_ids = torch.tensor(
        [[0, 0, 1, 2, 0, 3, 0, 0]],
        device="cuda",
        dtype=torch.int32,
    )

    _assert_dispatch_layout(
        topk_ids,
        block_size=32,
        num_experts=32,
    )


def test_tiny_moe_dispatch_returns_tiny_capacity_slice():
    topk_ids = torch.tensor(
        [[0, 0, 1, 2, 0, 3, 0, 0]],
        device="cuda",
        dtype=torch.int32,
    )

    sorted_ids, expert_ids, _ = moe_align_block_size(
        topk_ids,
        block_size=32,
        num_experts=32,
    )

    assert sorted_ids.shape == (256,)
    assert expert_ids.shape == (8,)


def test_tiny_moe_dispatch_skips_zero_weight_routes():
    topk_ids = torch.tensor(
        [[0, 0, 1, 2, 0, 3, 0, 0]],
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.tensor(
        [[0.0, 0.0, 0.2, 0.3, 0.0, 0.4, 0.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )

    _assert_dispatch_layout(
        topk_ids,
        block_size=32,
        num_experts=32,
        topk_weights=topk_weights,
    )


def test_moe_localize_topk_maps_global_ep_routes_to_local_ids():
    topk_ids = torch.tensor(
        [[0, 3, 4, 7], [5, 2, -1, 6]],
        device="cuda",
        dtype=torch.int64,
    )
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)

    local_ids, local_weights = tokenspeed_kernel.moe_localize_topk(
        topk_ids,
        topk_weights,
        ep_rank=1,
        num_local_experts=4,
        nonlocal_expert_id=-1,
        dtype=topk_ids.dtype,
        expected_kernel_name="triton_moe_localize_topk",
    )

    torch.testing.assert_close(
        local_ids.cpu(),
        torch.tensor([[-1, -1, 0, 3], [1, -1, -1, 2]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        local_weights.cpu(),
        torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
    )


def test_moe_localize_topk_keeps_zero_sentinel_for_generic_path():
    topk_ids = torch.tensor([[0, 2]], device="cuda", dtype=torch.int32)
    topk_weights = torch.tensor([[0.75, 0.25]], device="cuda")

    local_ids, local_weights = tokenspeed_kernel.moe_localize_topk(
        topk_ids,
        topk_weights,
        ep_rank=0,
        num_local_experts=2,
        nonlocal_expert_id=0,
        dtype=topk_ids.dtype,
        expected_kernel_name="triton_moe_localize_topk",
    )

    torch.testing.assert_close(
        local_ids.cpu(),
        torch.tensor([[0, 0]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        local_weights.cpu(),
        torch.tensor([[0.75, 0.0]], dtype=torch.float32),
    )


def test_masked_moe_combine_ignores_zero_weight_routes():
    x = (
        torch.arange(1 * 4 * 16, device="cuda", dtype=torch.float32)
        .to(torch.bfloat16)
        .view(1, 4, 16)
    )
    topk_weights = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device="cuda")
    out = torch.empty((1, 16), device="cuda", dtype=torch.bfloat16)

    tokenspeed_kernel.moe_combine(
        x,
        out,
        1.0,
        topk_weights,
        dtype=torch.bfloat16,
        traits={
            "num_tokens": 1,
            "comm_strategy": None,
            "skip_zero_weights": True,
        },
        expected_kernel_name="triton_moe_sum_reduce_skip_zero_weights",
    )

    expected = x[:, [0, 2], :].sum(dim=1)
    torch.testing.assert_close(out, expected)
