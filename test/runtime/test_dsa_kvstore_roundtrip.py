"""Regression: GLM DSA KVStore (L2 host cache) offload/reload must be lossless.

DSA's device KV is three heterogeneous per-layer buffers (packed FP8
``sparse_decode_kv_buffer``, the ``index_k_buffer`` indexer keys and the packed
FP8 ``index_k_with_scale_buffer``), not the single MLA latent the parent
``MLATokenToKVPoolHost`` assumes -- which is why ``DSAKVPoolHost`` exists. This
test exercises a device->host->device round trip through ``DSAKVPoolHost`` and
asserts every buffer comes back bit-exact, guarding against:

* offloading the wrong (freed placeholder) buffer -> out-of-bounds, and
* the non-8-byte-aligned scale row tripping the vectorized MLA copy kernel
  (it must use the alignment-tolerant direct copy instead).

Requires a CUDA device and the kvcacheio transfer kernels, so it is skipped on
CPU-only runners.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DSA KVStore round trip requires CUDA"
)


class _MockDSADevicePool:
    """Minimal stand-in exposing the attributes ``DSAKVPoolHost`` reads."""

    def __init__(self, layer_num: int, size: int, page_size: int, device: str):
        self.layer_num = layer_num
        self.size = size
        self.page_size = page_size
        self.device = device
        self.store_dtype = torch.float8_e4m3fn
        self.kv_lora_rank = 512
        self.qk_rope_head_dim = 64
        # Real GLM-5.1 DSA row widths: sparse-decode and index-K are 8-aligned,
        # the packed scale row (index_head_dim + 4 bytes / 128 lanes) is not.
        self.sparse_decode_kv_row_bytes = 656
        self.index_head_dim = 128
        self.index_k_with_scale_row_bytes = 132
        rows = size + page_size
        gen = torch.Generator(device=device).manual_seed(1234)
        self.sparse_decode_kv_buffer = [
            torch.randint(
                0,
                256,
                (rows, self.sparse_decode_kv_row_bytes),
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(layer_num)
        ]
        self.index_k_buffer = [
            torch.randn(
                rows,
                self.index_head_dim,
                dtype=torch.bfloat16,
                device=device,
                generator=gen,
            )
            for _ in range(layer_num)
        ]
        self.index_k_with_scale_buffer = [
            torch.randint(
                0,
                256,
                (rows, self.index_k_with_scale_row_bytes),
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(layer_num)
        ]


def test_dsa_kvstore_offload_reload_is_lossless() -> None:
    from tokenspeed.runtime.cache.kv_cache_host import DSAKVPoolHost

    device = "cuda"
    layer_num, size, page_size = 3, 256, 64
    pool = _MockDSADevicePool(layer_num, size, page_size, device)
    orig = (
        [b.clone() for b in pool.sparse_decode_kv_buffer],
        [b.clone() for b in pool.index_k_buffer],
        [b.clone() for b in pool.index_k_with_scale_buffer],
    )

    host = DSAKVPoolHost(
        pool,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=page_size,
        layout="layer_first",
        device="cpu",
    )

    # Offload device page #1 (tokens 64..127) to host page #0 (tokens 0..63).
    device_idx = torch.arange(64, 128, dtype=torch.int64, device=device)
    host_idx = torch.arange(0, 64, dtype=torch.int64, device=device)
    host.backup_from_device_all_layer(pool, host_idx, device_idx, "kernel")
    torch.cuda.synchronize()

    # Wipe the device page, then reload it from host.
    for buffers in (
        pool.sparse_decode_kv_buffer,
        pool.index_k_buffer,
        pool.index_k_with_scale_buffer,
    ):
        for b in buffers:
            b[64:128].zero_()
    for layer_id in range(layer_num):
        host.load_to_device_per_layer(pool, host_idx, device_idx, layer_id, "kernel")
    torch.cuda.synchronize()

    for name, cur, ref in (
        ("sparse_decode_kv_buffer", pool.sparse_decode_kv_buffer, orig[0]),
        ("index_k_buffer", pool.index_k_buffer, orig[1]),
        ("index_k_with_scale_buffer", pool.index_k_with_scale_buffer, orig[2]),
    ):
        for layer_id in range(layer_num):
            assert torch.equal(
                cur[layer_id][64:128], ref[layer_id][64:128]
            ), f"{name} layer {layer_id} corrupted across KVStore offload/reload"
