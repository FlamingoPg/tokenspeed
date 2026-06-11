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

from __future__ import annotations

import functools
import json
import os
from contextlib import contextmanager
from typing import Any

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.utils import get_colorful_logger, get_device_name
from tokenspeed.runtime.utils.env import envs

__all__ = [
    "get_config",
    "get_config_dtype_str",
    "get_config_file_name",
    "get_config_file_path",
    "get_default_config",
    "get_moe_configs",
    "override_config",
    "try_get_optimal_moe_config",
]

logger = get_colorful_logger(__name__)
_is_amd = current_platform().is_amd

_config: dict[str, Any] | None = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> dict[str, Any] | None:
    return _config


def get_config_file_name(
    E: int,
    inter_size: int,
    hidden_size: int,
    dtype: str | None,
    block_shape: int | None = None,
    per_channel_quant: bool = False,
    down_moe: bool = False,
) -> str:
    device_name = get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    )
    per_channel_quant_selector = ",per_channel_quant=True" if per_channel_quant else ""
    down_moe_selector = "_down" if down_moe else ""
    return f"{E=},{inter_size=},{hidden_size=},device_name={device_name}{dtype_selector}{block_shape_selector}{per_channel_quant_selector}{down_moe_selector}.json"


def get_config_file_path(
    E: int,
    inter_size: int,
    hidden_size: int,
    dtype: str | None,
    block_n: int | None = 0,
    block_k: int | None = 0,
    per_channel_quant: bool = False,
    down_moe: bool = False,
):
    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(
        E,
        inter_size,
        hidden_size,
        dtype,
        [block_n, block_k],
        per_channel_quant,
        down_moe=down_moe,
    )

    # We found that using the fused_moe_kernel config from Triton 3.1.0 with
    # Triton 3.2.0 results in negative performance gains, so we also include
    # the Triton version as a key for finding the fused_moe_kernel config to
    # achieve the best performance.
    config_dir = envs.TOKENSPEED_MOE_CONFIG_DIR.get_set_value_or(
        os.path.dirname(os.path.realpath(__file__))
    )

    return os.path.join(config_dir, json_file_name)


@functools.lru_cache
def get_moe_configs(
    E: int,
    inter_size: int,
    hidden_size: int,
    dtype: str | None,
    block_n: int | None = 0,
    block_k: int | None = 0,
    per_channel_quant: bool = False,
    down_moe: bool = False,
) -> dict[int, Any] | None:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    config_file_path = get_config_file_path(
        E,
        inter_size,
        hidden_size,
        dtype,
        block_n,
        block_k,
        per_channel_quant,
        down_moe,
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            # Please note that although we find the config files, performance
            # might still be suboptimal. This is because the tuning environment
            # might differ from your current environment. For example, updating
            # the Triton version might cause all old configs to become suboptimal.
            # To achieve the best performance, consider re-tuning the Triton
            # fused MOE kernel in your environment.
            logger.info("Using MoE kernel config from %s.", config_file_path)
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        (
            "Using default MoE kernel config. Performance might be sub-optimal! "
            "Config file not found at %s."
        ),
        config_file_path,
    )
    return None


def get_default_config(
    M: int,
    E: int,
    inter_size: int,
    hidden_size: int,
    topk: int,
    dtype: str | None,
    is_marlin: bool,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    if dtype == "fp8_w8a8":
        if block_shape is None:
            config = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 32,
                "num_warps": 8,
                "num_stages": 2 if _is_amd else 4,
            }
            if M <= E:
                config = {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 2 if _is_amd else 4,
                }
        else:
            # Block-wise quant: BLOCK_SIZE_K must be divisible by block_shape[1]
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": block_shape[0],
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": 32,
                "num_warps": 4,
                "num_stages": 2 if _is_amd else 3,
            }
            if M <= E:
                # Decode / tiny-M: shrink the M tile to cut padding waste.
                # The non-block-wise path already special-cases M <= E, but the
                # block-wise path (GLM5.1 FP8 weight_block_size=(128,128)) did
                # not, so bs=1 decode padded a single token up to
                # BLOCK_SIZE_M=64, wasting 63/64 of every expert MMA tile.
                # BLOCK_SIZE_K stays tied to block_shape[1] for quant alignment;
                # only the free M dimension shrinks.
                config["BLOCK_SIZE_M"] = 16
                config["GROUP_SIZE_M"] = 1
                # Deeper pipelining helps the tiny-M gate_up GEMM (~19% on a
                # B200 CUDA-graph config sweep, 36.6->29.8us); down is unchanged
                # and BLOCK_SIZE_N=block_shape[0]=128 stays best (256 regresses).
                config["num_stages"] = 2 if _is_amd else 4
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        # A heuristic: fused marlin works faster with this config for small M
        if M <= E or (is_marlin and M <= 32):
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
    return config


def try_get_optimal_moe_config(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    dtype: str | None,
    M: int,
    is_marlin: bool = False,
    block_shape: list[int] | None = None,
    return_down_config: bool = False,
):
    del w1_shape

    down_config = None
    max_block_m = None
    override_cfg = get_config()
    if override_cfg:
        config = override_cfg
    else:
        # First try to load optimal config from the file
        E, hidden_size, inter_size = w2_shape
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0

        configs = get_moe_configs(
            E, inter_size, hidden_size, dtype, block_n, block_k, down_moe=False
        )
        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(
                M, E, inter_size, hidden_size, top_k, dtype, is_marlin, block_shape
            )

        if return_down_config:
            down_configs = get_moe_configs(
                E, inter_size, hidden_size, dtype, block_n, block_k, down_moe=True
            )
            if down_configs:
                down_config = down_configs[
                    min(down_configs.keys(), key=lambda x: abs(x - M))
                ]
                down_config = dict(**down_config)
                max_block_m = max(
                    [cfg["BLOCK_SIZE_M"] for cfg in down_configs.values()]
                )
            else:
                # Fall back to the generic config path when there is no tuned
                # entry for this shape.
                down_config = get_default_config(
                    M, E, inter_size, hidden_size, top_k, dtype, is_marlin, block_shape
                )
                max_block_m = down_config["BLOCK_SIZE_M"]

    if return_down_config:
        assert (
            down_config is None or config["BLOCK_SIZE_M"] == down_config["BLOCK_SIZE_M"]
        )
        return config, (down_config, max_block_m)
    return config


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int8_w8a16: bool | None = False,
    use_int4_w4a16: bool | None = False,
    use_fp8_w8a8: bool | None = False,
):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    if use_int4_w4a16:
        return "int4_w4a16"
    if use_int8_w8a16:
        return "int8_w8a16"
    if dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None
