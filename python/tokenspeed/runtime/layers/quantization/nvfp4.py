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

"""NVFP4 quantization config for tokenspeed runtime (ModelOpt-produced checkpoints)."""

from __future__ import annotations

import logging
from typing import Any

import torch

from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.utils import (
    should_exclude_quant_module,
)

logger = logging.getLogger(__name__)


class Nvfp4Config(QuantizationConfig):
    """Config class for NVFP4 quantization (ModelOpt-produced checkpoints)."""

    def __init__(
        self,
        kv_cache_quant_algo: str | None = None,
        group_size: int = 16,
        exclude_modules: list[str] | None = None,
    ) -> None:
        super().__init__(exclude_modules=exclude_modules)
        self.kv_cache_quant_algo = kv_cache_quant_algo
        self.group_size = group_size
        self.weight_block_size = None  # FP4 uses group_size, not weight_block_size
        # Set by from_config when the checkpoint uses MIXED_PRECISION.
        self._has_mixed_precision: bool = False
        self._cached_mxfp8_config: QuantizationConfig | None = None

    @classmethod
    def get_name(cls) -> str:
        return "nvfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100  # Blackwell required

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Nvfp4Config":
        kv_cache_quant_algo = None
        group_size = 16
        exclude_modules = []

        # Try flat format first (config.json quantization_config)
        quant_method = config.get("quant_algo")
        if quant_method is not None:
            kv_cache_quant_algo = config.get("kv_cache_quant_algo", "auto")
            group_size = config.get("group_size", 16)
            exclude_modules = config.get("ignore", [])
        else:
            # Fall back to nested format (hf_quant_config.json)
            try:
                quant_config = cls.get_from_keys(config, ["quantization"])
                quant_method = quant_config["quant_algo"]
                kv_cache_quant_algo = quant_config.get("kv_cache_quant_algo", "auto")
                group_size = quant_config.get("group_size", 16)
                exclude_modules = quant_config.get("exclude_modules", [])
            except (ValueError, KeyError):
                raise ValueError(
                    "Cannot find quant_algo in the model quantization config."
                )

        ### TODO weicong
        if quant_method == "MIXED_PRECISION":
            return cls._from_mixed_precision_config(config)

        if quant_method != "NVFP4":
            raise ValueError(f"Nvfp4Config only supports NVFP4, got {quant_method}")

        return cls(
            kv_cache_quant_algo=kv_cache_quant_algo,
            group_size=group_size,
            exclude_modules=exclude_modules,
        )

    @classmethod
    def _from_mixed_precision_config(
        cls, config: dict[str, Any]
    ) -> "Nvfp4Config":
        """Build an Nvfp4Config from a ModelOpt MIXED_PRECISION checkpoint.

        The checkpoint ``quantized_layers`` dict maps every quantized weight
        to its ``quant_algo`` (``"MXFP8"`` or ``"NVFP4"``).  The config
        carries the original ``exclude_modules`` list and the NVFP4
        ``group_size``, plus a lazily-constructed ``Fp8Config`` for modules
        whose quant_algo is ``"MXFP8"``.
        """
        kv_cache_quant_algo = config.get("kv_cache_quant_algo", None)
        raw_exclude_modules: list[str] = config.get("exclude_modules", [])
        quantized_layers: dict[str, Any] = config.get("quantized_layers", {})

        # Normalise exclude_modules: strip the ``language_model.`` prefix
        # that ModelOpt serialises so they match the runtime parameter paths.
        exclude_modules = [
            m.removeprefix("language_model.") for m in raw_exclude_modules
        ]

        # Pick up NVFP4 group_size from any NVFP4 entry (uniformly 16).
        group_size = 16
        for _, layer_cfg in quantized_layers.items():
            if isinstance(layer_cfg, dict) and layer_cfg.get("quant_algo") == "NVFP4":
                group_size = int(layer_cfg.get("group_size", 16))
                break

        instance = cls(
            kv_cache_quant_algo=kv_cache_quant_algo,
            group_size=group_size,
            exclude_modules=exclude_modules,
        )
        instance._has_mixed_precision = True
        return instance

    @property
    def _mxfp8_config(self) -> QuantizationConfig:
        """Lazily-built Fp8Config for MXFP8 modules.

        Only valid when ``_has_mixed_precision`` is True.  The companion
        config describes ModelOpt's per-tensor MXFP8 format with E8M0 scale.
        """
        if self._cached_mxfp8_config is None:
            if not self._has_mixed_precision:
                raise RuntimeError(
                    "MXFP8 companion config is only available for "
                    "MIXED_PRECISION checkpoints."
                )
            from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config

            self._cached_mxfp8_config = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[1, 32],
                scale_fmt="ue8m0",
                quant_method="mxfp8",
            )
        return self._cached_mxfp8_config

    def resolve_quant_config(
        self, prefix: str
    ) -> QuantizationConfig | None:
        """Return the quant config to use for a runtime module *prefix*.

        * ``None`` -- the module is excluded (gate, embedding, lm_head, …).
        * ``self`` (NVFP4) -- the module holds NVFP4-packed expert weights.
        * ``self._mxfp8_config`` (Fp8Config) -- the module uses MXFP8
          (attention, dense MLP, shared experts).

        When ``_has_mixed_precision`` is ``False`` (pure NVFP4 checkpoint),
        returns ``self`` for every non-excluded prefix, preserving the
        previous uniform behaviour.
        """
        if should_exclude_quant_module(prefix, self.exclude_modules):
            return None
        if not self._has_mixed_precision:
            return self
        # Expert paths are NVFP4; everything else that is quantized is MXFP8.
        if ".experts." in prefix or prefix.endswith(".experts"):
            return self
        return self._mxfp8_config

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        """Detect NVFP4 from hf_quant_config and override."""
        quant_algo = ""
        if isinstance(hf_quant_cfg, dict):
            quant_algo = hf_quant_cfg.get("quant_algo", "")
            if not quant_algo:
                q = hf_quant_cfg.get("quantization", {})
                if isinstance(q, dict):
                    quant_algo = q.get("quant_algo", "")
        if "NVFP4" in quant_algo.upper() or "FP4" in quant_algo.upper():
            return "nvfp4"
        # Fallback: user requested nvfp4 and the checkpoint was produced by ModelOpt.
        if user_quant == "nvfp4" and hf_quant_cfg.get("quant_method") == "modelopt":
            return "nvfp4"
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []
