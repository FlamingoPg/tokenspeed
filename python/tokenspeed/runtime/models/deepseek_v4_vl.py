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

"""DeepSeek V4 Flash Vision wrapper around ``DeepseekV4ForCausalLM``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.layers.quantization import QuantizationConfig
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v4 import DeepseekV4ForCausalLM
from tokenspeed.runtime.models.deepseek_v4_image_processor import (
    pad_deepseek_v4_input_ids,
)
from tokenspeed.runtime.models.deepseek_v4_vision import DeepseekV4Vision
from tokenspeed.runtime.multimodal.embedder import EncoderSpec, VisionEmbedder
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalInputs
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext

logger = get_colorful_logger(__name__)

_VISION_SENTINEL_NAMES = frozenset(
    {"image_start", "image_end", "image_newline", "image_pad"}
)


def is_deepseek_v4_vision_config(config: PretrainedConfig) -> bool:
    return int(getattr(config, "vision_n_layers", 0) or 0) > 0


class DeepseekV4ForConditionalGeneration(nn.Module):
    """Vision-capable DeepSeek V4 registered under a distinct architecture.

    The HF checkpoint still says ``DeepseekV4ForCausalLM``. ``get_config``
    remaps ``vision_n_layers > 0`` onto this wrapper so text V4-Flash stays
    on the original entry class.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        is_multimodal_active: bool,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        del mm_attention_backend
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.is_multimodal_active = (
            is_multimodal_active and is_deepseek_v4_vision_config(config)
        )

        self.language_model = None
        if not getattr(config, "encoder_only", False):
            self.language_model = DeepseekV4ForCausalLM(
                config=config,
                mapping=mapping,
                quant_config=quant_config,
                prefix="",
                encoder_only=False,
            )

        if self.is_multimodal_active:
            self.vision = DeepseekV4Vision(config)
            if (
                self.language_model is not None
                and self.language_model.model is not None
                and self.language_model.model.embed_tokens is not None
            ):
                target_dtype = self.get_input_embeddings().weight.dtype
                self.vision = self.vision.to(dtype=target_dtype)
            self.vision_embedder = VisionEmbedder(encoder_mapping=mapping.vision)
            self.image_encoder = self.vision.embed_media
        else:
            self.vision = None
            self.vision_embedder = None
            self.image_encoder = None

    def get_input_embeddings(self) -> nn.Module:
        if self.language_model is None:
            raise AttributeError(
                "DeepSeek V4 encoder-only mode does not expose text embeddings."
            )
        return self.language_model.model.embed_tokens

    @property
    def logits_processor(self):
        if self.language_model is None:
            raise AttributeError(
                "DeepSeek V4 encoder-only mode does not expose a logits processor."
            )
        return self.language_model.logits_processor

    @property
    def lm_head(self):
        if self.language_model is None:
            raise AttributeError(
                "DeepSeek V4 encoder-only mode does not expose an LM head."
            )
        return self.language_model.lm_head

    @property
    def model(self):
        if self.language_model is None:
            raise AttributeError(
                "DeepSeek V4 encoder-only mode does not expose a text backbone."
            )
        return self.language_model.model

    def get_embed_and_head(self):
        return self.language_model.get_embed_and_head()

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        if self.language_model is None:
            raise AttributeError(
                "DeepSeek V4 encoder-only mode cannot capture target hidden states."
            )
        self.language_model.set_dspark_layers_to_capture(layer_ids)

    def get_multimodal_encoder_specs(self) -> dict[Modality, EncoderSpec]:
        if self.vision is None or self.image_encoder is None:
            return {}
        return {
            Modality.IMAGE: EncoderSpec(
                fn=self.image_encoder,
                deepstack=False,
                make_warmup_items=self.vision.make_image_warmup_items,
            )
        }

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        return pad_deepseek_v4_input_ids(
            input_ids,
            mm_inputs.mm_items,
            int(self.config.vocab_size),
        )

    @torch.no_grad()
    def multimodal_input_embeds(
        self,
        input_ids: torch.Tensor,
        ctx: "ForwardContext",
        multimodal_context,
    ) -> torch.Tensor | None:
        if (
            multimodal_context is None
            or self.vision_embedder is None
            or not multimodal_context.has_extend_inputs()
            or ctx.forward_mode.is_decode_or_idle()
            or not self.mapping.is_first_pp_rank
        ):
            return None
        input_embeds, model_kwargs = self.vision_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.get_input_embeddings(),
            ctx=multimodal_context,
            encoders=self.get_multimodal_encoder_specs(),
            multimodal_model=self,
        )
        if model_kwargs:
            raise RuntimeError("DeepSeek V4 multimodal path must stay embeds-only")
        return input_embeds

    @torch.no_grad()
    def forward(
        self,
        ctx: "ForwardContext",
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.language_model is None:
            raise RuntimeError(
                "DeepSeek V4 encoder-only mode cannot execute language-model forward."
            )
        multimodal_context = kwargs.pop("multimodal_context", None)
        input_embeds = self.multimodal_input_embeds(input_ids, ctx, multimodal_context)
        if input_embeds is not None:
            kwargs["input_embeds"] = input_embeds
        return self.language_model.forward(
            ctx,
            input_ids,
            positions,
            **kwargs,
        )

    def post_load_weights(self) -> None:
        if self.language_model is not None:
            self.language_model.post_load_weights()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        vision_params = (
            dict(self.vision.named_parameters(remove_duplicate=False))
            if self.vision is not None
            else {}
        )
        loaded_vision = 0

        def language_weights():
            nonlocal loaded_vision
            for name, weight in weights:
                mapped = name[6:] if name.startswith("model.") else name
                if mapped.startswith("vision.") or mapped.startswith("aligner."):
                    param_name = mapped
                elif mapped in _VISION_SENTINEL_NAMES:
                    param_name = mapped
                else:
                    yield name, weight
                    continue
                if self.vision is None:
                    continue
                param = vision_params.get(param_name)
                if param is None:
                    raise ValueError(
                        f"DeepSeek V4 vision checkpoint has unexpected tensor {name}"
                    )
                default_weight_loader(param, weight)
                loaded_vision += 1

        if self.language_model is not None:
            self.language_model.load_weights(language_weights())
        else:
            for _ in language_weights():
                pass
        logger.debug("Loaded %d DeepSeek V4 vision tensors.", loaded_vision)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return DeepseekV4ForCausalLM.get_model_config_for_expert_location(config)


EntryClass = [DeepseekV4ForConditionalGeneration]
