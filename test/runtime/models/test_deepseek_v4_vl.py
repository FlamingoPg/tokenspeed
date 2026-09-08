"""DeepSeek V4 Flash Vision: processor, routing, SWA, and wrapper seams."""

from __future__ import annotations

import dataclasses
import io
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tokenspeed_kernel.ops.attention.triton.dsv4 import dsv4_combine_topk_swa_indices
from torch import nn

from tokenspeed.runtime.configs.model_config import is_multimodal_model
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.engine.io_struct import TokenizedGenerateReqInput
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.engine.scheduler_utils import (
    make_spec,
    oversized_unsplittable_span,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_visible_window import (
    compute_visible_window_overrides,
    get_image_visible,
    image_span_aligned_extend_end,
    iter_image_spans,
    unsplittable_prefill_spans,
    visible_window_for_token,
)
from tokenspeed.runtime.models import deepseek_v4_vl as vl
from tokenspeed.runtime.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4MoEGate,
    dsv4_select_experts,
)
from tokenspeed.runtime.models.deepseek_v4_image_processor import (
    DEFAULT_VISION_MAX_N_TOKEN,
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    DeepseekV4VisionProcessConfig,
    build_image_block,
    extra_vocab_id,
    make_deepseek_v4_image_item,
    pad_deepseek_v4_input_ids,
    prepare_vl_inputs,
    vision_process_config_from_hf,
)
from tokenspeed.runtime.models.deepseek_v4_vision import DeepseekV4Vision
from tokenspeed.runtime.multimodal.embedder import pad_input_tokens
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalInputs
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils.hf_transformers_utils import (
    prefers_deepseek_v4_tokenizer,
    remap_deepseek_v4_vision_architecture,
)


class _FakeBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, hidden_size)


class _FakeLanguageModel(nn.Module):
    def __init__(self, config, **_kwargs) -> None:
        super().__init__()
        self.model = _FakeBackbone(int(config.hidden_size))
        self.lm_head = nn.Identity()
        self.logits_processor = object()
        self.forward_kwargs = None
        self.loaded_weights = None

    def forward(self, _ctx, _input_ids, _positions, **kwargs):
        self.forward_kwargs = kwargs
        return kwargs.get("input_embeds")

    def load_weights(self, weights) -> None:
        self.loaded_weights = list(weights)

    def get_embed_and_head(self):
        return self.model.embed_tokens, self.lm_head

    def set_dspark_layers_to_capture(self, layer_ids):
        self.captured_layers = list(layer_ids)

    def post_load_weights(self) -> None:
        return None


class _DummyTokenizer:
    unk_token_id = 0

    def __init__(self, placeholder_id: int = 7) -> None:
        self.placeholder_id = placeholder_id

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == IMAGE_PLACEHOLDER:
            return self.placeholder_id
        return self.unk_token_id

    def encode(self, prompt: str) -> list[int]:
        del prompt
        return [3, self.placeholder_id, 4]


def _tiny_vision_config(**overrides):
    values = {
        "hidden_size": 8,
        "vocab_size": 32,
        "vision_n_layers": 1,
        "vision_dim": 16,
        "vision_n_heads": 2,
        "vision_inter_dim": 32,
        "vision_patch_size": 2,
        "vision_rope_theta": 10000.0,
        "vision_downsample_ratio": 2,
        "vision_max_n_token": DEFAULT_VISION_MAX_N_TOKEN,
        "vision_min_pixels": 1,
        "vision_max_wh_ratio": None,
        "architectures": ["DeepseekV4ForCausalLM"],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _process_config(**overrides) -> DeepseekV4VisionProcessConfig:
    values = {
        "vision_patch_size": 14,
        "vision_downsample_ratio": 3,
        "vision_max_n_token": DEFAULT_VISION_MAX_N_TOKEN,
        "vision_min_pixels": 1,
        "vision_max_wh_ratio": None,
        "vocab_size": 100,
    }
    values.update(overrides)
    return DeepseekV4VisionProcessConfig(**values)


def _png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(40, 80, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_item(start: int, types: torch.Tensor, perm: torch.Tensor, vocab_size: int):
    n_image = int((types == IMAGE).sum())
    return make_deepseek_v4_image_item(
        start=start,
        patches=torch.zeros(n_image, 3, 2, 2),
        n_vit_h=2,
        n_vit_w=2,
        types=types,
        perm=perm,
        vocab_size=vocab_size,
        vision_max_n_token=DEFAULT_VISION_MAX_N_TOKEN,
    )


def test_remap_flash_vision_architecture_keeps_text_and_dspark():
    vision = SimpleNamespace(
        vision_n_layers=32,
        architectures=["DeepseekV4ForCausalLM"],
    )
    remap_deepseek_v4_vision_architecture(vision)
    assert vision.architectures[0] == "DeepseekV4ForConditionalGeneration"

    text = SimpleNamespace(vision_n_layers=0, architectures=["DeepseekV4ForCausalLM"])
    remap_deepseek_v4_vision_architecture(text)
    assert text.architectures[0] == "DeepseekV4ForCausalLM"

    draft = SimpleNamespace(
        vision_n_layers=32,
        architectures=["DeepseekV4ForCausalLMDSpark"],
    )
    remap_deepseek_v4_vision_architecture(draft)
    assert draft.architectures[0] == "DeepseekV4ForCausalLMDSpark"


def test_flash_vision_is_multimodal_and_uses_v4_tokenizer():
    assert is_multimodal_model(["DeepseekV4ForConditionalGeneration"])
    assert not is_multimodal_model(["DeepseekV4ForCausalLM"])
    assert prefers_deepseek_v4_tokenizer(["DeepseekV4ForConditionalGeneration"])


def test_vision_process_config_reads_flat_hf_keys():
    config = vision_process_config_from_hf(
        SimpleNamespace(
            vision_patch_size=14,
            vision_downsample_ratio=3,
            vision_max_n_token=384,
            vision_min_pixels=147456,
            vision_max_wh_ratio=8,
            vocab_size=129280,
        )
    )
    assert config.vision_max_n_token == 384
    assert config.vocab_size == 129280
    assert config.vision_max_wh_ratio == 8


def test_build_image_block_n_layout_and_compress_pad():
    types, perm = build_image_block(2, 2, 0)
    assert types[:4].tolist() == [IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, IMAGE_START]
    assert types[-1].item() == IMAGE_END
    assert types.tolist().count(IMAGE) == 4
    assert types.tolist().count(IMAGE_NEW_LINE) == 2
    assert perm.tolist() == [0, 2, 1, 3]

    types_shifted, _ = build_image_block(2, 2, 1)
    assert types_shifted[:3].tolist() == [IMAGE_PAD, IMAGE_PAD, IMAGE_START]


def test_prepare_vl_inputs_writes_official_extra_vocab_ids():
    process_config = _process_config()
    tokens, items = prepare_vl_inputs(
        "x",
        [{"data": _png_bytes(14, 14)}],
        _DummyTokenizer(),
        process_config,
    )
    assert tokens[0] == 3
    assert tokens[-1] == 4
    assert len(items) == 1
    item = items[0]
    types = item.model_specific_data["types"]
    block = tokens[item.offsets[0][0] : item.offsets[0][1] + 1]
    assert block == [
        extra_vocab_id(process_config.vocab_size, int(token_type))
        for token_type in types.tolist()
    ]
    assert block[0] >= process_config.vocab_size
    assert IMAGE_PLACEHOLDER


def test_pad_keeps_sentinels_and_hash_pads_image_only():
    vocab_size = 100
    types, perm = build_image_block(2, 2, 0)
    item = _image_item(1, types, perm, vocab_size)
    item.pad_value = 1_000_007
    official = [
        extra_vocab_id(vocab_size, int(token_type)) for token_type in types.tolist()
    ]
    input_ids = [9] + official + [11]
    hashed = pad_input_tokens(input_ids, MultimodalInputs(mm_items=[item]))
    restored = pad_deepseek_v4_input_ids(
        [9] + [7] * len(official) + [11],
        [item],
        vocab_size,
    )
    assert hashed == restored
    for token_id, token_type in zip(hashed[1:-1], types.tolist(), strict=True):
        if token_type == IMAGE:
            assert token_id == 1_000_007
        else:
            assert token_id == extra_vocab_id(vocab_size, token_type)
    assert hashed[0] == 9
    assert hashed[-1] == 11


def test_visible_window_matches_official_get_image_visible():
    vocab_size = 100
    types, perm = build_image_block(2, 2, 1)
    start = 2
    item = _image_item(start, types, perm, vocab_size)
    ids = (
        [1, 2]
        + [extra_vocab_id(vocab_size, int(token_type)) for token_type in types.tolist()]
        + [3]
    )
    input_ids = torch.tensor([ids], dtype=torch.int64)
    official_left, official_right = get_image_visible(
        input_ids,
        vocab_size,
        DEFAULT_VISION_MAX_N_TOKEN,
    )
    mm_inputs = [MultimodalInputs(mm_items=[item])]
    overrides = compute_visible_window_overrides(
        mm_inputs=mm_inputs,
        extend_prefix_lens=[0],
        extend_seq_lens=[len(ids)],
        swa_window=128,
        max_image_tokens=DEFAULT_VISION_MAX_N_TOKEN,
        padded_num_tokens=len(ids),
    )
    assert overrides is not None
    lefts, rights = overrides
    assert lefts == official_left[0].tolist()
    assert rights == official_right[0].tolist()
    assert list(iter_image_spans(mm_inputs[0])) == [(start, start + int(types.numel()))]


def test_image_span_must_stay_in_one_prefill_chunk():
    types, perm = build_image_block(2, 2, 0)
    item = _image_item(5, types, perm, 100)
    mm_input = MultimodalInputs(mm_items=[item])
    assert image_span_aligned_extend_end(mm_input, 10) == 5
    with pytest.raises(ValueError, match="crosses the prefill range"):
        compute_visible_window_overrides(
            mm_inputs=[mm_input],
            extend_prefix_lens=[0],
            extend_seq_lens=[10],
            swa_window=128,
            max_image_tokens=DEFAULT_VISION_MAX_N_TOKEN,
            padded_num_tokens=10,
        )


def test_bias_vl_score_and_hash_routing():
    logits = torch.tensor(
        [
            [0.2, 1.0, -0.5, 0.7],
            [1.5, -0.3, 0.8, 0.0],
        ],
        dtype=torch.float32,
    )
    text_bias = torch.tensor([0.0, -0.4, 0.6, 0.0], dtype=torch.float32)
    bias_vl = torch.tensor([0.5, 0.0, -0.2, 0.8], dtype=torch.float32)
    vocab_size = 10
    input_ids = torch.tensor([3, 12], dtype=torch.long)
    scores = torch.sqrt(F.softplus(logits.float()))

    _, score_ids, _ = dsv4_select_experts(
        logits,
        top_k=2,
        renormalize=True,
        correction_bias=text_bias,
        hash_indices_table=None,
        input_ids=input_ids,
        need_scores=True,
        bias_vl=bias_vl,
        vocab_size=vocab_size,
    )
    expected_score_ids = torch.stack(
        [
            torch.topk(scores[0] + text_bias, k=2, dim=-1, sorted=True).indices,
            torch.topk(scores[1] + bias_vl, k=2, dim=-1, sorted=True).indices,
        ]
    )
    assert torch.equal(score_ids, expected_score_ids.to(torch.int32))

    table = torch.tensor([[0, 1], [2, 3], [1, 0], [3, 1]], dtype=torch.int32)
    _, hash_ids, _ = dsv4_select_experts(
        logits,
        top_k=2,
        renormalize=True,
        correction_bias=None,
        hash_indices_table=table,
        input_ids=torch.tensor([3, 11], dtype=torch.long),
        need_scores=True,
        bias_vl=bias_vl,
        vocab_size=vocab_size,
    )
    expected_hash = torch.stack(
        [
            table[3],
            torch.topk(scores[1] + bias_vl, k=2, dim=-1, sorted=True).indices,
        ]
    )
    assert torch.equal(hash_ids, expected_hash.to(torch.int32))


def test_moe_gate_allocates_bias_vl_only_for_vision():
    text = DeepseekV4MoEGate(
        SimpleNamespace(
            n_routed_experts=4,
            hidden_size=8,
            num_hash_layers=0,
            topk_method="noaux_tc",
            vision_n_layers=0,
            vocab_size=16,
        ),
        layer_index=1,
    )
    assert text.bias_vl is None
    assert text.e_score_correction_bias is not None

    vision = DeepseekV4MoEGate(
        SimpleNamespace(
            n_routed_experts=4,
            hidden_size=8,
            num_hash_layers=3,
            topk_method="noaux_tc",
            vision_n_layers=32,
            vocab_size=16,
            num_experts_per_tok=2,
        ),
        layer_index=0,
    )
    assert vision.bias_vl is not None
    assert vision.tid2eid is not None
    assert vision.e_score_correction_bias is None


def test_tiny_vit_aligner_overwrites_image_slots():
    config = _tiny_vision_config()
    model = DeepseekV4Vision(config)
    with torch.no_grad():
        model.image_start.fill_(1.0)
        model.image_end.fill_(2.0)
        model.image_newline.fill_(3.0)
        model.image_pad.fill_(4.0)
    types, perm = build_image_block(1, 1, 0)
    item = make_deepseek_v4_image_item(
        start=0,
        patches=torch.randn(4, 3, 2, 2),
        n_vit_h=2,
        n_vit_w=2,
        types=types,
        perm=perm,
        vocab_size=32,
        vision_max_n_token=DEFAULT_VISION_MAX_N_TOKEN,
    )
    block = model.embed_one(item)
    assert block.shape == (int(types.numel()), config.hidden_size)
    image_mask = types == IMAGE
    assert int(image_mask.sum()) == 1
    expected = {
        IMAGE_START: 1.0,
        IMAGE_PAD: 4.0,
        IMAGE_NEW_LINE: 3.0,
        IMAGE_END: 2.0,
    }
    for index, token_type in enumerate(types.tolist()):
        if token_type == IMAGE:
            assert not torch.allclose(
                block[index],
                torch.full_like(block[index], 4.0),
            )
            continue
        torch.testing.assert_close(
            block[index],
            torch.full_like(block[index], expected[token_type]),
        )


def test_wrapper_loads_vision_tensors_and_forwards_the_rest(monkeypatch):
    monkeypatch.setattr(vl, "DeepseekV4ForCausalLM", _FakeLanguageModel)
    model = vl.DeepseekV4ForConditionalGeneration(
        _tiny_vision_config(),
        mapping=Mapping(rank=0, world_size=1),
        quant_config=None,
        is_multimodal_active=True,
        mm_attention_backend=None,
    )
    loaded = []
    expected = {}
    for index, (name, param) in enumerate(
        model.vision.named_parameters(remove_duplicate=False),
        start=1,
    ):
        value = torch.full_like(param, index / 100)
        loaded.append((name, value))
        expected[name] = value
    loaded.append(("layers.1.ffn.gate.bias_vl", torch.tensor([0.25])))
    model.load_weights(loaded)

    for name, param in model.vision.named_parameters(remove_duplicate=False):
        torch.testing.assert_close(param, expected[name])
    assert len(model.language_model.loaded_weights) == 1
    assert model.language_model.loaded_weights[0][0] == "layers.1.ffn.gate.bias_vl"
    torch.testing.assert_close(
        model.language_model.loaded_weights[0][1],
        torch.tensor([0.25]),
    )
    specs = model.get_multimodal_encoder_specs()
    assert specs[Modality.IMAGE].fn is model.image_encoder
    assert specs[Modality.IMAGE].deepstack is False


def test_wrapper_splices_prefill_and_skips_decode(monkeypatch):
    monkeypatch.setattr(vl, "DeepseekV4ForCausalLM", _FakeLanguageModel)
    model = vl.DeepseekV4ForConditionalGeneration(
        _tiny_vision_config(),
        mapping=Mapping(rank=0, world_size=1),
        quant_config=None,
        is_multimodal_active=True,
        mm_attention_backend=None,
    )
    merged = torch.randn(3, 8)
    apply_calls = []

    class FakeEmbedder:
        def apply(self, **kwargs):
            apply_calls.append(kwargs)
            return merged, {}

    model.vision_embedder = FakeEmbedder()
    multimodal_context = SimpleNamespace(has_extend_inputs=lambda: True)
    args = (torch.tensor([1, 2, 3]), torch.arange(3))
    output = model.forward(
        SimpleNamespace(forward_mode=SimpleNamespace(is_decode_or_idle=lambda: False)),
        *args,
        multimodal_context=multimodal_context,
    )
    assert output is merged
    assert model.language_model.forward_kwargs["input_embeds"] is merged
    assert apply_calls[0]["encoders"][Modality.IMAGE].fn is model.image_encoder

    output = model.forward(
        SimpleNamespace(forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True)),
        *args,
        multimodal_context=multimodal_context,
    )
    assert output is None
    assert len(apply_calls) == 1
    assert "input_embeds" not in model.language_model.forward_kwargs


def test_encoder_only_skips_language_model(monkeypatch):
    def fail_language_model(*_args, **_kwargs):
        raise AssertionError("encoder-only Flash Vision must not construct the LM")

    monkeypatch.setattr(vl, "DeepseekV4ForCausalLM", fail_language_model)
    config = _tiny_vision_config(encoder_only=True)
    model = vl.DeepseekV4ForConditionalGeneration(
        config,
        mapping=Mapping(rank=0, world_size=1),
        quant_config=None,
        is_multimodal_active=True,
        mm_attention_backend=None,
    )
    assert model.language_model is None
    assert model.vision is not None


def test_map_weight_name_does_not_eat_bias_vl():
    model = object.__new__(DeepseekV4ForCausalLM)
    assert model._map_weight_name("layers.0.ffn.gate.bias").endswith(
        "e_score_correction_bias"
    )
    assert model._map_weight_name("layers.0.ffn.gate.bias_vl").endswith("bias_vl")


def test_unsplittable_prefill_spans_are_the_image_blocks():
    types, perm = build_image_block(2, 2, 0)
    item = _image_item(5, types, perm, 100)
    mm_input = MultimodalInputs(mm_items=[item])
    assert unsplittable_prefill_spans(mm_input) == [(5, 5 + int(types.numel()))]
    assert unsplittable_prefill_spans(None) == []


def _long_image_block(start_pos: int = 0):
    types, perm = build_image_block(8, 16, start_pos)
    assert int(types.numel()) > 128
    return types, perm


def test_long_image_prepare_vl_inputs_span_exceeds_swa_window():
    process_config = _process_config(vision_min_pixels=1)
    tokens, items = prepare_vl_inputs(
        "x",
        [{"data": _png_bytes(512, 512)}],
        _DummyTokenizer(),
        process_config,
    )
    assert len(items) == 1
    start, end = items[0].offsets[0]
    span = end - start + 1
    assert span > 128
    assert unsplittable_prefill_spans(MultimodalInputs(mm_items=items)) == [
        (start, end + 1)
    ]
    types = items[0].model_specific_data["types"].tolist()
    official = [
        extra_vocab_id(process_config.vocab_size, int(token_type))
        for token_type in types
    ]
    assert tokens[start : end + 1] == official
    rel_start = types.index(IMAGE_START)
    assert tokens[start + rel_start] == extra_vocab_id(
        process_config.vocab_size, IMAGE_START
    )
    assert tokens[end] == extra_vocab_id(process_config.vocab_size, IMAGE_END)


# --- Reference: DeepSeek-V4-Flash-Vision-Exp inference/image_processor.py ---
# Ported line for line (only the bytes branch of load_image_bytes is kept) so
# the paircheck below compares against the official layout, not our own.


def _official_grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def _official_solve_resize_ratio(
    height, width, patch_size, downsample_ratio, max_n_token
):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = _official_grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def _official_safe_resize(
    height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token
):
    max_n_token -= 4 - 1  # COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = _official_grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = (
            _official_solve_resize_ratio(
                height, width, patch_size, downsample_ratio, budget
            )
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def _official_load_image(record, args):
    p = args.vision_patch_size
    with Image.open(io.BytesIO(record["data"])) as source:
        image = source.convert("RGB")
        width, height = image.size
        if (
            args.vision_max_wh_ratio is not None
            and width > height * args.vision_max_wh_ratio
        ):
            width = height * args.vision_max_wh_ratio
        if 0 < width * height < args.vision_min_pixels:
            ratio = (args.vision_min_pixels / (width * height)) ** 0.5
            width = int(width * ratio)
            height = int(height * ratio)
        best_width = math.ceil(width / p) * p
        best_height = math.ceil(height / p) * p
        n_llm_h, n_llm_w, best_height, best_width = _official_safe_resize(
            height,
            width,
            best_height,
            best_width,
            p,
            args.vision_downsample_ratio,
            args.vision_max_n_token,
        )
        n_vit_h, n_vit_w = best_height // p, best_width // p
        if (
            args.vision_max_wh_ratio is not None
            and image.width >= args.vision_max_wh_ratio * image.height
        ):
            image = image.resize((best_width, best_height))
        else:
            image = ImageOps.pad(
                image, (best_width, best_height), color=(127, 127, 127)
            )
        x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
        x = ((x - 0.5) / 0.5).to(torch.bfloat16)
        patches = (
            x.reshape(3, n_vit_h, p, n_vit_w, p)
            .permute(1, 3, 0, 2, 4)
            .reshape(n_vit_h * n_vit_w, 3, p, p)
        )
        return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def _official_build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    compress_pad = 4 - 1 - start_pos % 4
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = (
        torch.arange(rows * row_len)
        .view(rows // 2, 2, row_len)
        .transpose(1, 2)
        .reshape(-1)
    )
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, perm


def _official_prepare_vl_inputs(prompt, images, tokenizer, args):
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if image_token_id is None or image_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Token not found in tokenizer: {IMAGE_PLACEHOLDER}")
    prompt_tokens = tokenizer.encode(prompt)
    num_placeholders = sum(token == image_token_id for token in prompt_tokens)
    if num_placeholders != len(images):
        raise ValueError(
            f"Found {num_placeholders} image tokens but got {len(images)} images"
        )

    tokens, image_inputs = [], []
    image_iter = iter(images)
    for tok in prompt_tokens:
        if tok != image_token_id:
            tokens.append(tok)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = _official_load_image(
            next(image_iter), args
        )
        types, perm = _official_build_image_block(n_llm_h, n_llm_w, len(tokens))
        image_inputs.append(
            SimpleNamespace(
                start=len(tokens),
                patches=patches,
                n_vit_h=n_vit_h,
                n_vit_w=n_vit_w,
                types=types,
                perm=perm,
            )
        )
        tokens += (args.vocab_size + types).tolist()
    if not image_inputs:
        return tokens, None
    return tokens, image_inputs


def _gradient_png_bytes(width: int, height: int) -> bytes:
    """A non-uniform image so a wrong patch order or resize would show."""
    xs = np.linspace(0, 255, width, dtype=np.float32)
    ys = np.linspace(0, 255, height, dtype=np.float32)
    array = np.stack(
        [
            np.tile(xs, (height, 1)),
            np.tile(ys[:, None], (1, width)),
            np.full((height, width), 90.0, dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("width", "height"),
    [
        # 12x10 LLM grid at patch 14 / downsample 3: fits, no resize.
        (504, 420),
        # 34x24 grid would be 842 tokens: goes through safe_resize.
        (1400, 1000),
    ],
)
def test_long_image_prepare_vl_inputs_paircheck_against_official(width, height):
    """Our prepare_vl_inputs reproduces the official image_processor.py for a
    >128-token image: token ids, block position, types, aligner perm, ViT grid
    and the bf16 patches themselves."""
    process_config = _process_config()
    args = SimpleNamespace(**dataclasses.asdict(process_config))
    records = [{"data": _gradient_png_bytes(width, height)}]

    tokens, items = prepare_vl_inputs("x", records, _DummyTokenizer(), process_config)
    official_tokens, official_inputs = _official_prepare_vl_inputs(
        "x", records, _DummyTokenizer(), args
    )

    assert tokens == official_tokens
    assert len(items) == 1 and len(official_inputs) == 1
    item, official = items[0], official_inputs[0]
    assert item.offsets == [
        (official.start, official.start + official.types.numel() - 1)
    ]
    assert torch.equal(item.model_specific_data["types"], official.types)
    assert torch.equal(item.model_specific_data["perm"], official.perm)
    assert int(item.model_specific_data["n_vit_h"]) == official.n_vit_h
    assert int(item.model_specific_data["n_vit_w"]) == official.n_vit_w
    assert item.feature.dtype == official.patches.dtype == torch.bfloat16
    assert torch.equal(item.feature, official.patches)

    span_start, span_end = unsplittable_prefill_spans(
        MultimodalInputs(mm_items=[item])
    )[0]
    assert span_end - span_start == official.types.numel()
    assert span_end - span_start > 128
    assert span_end - span_start <= process_config.vision_max_n_token


def _official_get_image_visible(
    input_ids: torch.Tensor, vocab_size: int, max_image_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``get_image_visible`` from the official inference model.py, verbatim."""
    seqlen = input_ids.size(1)
    idx = torch.arange(seqlen, dtype=torch.int32).unsqueeze(0)
    is_start = input_ids == vocab_size + IMAGE_START
    is_end = input_ids == vocab_size + IMAGE_END
    valid = (is_start.cumsum(1) > is_end.cumsum(1)) | is_end
    starts = torch.where(is_start, idx, 0).cummax(1)[0]
    left = (idx - starts) * valid
    ends = torch.where(is_end, idx, seqlen).flip(1).cummin(1)[0].flip(1)
    right = (ends - idx) * valid
    return left.clamp(max=max_image_tokens - 1), right.clamp(max=max_image_tokens)


def test_long_image_visible_window_paircheck_against_official():
    vocab_size = 100
    types, perm = _long_image_block(3)
    start = 4
    item = _image_item(start, types, perm, vocab_size)
    ids = (
        [1, 2, 3, 4]
        + [extra_vocab_id(vocab_size, int(token_type)) for token_type in types.tolist()]
        + [5]
    )
    input_ids = torch.tensor([ids], dtype=torch.int64)
    official_left, official_right = _official_get_image_visible(
        input_ids,
        vocab_size,
        DEFAULT_VISION_MAX_N_TOKEN,
    )
    # The runtime port of get_image_visible must agree with the official one
    # for a span wider than the 128-token SWA window.
    runtime_left, runtime_right = get_image_visible(
        input_ids,
        vocab_size,
        DEFAULT_VISION_MAX_N_TOKEN,
    )
    assert torch.equal(runtime_left.long(), official_left.long())
    assert torch.equal(runtime_right.long(), official_right.long())
    assert int(official_left.max()) > 128
    overrides = compute_visible_window_overrides(
        mm_inputs=[MultimodalInputs(mm_items=[item])],
        extend_prefix_lens=[0],
        extend_seq_lens=[len(ids)],
        swa_window=128,
        max_image_tokens=DEFAULT_VISION_MAX_N_TOKEN,
        padded_num_tokens=len(ids),
    )
    assert overrides is not None
    lefts, rights = overrides
    assert lefts == official_left[0].tolist()
    assert rights == official_right[0].tolist()

    rel = 130
    pos = start + rel
    left_add = max(0, lefts[pos] - 127)
    assert left_add > 0
    win_start, win_end = visible_window_for_token(
        pos,
        lefts[pos],
        rights[pos],
        128,
        len(ids),
    )
    assert win_start == pos - 127 - left_add
    assert win_end == pos + rights[pos]


def test_make_spec_carries_image_spans_to_the_scheduler():
    types, perm = _long_image_block(0)
    start = 4
    item = _image_item(start, types, perm, 100)
    mm_input = MultimodalInputs(mm_items=[item])
    tokens = list(range(start + int(types.numel()) + 1))
    spec = make_spec(
        rid="r",
        tokens=tokens,
        max_new_tokens=0,
        unsplittable_spans=unsplittable_prefill_spans(mm_input),
    )
    assert spec.unsplittable_spans == [(start, start + int(types.numel()))]
    assert (
        make_spec(
            rid="t", tokens=[1, 2], max_new_tokens=0, unsplittable_spans=[]
        ).unsplittable_spans
        == []
    )


def test_oversized_unsplittable_span_is_the_first_one_wider_than_the_budget():
    assert oversized_unsplittable_span([], 16) is None
    assert oversized_unsplittable_span([(0, 16), (20, 36)], 16) is None
    assert oversized_unsplittable_span([(0, 16), (20, 37)], 16) == (20, 37)


def test_handle_generate_request_aborts_an_image_block_wider_than_the_budget():
    """An image block wider than --chunked-prefill-size can never be
    prefilled whole: the request is admitted pre-aborted with a message and
    submitted span-free, so the C++ scheduler never sees a spec it refuses."""
    types, perm = _long_image_block(0)
    start = 2
    span_len = int(types.numel())
    mm_input = MultimodalInputs(mm_items=[_image_item(start, types, perm, 100)])
    input_ids = list(range(start + span_len + 1))

    def handle(chunked_prefill_size: int):
        handler = RequestHandler.__new__(RequestHandler)
        handler.server_args = SimpleNamespace(
            chunked_prefill_size=chunked_prefill_size,
            disaggregation_bootstrap_port=None,
        )
        handler.tokenizer = None
        handler.hf_eos_token_id = [1]
        handler.max_req_len = 4096
        recv_req = TokenizedGenerateReqInput(
            rid="r",
            input_ids=input_ids,
            sampling_params=SamplingParams(),
            multimodal_inputs=mm_input,
        )
        spec, state, _bootstrap = handler.handle_generate_request(recv_req)
        return spec, state

    spec, state = handle(chunked_prefill_size=span_len)
    assert spec.unsplittable_spans == [(start, start + span_len)]
    assert state.finished_reason is None

    spec, state = handle(chunked_prefill_size=span_len - 1)
    assert spec.unsplittable_spans == []
    assert state.finished_reason is not None
    assert "--chunked-prefill-size" in state.finished_reason.message
    assert f"[{start}, {start + span_len})" in state.finished_reason.message


def _official_window_topk_idxs(window_size: int, seqlen: int) -> torch.Tensor:
    """``get_window_topk_idxs`` (start_pos=0) from the official inference code."""
    base = torch.arange(seqlen).unsqueeze(1)
    matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
    return torch.where(matrix > base, -1, matrix).int()


def _official_window_topk_idxs_visible(
    window_size: int,
    seqlen: int,
    left: torch.Tensor,
    right: torch.Tensor,
    max_image_tokens: int,
) -> torch.Tensor:
    """``get_window_topk_idxs_visible`` from the official inference code."""
    width = min(seqlen, window_size + max_image_tokens)
    idx = torch.arange(seqlen).unsqueeze(0)
    left_add = (left - (window_size - 1)).clamp(min=0)
    starts = (idx - (window_size - 1) - left_add).clamp(min=0)
    matrix = starts.unsqueeze(-1) + torch.arange(width)
    matrix = torch.where(matrix > (idx + right).unsqueeze(-1), -1, matrix)
    return matrix.int().contiguous()


def _sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    visible: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Softmax over the gathered kv slots only, like the official ``sparse_attn``."""
    gathered = kv[visible.clamp(min=0).long()]
    logits = torch.einsum("shd,swd->shw", q, gathered) * scale
    logits = logits.masked_fill((visible < 0).unsqueeze(1), float("-inf"))
    return torch.einsum("shw,swd->shd", torch.softmax(logits, dim=-1), gathered)


def test_long_image_attention_logits_paircheck_against_official():
    """The kernel-facing SWA indices for a >128-token image span reproduce the
    official ``get_window_topk_idxs_visible`` attention exactly."""
    torch.manual_seed(0)
    vocab_size = 100
    window = 128
    types, perm = _long_image_block(3)
    start = 4
    item = _image_item(start, types, perm, vocab_size)
    ids = (
        [1, 2, 3, 4]
        + [extra_vocab_id(vocab_size, int(token_type)) for token_type in types.tolist()]
        + [5, 6, 7]
    )
    seq_len = len(ids)
    assert seq_len - 7 > window

    official_left, official_right = _official_get_image_visible(
        torch.tensor([ids], dtype=torch.int64), vocab_size, DEFAULT_VISION_MAX_N_TOKEN
    )
    official = _official_window_topk_idxs_visible(
        window, seq_len, official_left, official_right, DEFAULT_VISION_MAX_N_TOKEN
    )[0]

    lefts, rights = compute_visible_window_overrides(
        mm_inputs=[MultimodalInputs(mm_items=[item])],
        extend_prefix_lens=[0],
        extend_seq_lens=[seq_len],
        swa_window=window,
        max_image_tokens=DEFAULT_VISION_MAX_N_TOKEN,
        padded_num_tokens=seq_len,
    )
    # Same call shape as the pure-SWA prefill path of the V4 backend: no
    # compressed prefix, the kv workspace holds the whole sequence.
    ours, our_lens = dsv4_combine_topk_swa_indices(
        topk_indices=torch.full((seq_len, 1), -1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        gather_lens=torch.tensor([seq_len], dtype=torch.int32),
        window_size=window,
        compress_ratio=4,
        topk=0,
        workspace_width=seq_len,
        compressed_base=0,
        block_table_base_offsets=None,
        compressed_block_size=1,
        compressed_table_capacity=None,
        swa_left=torch.tensor(lefts, dtype=torch.int32),
        swa_right=torch.tensor(rights, dtype=torch.int32),
        max_image_tokens=DEFAULT_VISION_MAX_N_TOKEN,
    )
    for pos in range(seq_len):
        visible = sorted(ours[pos, : int(our_lens[pos])].tolist())
        expected = sorted(int(v) for v in official[pos].tolist() if v >= 0)
        assert visible == expected, pos
    # Every image token sees the whole [IMAGE_START, IMAGE_END] block, which
    # is wider than the 128-token causal window on both sides.
    span_start = start + types.tolist().index(IMAGE_START)
    span_end = start + len(types) - 1 - types.tolist()[::-1].index(IMAGE_END)
    for pos in range(span_start, span_end + 1):
        visible = ours[pos, : int(our_lens[pos])].tolist()
        assert min(visible) <= span_start and max(visible) == span_end, pos

    q = torch.randn(seq_len, 2, 16, dtype=torch.float64)
    kv = torch.randn(seq_len, 16, dtype=torch.float64)
    scale = 16**-0.5
    out_ours = _sparse_attention(q, kv, ours, scale)
    out_official = _sparse_attention(q, kv, official, scale)
    torch.testing.assert_close(out_ours, out_official, rtol=0.0, atol=1e-12)
    causal = _sparse_attention(
        q, kv, _official_window_topk_idxs(window, seq_len), scale
    )
    assert not torch.allclose(
        out_official[span_start:span_end], causal[span_start:span_end]
    )
    torch.testing.assert_close(out_ours[:start], causal[:start], rtol=0.0, atol=1e-12)


def _official_gate_route(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    vocab_size: int,
    topk: int,
    bias: torch.Tensor,
    bias_vl: torch.Tensor,
    tid2eid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``Gate.forward`` from the official inference model.py (sqrt-softplus
    score_func, route_scale 1): bias shifts the top-k choice only, weights
    come from the unbiased scores."""
    scores = F.softplus(logits.float()).sqrt()
    original_scores = scores
    image_mask = input_ids >= vocab_size
    if tid2eid is not None:
        indices = tid2eid[torch.where(image_mask, 0, input_ids)]
        vl_indices = (scores + bias_vl).topk(topk, dim=-1)[1]
        indices = torch.where(
            image_mask.unsqueeze(-1), vl_indices.to(indices.dtype), indices
        )
    else:
        scores = scores + torch.where(image_mask.unsqueeze(-1), bias_vl, bias)
        indices = scores.topk(topk, dim=-1)[1]
    weights = original_scores.gather(1, indices.long())
    weights /= weights.sum(dim=-1, keepdim=True)
    return weights, indices


@pytest.mark.parametrize("hash_layer", [False, True])
def test_long_image_bias_vl_routing_paircheck_against_official_gate(hash_layer):
    """Routing for a >128-token image prompt matches the official Gate on
    both expert ids and weights, with TokenSpeed's hash-padded IMAGE slots
    standing in for the official extra-vocab ids."""
    torch.manual_seed(1)
    vocab_size = 100
    n_experts, topk = 16, 4
    start = 2
    types, perm = _long_image_block(start)
    item = _image_item(start, types, perm, vocab_size)
    item.set_pad_value()
    official_ids = (
        [5, 6]
        + [extra_vocab_id(vocab_size, int(token_type)) for token_type in types.tolist()]
        + [7]
    )
    tokenspeed_ids = pad_input_tokens(official_ids, MultimodalInputs(mm_items=[item]))
    assert len(tokenspeed_ids) - 3 > 128
    assert tokenspeed_ids != official_ids
    assert [tid >= vocab_size for tid in tokenspeed_ids] == [
        tid >= vocab_size for tid in official_ids
    ]

    logits = torch.randn(len(tokenspeed_ids), n_experts)
    bias = torch.randn(n_experts)
    bias_vl = torch.randn(n_experts)
    tid2eid = (
        torch.randint(0, n_experts, (vocab_size, topk), dtype=torch.int32)
        if hash_layer
        else None
    )
    weights, ids, _scores = dsv4_select_experts(
        logits,
        top_k=topk,
        renormalize=True,
        correction_bias=None if hash_layer else bias,
        hash_indices_table=tid2eid,
        input_ids=torch.tensor(tokenspeed_ids, dtype=torch.long),
        need_scores=True,
        bias_vl=bias_vl,
        vocab_size=vocab_size,
    )
    expected_weights, expected_ids = _official_gate_route(
        logits,
        torch.tensor(official_ids, dtype=torch.long),
        vocab_size,
        topk,
        bias,
        bias_vl,
        tid2eid,
    )
    assert torch.equal(ids.long(), expected_ids.long())
    torch.testing.assert_close(weights.float(), expected_weights.float())
    # bias_vl really steered the image tokens: text routing would differ.
    text_weights, text_ids = _official_gate_route(
        logits,
        torch.tensor([0] * len(official_ids), dtype=torch.long),
        vocab_size,
        topk,
        bias,
        bias_vl,
        tid2eid,
    )
    assert not torch.equal(ids.long()[start:-1], text_ids.long()[start:-1])
    if hash_layer:
        assert torch.equal(ids.long()[:start], tid2eid[torch.tensor([5, 6])].long())
    else:
        assert torch.equal(ids.long()[:start], text_ids.long()[:start])
        assert torch.equal(weights[:start], text_weights[:start])


def test_long_image_hash_pad_still_selects_bias_vl():
    vocab_size = 100
    types, perm = _long_image_block(0)
    item = _image_item(0, types, perm, vocab_size)
    item.pad_value = 1_000_007
    official = [
        extra_vocab_id(vocab_size, int(token_type)) for token_type in types.tolist()
    ]
    hashed = pad_input_tokens(official, MultimodalInputs(mm_items=[item]))
    image_slots = [
        token_id
        for token_id, token_type in zip(hashed, types.tolist(), strict=True)
        if token_type == IMAGE
    ]
    assert image_slots
    assert all(token_id == 1_000_007 for token_id in image_slots)
    assert all(token_id >= vocab_size for token_id in hashed)

    logits = torch.tensor(
        [
            [0.2, 1.0, -0.5, 0.7],
            [1.5, -0.3, 0.8, 0.0],
        ],
        dtype=torch.float32,
    )
    text_bias = torch.tensor([0.0, -0.4, 0.6, 0.0], dtype=torch.float32)
    bias_vl = torch.tensor([0.5, 0.0, -0.2, 0.8], dtype=torch.float32)
    scores = torch.sqrt(F.softplus(logits.float()))
    table = torch.tensor([[0, 1], [2, 3], [1, 0], [3, 1]], dtype=torch.int32)
    _, hash_ids, _ = dsv4_select_experts(
        logits,
        top_k=2,
        renormalize=True,
        correction_bias=text_bias,
        hash_indices_table=table,
        input_ids=torch.tensor([3, 1_000_007], dtype=torch.long),
        need_scores=True,
        bias_vl=bias_vl,
        vocab_size=vocab_size,
    )
    expected = torch.stack(
        [
            table[3],
            torch.topk(scores[1] + bias_vl, k=2, dim=-1, sorted=True).indices,
        ]
    )
    assert torch.equal(hash_ids, expected.to(torch.int32))
