"""DeepSeek V4 runtime consumes precomputed image patches and token metadata."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.configs.deepseek_v4_config import (
    DeepseekV4Config,
)
from tokenspeed.runtime.configs.deepseek_v4_config import (
    DeepseekV4ImageTokenType as ImageTokenType,
)
from tokenspeed.runtime.configs.model_config import is_multimodal_model
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.engine.io_struct import TokenizedGenerateReqInput
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
    DEFAULT_VISION_MAX_N_TOKEN,
    build_image_window,
)
from tokenspeed.runtime.layers.attention.mm_encoder_attention import VisionAttention
from tokenspeed.runtime.models import deepseek_v4_vl as vl
from tokenspeed.runtime.models.deepseek_v4 import dsv4_select_experts
from tokenspeed.runtime.models.deepseek_v4_vl import DeepseekV4Vision
from tokenspeed.runtime.multimodal.embedder import pad_input_tokens
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalInputs
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils.hf_transformers_utils import (
    get_config,
)


class _FakeLanguageModel(nn.Module):
    def __init__(self, config, **_kwargs):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.forward_kwargs = None
        self.loaded_weights = None

    def forward(self, _ctx, _input_ids, _positions, **kwargs):
        self.forward_kwargs = kwargs
        return kwargs.get("input_embeds")

    def load_weights(self, weights):
        self.loaded_weights = list(weights)


def _tiny_config():
    return DeepseekV4Config(
        hidden_size=8,
        vocab_size=100,
        vision_n_layers=1,
        vision_dim=16,
        vision_n_heads=2,
        vision_inter_dim=32,
        vision_patch_size=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=2,
        vision_max_n_token=DEFAULT_VISION_MAX_N_TOKEN,
        architectures=["DeepseekV4ForCausalLM"],
    )


@pytest.fixture(params=[False])
def model(monkeypatch, request):
    monkeypatch.setattr(vl, "DeepseekV4ForCausalLM", _FakeLanguageModel)
    config = _tiny_config()
    config.encoder_only = request.param
    return vl.DeepseekV4ForConditionalGeneration(
        config,
        mapping=Mapping(rank=0, world_size=1),
        quant_config=None,
        is_multimodal_active=True,
        mm_attention_backend="triton_attn",
    )


@pytest.fixture
def item(model):
    item = model.vision.make_image_warmup_items()[0]
    item.pad_value = 1_000_007
    item.model_specific_data["vocab_size"] = torch.tensor(100)
    return item


@pytest.mark.parametrize(
    ("layers", "architecture"),
    [
        (1, "DeepseekV4ForCausalLM"),
        (0, "DeepseekV4ForCausalLM"),
        (None, "DeepseekV4ForCausalLM"),
        (1, "DeepseekV4ForCausalLMDSpark"),
        (1, None),
        (1, ""),
    ],
)
def test_vision_architecture(layers, architecture, tmp_path):
    architectures = (
        [architecture] if architecture else ([] if architecture == "" else None)
    )
    config = DeepseekV4Config(architectures=architectures, vocab_size=100)
    if layers is not None:
        config.vision_n_layers = layers
    vision = layers == 1 and architecture == "DeepseekV4ForCausalLM"
    expected = ["DeepseekV4ForConditionalGeneration"] if vision else architectures
    config.save_pretrained(tmp_path)
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert config.architectures == expected
    assert is_multimodal_model(config.architectures) == vision


@pytest.mark.parametrize("hash_layer", [False, True])
def test_visual_routing(hash_layer):
    _, ids, _ = dsv4_select_experts(
        torch.tensor([[0.2, 1.0, -0.5, 0.7]]).repeat(2, 1),
        top_k=2,
        renormalize=True,
        need_scores=True,
        correction_bias=torch.tensor([0.0, -0.4, 0.6, 0.0]),
        hash_indices_table=torch.tensor([[1, 2]] * 100) if hash_layer else None,
        input_ids=torch.tensor([3, 1_000_007]),
        vocab_size=100,
        bias_vl=torch.tensor([0.5, 0.0, -0.2, 0.8]),
    )
    assert ids.tolist() == [[1, 2] if hash_layer else [2, 3], [3, 0]]


@pytest.mark.parametrize("upstream_vocab", [None, 999])
def test_padding_preserves_sentinels(model, item, upstream_vocab):
    types = item.model_specific_data["types"]
    tokens = (100 + types).tolist() + [4]
    mm_inputs = MultimodalInputs(mm_items=[item], im_token_id=102)
    expected = torch.where(
        types == ImageTokenType.IMAGE, item.pad_value, 100 + types
    ).tolist() + [4]
    assert pad_input_tokens(tokens, mm_inputs) == expected
    if upstream_vocab is None:
        item.model_specific_data.pop("vocab_size")
    else:
        item.model_specific_data["vocab_size"] = torch.tensor(upstream_vocab)
    assert model.pad_input_ids([7] * len(types) + [4], mm_inputs) == expected


@pytest.mark.parametrize(("prefix", "length"), [(0, 144), (0, 10), (10, 134)])
def test_image_prefill_boundaries(item, prefix, length):
    types = (
        [ImageTokenType.PAD] * 3
        + [ImageTokenType.START]
        + [ImageTokenType.IMAGE] * 136
        + [ImageTokenType.END]
    )
    item.model_specific_data["types"] = torch.tensor(types)
    item.offsets = [(2, 142)]
    kwargs = dict(
        mm_inputs=[MultimodalInputs(mm_items=[item]), None],
        prefix_lens=[prefix, 0],
        query_lens=[length, 3],
        device=torch.device("cpu"),
    )
    if prefix or length < 144:
        with pytest.raises(ValueError, match="crosses the prefill range"):
            build_image_window(**kwargs)
    else:
        left, right, max_image_tokens = build_image_window(**kwargs)
        assert max_image_tokens == 384
        assert left.tolist() == [0] * 5 + list(range(138)) + [0] * 4
        assert right.tolist() == [0] * 5 + list(reversed(range(138))) + [0] * 4


@pytest.mark.parametrize("budget_delta", [0, -1])
@pytest.mark.parametrize("validation_error", [None, "malformed request"])
def test_request_handler_checks_image_budget(item, budget_delta, validation_error):
    start, end = item.offsets[0]
    span_length = end - start + 1
    handler = RequestHandler.__new__(RequestHandler)
    handler.server_args = SimpleNamespace(
        chunked_prefill_size=span_length + budget_delta,
        disaggregation_bootstrap_port=None,
    )
    handler.tokenizer = None
    handler.hf_eos_token_id = [1]
    handler.max_req_len = 4096
    request = TokenizedGenerateReqInput(
        rid="r",
        input_ids=(100 + item.model_specific_data["types"]).tolist() + [4],
        sampling_params=SamplingParams(),
        multimodal_inputs=MultimodalInputs(mm_items=[item]),
        validation_error=validation_error,
    )
    spec, state, _ = handler.handle_generate_request(request)
    assert spec.unsplittable_spans == [(start, end + 1)]
    if validation_error:
        assert state.finished
        assert state.finished_reason.message == f"Invalid request: {validation_error}"
    elif budget_delta == 0:
        assert state.finished_reason is None
    else:
        assert state.finished
        assert "--chunked-prefill-size" in state.finished_reason.message
        assert f"[{start}, {end + 1})" in state.finished_reason.message


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("downsample_ratio", "backend"),
    [(2, "triton_attn"), (3, None), (2, "flashinfer_cudnn")],
)
def test_vision_embeddings(downsample_ratio, backend, monkeypatch):
    if backend == "flashinfer_cudnn" and torch.version.hip is not None:
        pytest.skip("cuDNN requires NVIDIA")
    config = _tiny_config()
    config.vision_downsample_ratio = downsample_ratio
    config.vision_dim = 128
    vision = (
        DeepseekV4Vision(config, Mapping(rank=0, world_size=1), backend)
        .cuda()
        .bfloat16()
    )
    assert isinstance(vision.vision.blocks[0].attn, VisionAttention)
    with torch.no_grad():
        for parameter in vision.parameters():
            parameter.uniform_(-0.1, 0.1)
    item = vision.make_image_warmup_items()[0]
    assert item.offsets == [(0, 8)]
    assert item.feature.shape == (downsample_ratio**2, 3, 2, 2)
    assert item.feature.dtype == vision.vision.patch_embed.proj.weight.dtype
    item.feature.normal_()
    types = item.model_specific_data["types"]
    assert types.tolist() == [ImageTokenType.PAD] * 3 + [
        ImageTokenType.START,
        ImageTokenType.IMAGE,
        ImageTokenType.PAD,
        ImageTokenType.NEW_LINE,
        ImageTokenType.PAD,
        ImageTokenType.END,
    ]
    block = vision.embed_one(item)
    assert block.shape == (9, 8)

    def sdpa(q, k, v, **kwargs):
        return F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        ).transpose(0, 1)

    for layer in vision.vision.blocks:
        monkeypatch.setattr(layer.attn, "_backend_fn", sdpa)
    expected = vision.encode_image(
        item.feature.to(vision.image_start), downsample_ratio, downsample_ratio
    )
    torch.testing.assert_close(
        block[types == ImageTokenType.IMAGE], expected, atol=2e-3, rtol=2e-2
    )
    for kind, parameter in (
        (ImageTokenType.START, vision.image_start),
        (ImageTokenType.END, vision.image_end),
        (ImageTokenType.NEW_LINE, vision.image_newline),
        (ImageTokenType.PAD, vision.image_pad),
    ):
        torch.testing.assert_close(
            block[types == kind], parameter.expand_as(block[types == kind])
        )
    types[0] = ImageTokenType.IMAGE
    with pytest.raises(ValueError, match="IMAGE slots"):
        vision.embed_one(item)


@pytest.mark.parametrize("model", [False, True], indirect=True)
def test_wrapper_loads_vision_weights(model):
    parameters = dict(model.vision.named_parameters(remove_duplicate=False))
    expected = {name: torch.randn_like(param) for name, param in parameters.items()}
    language_weight = ("layers.1.ffn.gate.bias_vl", torch.tensor([0.25]))
    checkpoint = [
        (
            name.replace(".attn.qkv_proj.", ".attn.wqkv.").replace(
                ".attn.proj.", ".attn.wo."
            ),
            value,
        )
        for name, value in expected.items()
    ]
    model.load_weights([*checkpoint, language_weight])
    for name, param in parameters.items():
        torch.testing.assert_close(param, expected[name])
    if model.language_model is not None:
        assert model.language_model.loaded_weights == [language_weight]
    spec = model.get_multimodal_encoder_specs()[Modality.IMAGE]
    assert spec.fn is model.image_encoder
    assert spec.deepstack is False
    with pytest.raises(KeyError, match="vision.nonexistent"):
        model.load_weights([("vision.nonexistent", torch.ones(1))])


def test_wrapper_splices_prefill_and_skips_decode(model):
    merged = torch.randn(3, 8)
    apply = Mock(return_value=(merged, {}))
    model.vision_embedder = SimpleNamespace(apply=apply)
    for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
        output = model.forward(
            SimpleNamespace(forward_mode=mode),
            torch.tensor([1, 2, 3]),
            torch.arange(3),
            multimodal_context=SimpleNamespace(has_extend_inputs=lambda: True),
        )
        if mode == ForwardMode.EXTEND:
            assert output is merged
            assert model.language_model.forward_kwargs["input_embeds"] is merged
        else:
            assert output is None
            assert "input_embeds" not in model.language_model.forward_kwargs
        apply.assert_called_once()
    assert apply.call_args.kwargs["encoders"][Modality.IMAGE].fn is model.image_encoder


@pytest.mark.parametrize("model", [True], indirect=True)
def test_encoder_only_skips_language_model(model):
    assert model.language_model is None
    assert model.vision is not None
    with pytest.raises(AttributeError):
        model.get_input_embeddings()
    with pytest.raises(RuntimeError, match="encoder-only"):
        model.forward(None, None, None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
