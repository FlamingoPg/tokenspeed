import torch

from tokenspeed.runtime.layers.dense import fp8 as fp8_dense
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization import Fp8Config


def test_block_fp8_linear_pads_output_dim_after_loading() -> None:
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    layer = ReplicatedLinear(
        input_size=256,
        output_size=130,
        bias=False,
        quant_config=quant_config,
        params_dtype=torch.bfloat16,
    )

    layer.weight.data.zero_()
    layer.weight_scale_inv.data.fill_(1.0)
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight.shape == (256, 256)
    assert layer.weight_scale_inv.shape == (2, 2)
    assert layer._fp8_unpadded_output_size == 130


def test_block_fp8_linear_slices_padded_output(monkeypatch) -> None:
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    layer = ReplicatedLinear(
        input_size=256,
        output_size=130,
        bias=False,
        quant_config=quant_config,
        params_dtype=torch.bfloat16,
    )
    layer.weight.data.zero_()
    layer.weight_scale_inv.data.fill_(1.0)
    layer.quant_method.process_weights_after_loading(layer)

    calls = []

    def fake_mm(A, B, **kwargs):
        calls.append((A, B, kwargs))
        return torch.arange(A.shape[0] * B.shape[0], dtype=torch.bfloat16).view(
            A.shape[0],
            B.shape[0],
        )

    monkeypatch.setattr(fp8_dense.tokenspeed_kernel, "mm", fake_mm)

    output, _ = layer(torch.ones(2, 256, dtype=torch.bfloat16))

    assert output.shape == (2, 130)
    assert calls[0][1].shape == (256, 256)
    assert calls[0][2]["quant"] == "mxfp8"
