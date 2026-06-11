import os
import pathlib
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_REPO = pathlib.Path(__file__).resolve().parents[2]
_GLM5_NEXTN = _REPO / "python/tokenspeed/runtime/models/glm5_nextn.py"
_MODEL_CONFIG = _REPO / "python/tokenspeed/runtime/configs/model_config.py"
_HF_UTILS = _REPO / "python/tokenspeed/runtime/utils/hf_transformers_utils.py"


class TestGlm5NextNWiring(unittest.TestCase):
    """GLM5 MTP / NextN draft-model wiring (source-level guards, CPU-only).

    GLM5.1 ships a single MTP / NextN predict layer in its checkpoint
    (``model.layers.{num_hidden_layers}``). ``--speculative-algorithm MTP`` must
    dispatch to a GLM5-specific NextN class whose draft decoder is a
    :class:`GlmMoeDsaDecoderLayer` (with the GLM DSA lightning indexer) — NOT the
    DeepSeek V3 NextN, which uses a plain MLA decoder with no indexer and would
    load the wrong weights. These guard the three wiring points + the structure
    so the GLM5-specific path can't silently regress to the old DeepSeek hack.
    """

    def test_draft_architecture_routes_to_glm5_nextn(self):
        # hf_transformers_utils rewrites the draft worker's architecture; GLM5
        # must map to its own NextN class, not DeepseekV3ForCausalLMNextN.
        src = _HF_UTILS.read_text()
        self.assertIn('== "GlmMoeDsaForCausalLM"', src)
        self.assertIn('= "GlmMoeDsaForCausalLMNextN"', src)
        # The old hack (routing GLM5 to the DeepSeek V3 NextN class) is gone.
        self.assertNotIn('= "DeepseekV3ForCausalLMNextN"', src)

    def test_dsa_architectures_includes_nextn(self):
        # The draft config must be recognized as DSA (DSA attention backend /
        # KV pool / cudagraph gating all key off _DSA_ARCHITECTURES).
        src = _MODEL_CONFIG.read_text()
        self.assertIn('"GlmMoeDsaForCausalLMNextN"', src)

    def test_glm5_nextn_structure(self):
        src = _GLM5_NEXTN.read_text()
        # Entry class registered for architecture resolution.
        self.assertIn("EntryClass = [GlmMoeDsaForCausalLMNextN]", src)
        # The nextn decoder is the DSA layer, built as a single nextn layer.
        self.assertIn("GlmMoeDsaDecoderLayer(", src)
        self.assertIn("is_nextn=True", src)
        # load_weights routes DSA indexer projection weights through the fused
        # indexer loaders (FP8 wk dequant + wk_weights_proj fusion).
        self.assertIn("_try_load_fused_indexer_projection", src)
        # nextn-specific weight remap + shared embed/head skip.
        self.assertIn("num_nextn_predict_layers", src)
        self.assertIn("model.decoder", src)


class TestGlm5NextNImport(unittest.TestCase):
    """Import / registry checks. Require full runtime deps (torch + kernel);
    skipped when unavailable (e.g. CPU-only dev box)."""

    def test_nextn_is_glm5_subclass_and_registered(self):
        try:
            from tokenspeed.runtime.models.glm5 import GlmMoeDsaForCausalLM
            from tokenspeed.runtime.models.glm5_nextn import (
                EntryClass,
                GlmMoeDsaForCausalLMNextN,
            )
        except Exception as e:  # pragma: no cover - dep-gated
            self.skipTest(f"runtime deps unavailable: {e}")

        # NextN reuses GLM5's indexer-aware load path via inheritance.
        self.assertTrue(issubclass(GlmMoeDsaForCausalLMNextN, GlmMoeDsaForCausalLM))
        self.assertIn(GlmMoeDsaForCausalLMNextN, EntryClass)

    def test_registry_resolves_nextn_architecture(self):
        try:
            from tokenspeed.runtime.models.glm5_nextn import GlmMoeDsaForCausalLMNextN
            from tokenspeed.runtime.models.registry import ModelRegistry
        except Exception as e:  # pragma: no cover - dep-gated
            self.skipTest(f"runtime deps unavailable: {e}")

        cls, _ = ModelRegistry.resolve_model_cls(["GlmMoeDsaForCausalLMNextN"])
        self.assertIs(cls, GlmMoeDsaForCausalLMNextN)


if __name__ == "__main__":
    unittest.main()
