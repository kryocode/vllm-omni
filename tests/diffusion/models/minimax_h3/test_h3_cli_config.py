# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public optimization configuration contracts, independent of accelerator imports."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
validation_spec = importlib.util.spec_from_file_location(
    "vllm_omni.diffusion.optimization_config", ROOT / "vllm_omni/diffusion/optimization_config.py"
)
validation = importlib.util.module_from_spec(validation_spec)
sys.modules[validation_spec.name] = validation
validation_spec.loader.exec_module(validation)
spec = importlib.util.spec_from_file_location(
    "h3_cli_config_test", ROOT / "vllm_omni/diffusion/models/minimax_h3/optimization.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Config = module.H3OptimizationConfig


def options(**kwargs):
    return SimpleNamespace(**kwargs)


class CLIConfigTest(unittest.TestCase):
    def test_defaults_ignore_old_environment_and_do_not_mutate_it(self):
        dirty = {"VLLM_OMNI_MINIMAX_H3_OPTIMIZATION_PROFILE": "u8", "MINDIESD_ATTENTION_GRAPH": "bad"}
        with patch.dict(os.environ, dirty, clear=True):
            config = Config.from_config(options())
            self.assertEqual(config.comm_mode, "legacy")
            self.assertEqual(config.graph_mode, "off")
            self.assertEqual(config.weight_layout, "nd")
            self.assertEqual(dict(os.environ), dirty)

    def test_full_graph_variants_and_layouts(self):
        for graph in ("graph1", "graph2"):
            for layout in ("post_embed", "pre_embed"):
                with self.subTest(graph=graph, layout=layout):
                    config = Config.from_config(
                        options(
                            diffusion_attn_head_overlap_mode="pipeline",
                            diffusion_attn_rf_cache_mode="cached",
                            minimax_h3_rf_reorder_mode=layout,
                            diffusion_attn_head_overlap_graph_mode=graph,
                            dlo_allgather_gate_mode="first_head",
                            dlo_weight_format="nz",
                            enable_distributed_layerwise_offload=True,
                            dlo_use_allgather=True,
                        )
                    )
                    self.assertTrue(config.weight_gate)
                    self.assertEqual(config.rf_mode, "optimized")
                    self.assertEqual(config.graph_mode, "graph2" if graph == "graph2" else "standard")
                    self.assertEqual(config.value("MINDIESD_ATTENTION_GRAPH_SINGLE"), str(int(graph == "graph2")))
                    self.assertEqual(config.value("MINDIESD_QBMM_NZ"), "1")

    def test_invalid_combinations_fail(self):
        invalid = [
            {"diffusion_attn_head_overlap_mode": "oops"},
            {"diffusion_attn_head_overlap_graph_mode": "graph1"},
            {
                "diffusion_attn_head_overlap_mode": "pipeline",
                "diffusion_attn_rf_cache_mode": "native",
                "diffusion_attn_head_overlap_graph_mode": "graph2",
            },
            {"diffusion_attn_head_overlap_mode": "pipeline", "dlo_allgather_gate_mode": "first_head"},
            {"dlo_weight_format": "nz"},
            {
                "diffusion_attn_head_overlap_mode": "pipeline",
                "dlo_allgather_gate_mode": "first_head",
                "enable_distributed_layerwise_offload": True,
                "dlo_use_allgather": False,
            },
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config.from_config(options(**values))

    def test_independent_cache_and_layout(self):
        config = Config.from_config(options(diffusion_attn_rf_cache_mode="cached"))
        self.assertEqual(config.rf_layout, "none")
        self.assertEqual(config.rf_mode, "cached")
        self.assertEqual(config.value("MINDIESD_RF_GEOMETRY_CACHE"), "1")

    def test_reorder_and_cache_are_independent_in_both_comm_modes(self):
        dirty = {"MINDIESD_RF_GEOMETRY_CACHE": "bad", "VLLM_OMNI_MINIMAX_H3_RF_MODE": "optimized"}
        for comm in ("legacy", "pipeline"):
            for cache in ("native", "cached"):
                for layout in ("none", "post_embed", "pre_embed"):
                    with self.subTest(comm=comm, cache=cache, layout=layout), patch.dict(os.environ, dirty, clear=True):
                        config = Config.from_config(
                            options(
                                diffusion_attn_head_overlap_mode=comm,
                                diffusion_attn_rf_cache_mode=cache,
                                minimax_h3_rf_reorder_mode=layout,
                            )
                        )
                        snapshot = config.cli_snapshot()
                        self.assertEqual(snapshot["minimax_h3_rf_reorder_mode"], layout)
                        self.assertEqual(snapshot["diffusion_attn_rf_cache_mode"], cache)
                        self.assertEqual(config.execution_options()["rf_geometry_cache"], cache == "cached")
                        self.assertEqual(config.rf_layout, "legacy_input_hoist" if layout == "pre_embed" else layout)
                        self.assertEqual(dict(os.environ), dirty)

    def test_reorder_does_not_bypass_graph_cache_requirement(self):
        for graph in ("graph1", "graph2"):
            for layout in ("post_embed", "pre_embed"):
                with self.subTest(graph=graph, layout=layout), self.assertRaisesRegex(ValueError, "Graph requires"):
                    Config.from_config(
                        options(
                            diffusion_attn_head_overlap_mode="pipeline",
                            diffusion_attn_rf_cache_mode="native",
                            minimax_h3_rf_reorder_mode=layout,
                            diffusion_attn_head_overlap_graph_mode=graph,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
