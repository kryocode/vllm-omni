# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only configuration contracts; runnable with stdlib unittest."""

import importlib.util
import itertools
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_SPEC = importlib.util.spec_from_file_location(
    "h3_optimization_under_test", _ROOT / "vllm_omni/diffusion/models/minimax_h3/optimization.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
Config = _MODULE.H3OptimizationConfig
P = "VLLM_OMNI_MINIMAX_H3_"


class OptimizationGroupsTest(unittest.TestCase):
    def test_legacy_custom_activation_does_not_add_defaults(self):
        env = {}
        config = Config.from_env(env)
        self.assertEqual(config.rf_mode, "native")
        config.activate(env)
        self.assertEqual(env, {})

    def test_weight_layout_alone_keeps_unrelated_policies_absent(self):
        env = {P + "WEIGHT_LAYOUT": "nz"}
        config = Config.from_env(env)
        self.assertEqual(config.rf_mode, "native")
        config.activate(env)
        for name in ("MINDIESD_RF_GEOMETRY_CACHE", "MINDIESD_RF_POOLING", "MINDIESD_ATTENTION_FUSED_INPUT"):
            self.assertNotIn(name, env)
        self.assertEqual(Config.from_env(env).group_snapshot(), config.group_snapshot())

    def test_old_profile_keeps_pooling_sentinel_fallback(self):
        env = {P + "OPTIMIZATION_PROFILE": "u8", P + "RF_LAYOUT": "none", "OMNI_H3_SPARSE_POOL_SPLIT": "1"}
        config = Config.from_env(env)
        self.assertEqual(config.rf_mode, "cached")
        config.activate(env)
        self.assertEqual(env["OMNI_H3_SPARSE_POOL_SPLIT"], "1")
        for name in ("MINDIESD_RF_GEOMETRY_CACHE", "MINDIESD_RF_POOLING", "MINDIESD_ATTENTION_FUSED_INPUT"):
            self.assertNotIn(name, env)

    def test_graph_group_only_publishes_its_owned_lower_policy(self):
        env = {P + "COMM_MODE": "pipeline", P + "GRAPH_MODE": "standard"}
        config = Config.from_env(env)
        self.assertEqual(config.rf_mode, "cached")
        config.activate(env)
        self.assertEqual(env["MINDIESD_ATTENTION_FUSED_INPUT"], "0")
        self.assertNotIn("MINDIESD_RF_POOLING", env)
        self.assertNotIn("MINDIESD_RF_GEOMETRY_CACHE", env)

    def test_explicit_lower_policies_are_published(self):
        env = {}
        config = Config.from_env(env, settings={"MINDIESD_RF_GEOMETRY_CACHE": "1", "MINDIESD_RF_POOLING": "legacy"})
        config.activate(env)
        self.assertEqual(env["MINDIESD_RF_GEOMETRY_CACHE"], "1")
        self.assertEqual(env["MINDIESD_RF_POOLING"], "legacy")

    def test_legacy_u8_layout_uses_standard_graph(self):
        config = Config.from_env({P + "OPTIMIZATION_PROFILE": "u8", P + "RF_LAYOUT": "legacy_input_hoist"})
        self.assertEqual(config.rf_layout, "legacy_input_hoist")
        self.assertEqual(config.graph_mode, "standard")
        self.assertEqual(config.weight_layout, "nz")

    def test_grouped_full_pipeline_publishes_and_is_idempotent(self):
        env = {
            P + "COMM_MODE": "pipeline",
            P + "RF_MODE": "optimized",
            P + "GRAPH_MODE": "standard",
            P + "WEIGHT_GATE": "1",
            P + "WEIGHT_LAYOUT": "nz",
        }
        config = Config.from_env(env)
        config.activate(env)
        self.assertEqual(env["MINDIESD_ATTENTION_PREPACK"], "1")
        self.assertEqual(env["MINDIESD_ATTENTION_FUSED_INPUT"], "0")
        self.assertEqual(env["MINDIESD_RF_GEOMETRY_CACHE"], "1")
        self.assertEqual(env["MINDIESD_RF_POOLING"], "legacy")
        self.assertEqual(env["MINDIESD_QBMM_NZ"], "1")
        self.assertEqual(Config.from_env(env).group_snapshot(), config.group_snapshot())

    def test_standard_graph_does_not_require_weight_gate(self):
        config = Config.from_env(
            {P + "COMM_MODE": "pipeline", P + "RF_MODE": "cached", P + "GRAPH_MODE": "standard", P + "WEIGHT_GATE": "0"}
        )
        values = dict(config.switches)
        self.assertEqual(values["MINDIESD_ATTENTION_GRAPH_PREFIX"], "1")
        self.assertEqual(values["MINDIESD_ATTENTION_FUSED_INPUT"], "0")
        self.assertEqual(values["MINDIESD_FIRST_HEAD_AG_GATE"], "0")
        self.assertEqual(config.rf_layout, "none")

    def test_rf_native_disables_cache_and_fused_pool(self):
        config = Config.from_env({P + "RF_MODE": "native"})
        self.assertEqual(dict(config.switches)["MINDIESD_RF_GEOMETRY_CACHE"], "0")
        self.assertEqual(dict(config.switches)["MINDIESD_RF_POOLING"], "legacy")

    def test_all_rf_profiles_keep_legacy_pooling(self):
        for env in (
            {},
            {P + "OPTIMIZATION_PROFILE": "u8"},
            {P + "RF_MODE": "optimized"},
            {P + "RF_LAYOUT": "post_embed"},
            {P + "RF_LAYOUT": "legacy_input_hoist"},
        ):
            with self.subTest(env=env):
                config = Config.from_env(env)
                self.assertEqual(dict(config.switches)["MINDIESD_RF_POOLING"], "legacy")
                config.activate(env)
                self.assertEqual(Config.from_env(env).group_snapshot(), config.group_snapshot())
                self.assertEqual(dict(Config.from_env(env).switches)["MINDIESD_RF_POOLING"], "legacy")

    def test_removed_fused_pooling_is_rejected_clearly(self):
        for env in ({}, {P + "RF_MODE": "optimized"}, {P + "OPTIMIZATION_PROFILE": "u8"}):
            for from_settings in (False, True):
                with self.subTest(env=env, from_settings=from_settings):
                    pooling = {"MINDIESD_RF_POOLING": "fused"}
                    with self.assertRaisesRegex(ValueError, "fused.*removed.*legacy"):
                        Config.from_env(
                            env if from_settings else {**env, **pooling},
                            settings=pooling if from_settings else None,
                        )

    def test_explicit_legacy_setting_replaces_stale_fused_environment(self):
        env = {"MINDIESD_RF_POOLING": "fused"}
        config = Config.from_env(env, settings={"MINDIESD_RF_POOLING": "legacy"})
        config.activate(env)
        self.assertEqual(env["MINDIESD_RF_POOLING"], "legacy")
        self.assertEqual(Config.from_env(env).group_snapshot(), config.group_snapshot())

    def test_unknown_pooling_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "MINDIESD_RF_POOLING must be legacy"):
            Config.from_env({"MINDIESD_RF_POOLING": "unknown"})

    def test_standard_graph_remains_with_original_pooling(self):
        env = {P + "COMM_MODE": "pipeline", P + "RF_MODE": "optimized", P + "GRAPH_MODE": "standard"}
        config = Config.from_env(env)
        config.activate(env)
        self.assertEqual(config.graph_mode, "standard")
        self.assertEqual(env["MINDIESD_ATTENTION_FUSED_INPUT"], "0")
        self.assertEqual(env["MINDIESD_RF_POOLING"], "legacy")

    def test_removed_graph_fusion_is_rejected(self):
        for env in ({P + "GRAPH_MODE": "fused"}, {"MINDIESD_ATTENTION_FUSED_INPUT": "1"}):
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, "removed"):
                Config.from_env(env)

    def test_graph_dependencies_are_checked(self):
        with self.assertRaisesRegex(ValueError, "pipeline"):
            Config.from_env({P + "GRAPH_MODE": "standard"})
        with self.assertRaisesRegex(ValueError, "cached|optimized"):
            Config.from_env({P + "COMM_MODE": "pipeline", P + "GRAPH_MODE": "standard", P + "RF_MODE": "native"})

    def test_weight_gate_requires_pipeline(self):
        with self.assertRaisesRegex(ValueError, "pipeline"):
            Config.from_env({P + "WEIGHT_GATE": "1"})

    def test_same_priority_conflict_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflict"):
            Config.from_env({P + "COMM_MODE": "pipeline", P + "MINDIESD": "0"})

    def test_explicit_group_settings_override_lower_priority_env(self):
        config = Config.from_env({P + "MINDIESD": "0"}, settings={P + "COMM_MODE": "pipeline"})
        env = {P + "MINDIESD": "0"}
        config.activate(env)
        self.assertEqual(env[P + "MINDIESD"], "1")
        self.assertEqual(Config.from_env(env).comm_mode, "pipeline")

    def test_group_cannot_override_higher_priority_legacy_setting(self):
        with self.assertRaisesRegex(ValueError, "conflict"):
            Config.from_env({P + "COMM_MODE": "pipeline"}, settings={P + "MINDIESD": "0"})

    def test_optimized_preserves_existing_case_layout(self):
        config = Config.from_env({P + "RF_MODE": "optimized", P + "RF_LAYOUT": "legacy_input_hoist"})
        self.assertEqual(config.rf_layout, "legacy_input_hoist")

    def test_invalid_group_values_fail(self):
        for name in ("COMM_MODE", "RF_MODE", "GRAPH_MODE", "WEIGHT_GATE", "WEIGHT_LAYOUT"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Config.from_env({P + name: "invalid"})

    def test_legacy_layout_compatibility(self):
        cases = [
            ({}, "none"),
            ({P + "OPTIMIZATION_PROFILE": "u8"}, "post_embed"),
            ({P + "OPTIMIZATION_PROFILE": "u8", "H3_SAFE_RF_POST_EMBED": "0"}, "none"),
            ({P + "RF_HOIST_REORDER": "true"}, "legacy_input_hoist"),
            ({P + "RF_LAYOUT": "post_embed", P + "RF_HOIST_REORDER": "0"}, "post_embed"),
        ]
        for env, expected in cases:
            with self.subTest(env=env):
                self.assertEqual(Config.from_env(env).rf_layout, expected)

    def test_legacy_activation_preserves_explicit_graph_disable(self):
        env = {P + "OPTIMIZATION_PROFILE": "u8", "MINDIESD_ATTENTION_GRAPH": "0", P + "RF_HOIST_REORDER": "1"}
        Config.from_env(env).activate(env)
        self.assertEqual(env["MINDIESD_ATTENTION_GRAPH"], "0")
        self.assertEqual(env[P + "RF_HOIST_REORDER"], "1")
        self.assertEqual(env["MINDIESD_QBMM_NZ"], "1")

    def test_legacy_explicit_layout_setting_supersedes_hoist(self):
        env = {P + "RF_HOIST_REORDER": "1"}
        config = Config.from_env(env, settings={P + "RF_LAYOUT": "post_embed"})
        config.activate(env)
        self.assertEqual(env[P + "RF_HOIST_REORDER"], "0")
        self.assertEqual(Config.from_env(env).rf_layout, "post_embed")

    def test_group_graph_rejects_disabled_geometry_even_with_layout(self):
        with self.assertRaisesRegex(ValueError, "cached|optimized"):
            Config.from_env(
                {P + "OPTIMIZATION_PROFILE": "u8", P + "GRAPH_MODE": "standard", "MINDIESD_RF_GEOMETRY_CACHE": "0"}
            )

    def test_explicit_rf_mode_can_replace_old_layout_flags(self):
        config = Config.from_env(
            {P + "RF_LAYOUT": "legacy_input_hoist", P + "RF_HOIST_REORDER": "1"}, settings={P + "RF_MODE": "native"}
        )
        self.assertEqual(config.rf_layout, "none")
        self.assertEqual(config.rf_mode, "native")

    def test_complete_group_matrix_and_published_round_trip(self):
        choices = [
            ("legacy", "pipeline"),
            ("native", "cached", "optimized"),
            ("off", "standard"),
            ("0", "1"),
            ("nd", "nz"),
        ]
        names = ("COMM_MODE", "RF_MODE", "GRAPH_MODE", "WEIGHT_GATE", "WEIGHT_LAYOUT")
        for combination in itertools.product(*choices):
            comm, rf, graph, gate, _ = combination
            valid = graph == "off" or (comm == "pipeline" and rf != "native")
            valid = valid and (gate == "0" or comm == "pipeline")
            env = dict(zip((P + name for name in names), combination))
            with self.subTest(combination=combination):
                if not valid:
                    with self.assertRaises(ValueError):
                        Config.from_env(env)
                    continue
                config = Config.from_env(env)
                self.assertEqual(dict(config.switches)["MINDIESD_RF_POOLING"], "legacy")
                self.assertEqual(config.group_snapshot()["effective"], dict(zip(names, combination)))
                config.activate(env)
                self.assertEqual(Config.from_env(env).group_snapshot(), config.group_snapshot())


if __name__ == "__main__":
    unittest.main()
