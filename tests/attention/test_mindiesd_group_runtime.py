# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-only tests for RF policy routing; no torch or device import required."""

import importlib.util
import os
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "vllm_omni/diffusion/attention/backends/mindiesd_native.py"


@dataclass
class Metadata:
    extra: dict = field(default_factory=dict)


class Cache:
    pass


class RuntimePolicyTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("_group_backend_contract", PATH)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.layer = SimpleNamespace(
            attn_backend=SimpleNamespace(get_name=lambda: "RAINFUSION_ATTN"),
            attention=SimpleNamespace(
                _resolve_plan=lambda _: SimpleNamespace(video_spans=[(0, (1, 2, 3))]),
                rainfusion=SimpleNamespace(precision="bf16"),
            ),
            _get_active_parallel_strategy=lambda: SimpleNamespace(enabled=True, name="ulysses"),
            skip_sequence_parallel=False,
            _active_paged_kv_adapter=lambda: None,
        )

    def test_disabled_cache_does_not_import_or_attach(self):
        metadata = Metadata(extra={"mindiesd_rf_geometry_cache": object()})
        factory = self.module.NativeBackendFactory()
        with patch.dict(os.environ, {"MINDIESD_RF_GEOMETRY_CACHE": "0"}):
            with patch.object(self.module, "import_module", side_effect=AssertionError("cache imported")):
                backend = factory.build(self.layer, metadata)
        self.assertIsNone(factory.geometry_cache)
        self.assertNotIn("mindiesd_rf_geometry_cache", backend.metadata.extra)
        self.assertIn("mindiesd_rf_geometry_cache", metadata.extra)

    def test_enabled_cache_is_reused(self):
        factory = self.module.NativeBackendFactory()
        geometry = SimpleNamespace(RFGeometryCache=Cache)
        with patch.dict(os.environ, {"MINDIESD_RF_GEOMETRY_CACHE": "1"}):
            with patch.object(self.module, "import_module", return_value=geometry):
                a = factory.build(self.layer, Metadata())
                b = factory.build(self.layer, Metadata())
        self.assertIs(a.metadata.extra["mindiesd_rf_geometry_cache"], factory.geometry_cache)
        self.assertIs(b.metadata.extra["mindiesd_rf_geometry_cache"], factory.geometry_cache)

    def test_graph_rejects_disabled_static_cache(self):
        with patch.dict(os.environ, {"MINDIESD_RF_GEOMETRY_CACHE": "0"}):
            reason = self.module.NativeBackendFactory.graph_prefix_reason(self.layer, None, 1)
        self.assertIn("cache", reason)

    def test_policy_changes_change_request_contract(self):
        with patch.dict(os.environ, {}, clear=True):
            original = self.module.NativeBackendFactory.contract(self.layer)
            for name, value in (
                ("MINDIESD_RF_GEOMETRY_CACHE", "0"),
                ("MINDIESD_RF_POOLING", "legacy"),
                ("MINDIESD_ATTENTION_FUSED_INPUT", "0"),
            ):
                with patch.dict(os.environ, {name: value}):
                    self.assertNotEqual(original, self.module.NativeBackendFactory.contract(self.layer))

    def test_invalid_cache_policy_is_rejected(self):
        with patch.dict(os.environ, {"MINDIESD_RF_GEOMETRY_CACHE": "invalid"}):
            with self.assertRaises(ValueError):
                self.module.NativeBackendFactory().build(self.layer, Metadata())


if __name__ == "__main__":
    unittest.main()
