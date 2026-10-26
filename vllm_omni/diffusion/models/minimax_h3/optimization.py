# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Project-scoped H3 optimization settings, resolved before worker execution."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, replace

_PREFIX = "VLLM_OMNI_MINIMAX_H3_"
_LAYOUT_ENV = _PREFIX + "RF_LAYOUT"
_HOIST_ENV = _PREFIX + "RF_HOIST_REORDER"
_POST_ENV = "H3_SAFE_RF_POST_EMBED"
_GROUP_CHOICES = {
    "COMM_MODE": ("legacy", "pipeline"),
    "RF_MODE": ("native", "cached", "optimized"),
    "GRAPH_MODE": ("off", "standard", "graph2"),
    "WEIGHT_GATE": ("0", "1"),
    "WEIGHT_LAYOUT": ("nd", "nz"),
}
_LOWER_POLICIES = (
    "MINDIESD_ATTENTION_FUSED_INPUT",
    "MINDIESD_RF_GEOMETRY_CACHE",
    "MINDIESD_RF_POOLING",
)
_BOOL_DEFAULTS = {
    _PREFIX + "MINDIESD": "0",
    _PREFIX + "MINDIESD_BRIDGE": "0",
    _PREFIX + "MINDIESD_OVERLAP": "1",
    _PREFIX + "MINDIESD_VERIFY_FIRST": "0",
    "MINDIESD_ATTENTION_PIPELINE": "0",
    "MINDIESD_ATTENTION_PREPACK": "0",
    "MINDIESD_ATTENTION_D": "0",
    "MINDIESD_ATTENTION_PR658": "0",
    "MINDIESD_ATTENTION_GRAPH": "0",
    "MINDIESD_ATTENTION_GRAPH_PREFIX": "0",
    "MINDIESD_ATTENTION_GRAPH_SINGLE": "0",
    "MINDIESD_FIRST_HEAD_AG_GATE": "0",
    "MINDIESD_QBMM_NZ": "0",
}


def _bool(name: str, value: str) -> bool:
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


@dataclass(frozen=True)
class H3OptimizationConfig:
    """Defaults < compatibility env < explicit settings.

    Layout is a single choice; an explicitly disabled old layout flag also
    suppresses the preset's layout default. No paths, device selections,
    sparsity, or quantization precision are implicit in the preset.

    New group controls normalize the compatibility settings before consumers
    initialize. Contradictory group/old settings at the same priority fail;
    a group passed through ``settings`` may override old environment values.
    Without any group controls the historical precedence and paths are kept.
    RF pooling always uses the original mean path. Graph uses ordinary stable
    inputs; the former fused input mode is no longer supported.
    """

    profile: str
    rf_layout: str
    switches: tuple[tuple[str, str], ...]
    explicit_keys: tuple[str, ...] = ()
    comm_mode: str = "legacy"
    rf_mode: str = "native"
    graph_mode: str = "off"
    weight_gate: bool = False
    weight_layout: str = "nd"
    requested_groups: tuple[tuple[str, str], ...] = ()
    lower_policy_keys: tuple[str, ...] = ()
    isolated: bool = False

    @classmethod
    def from_config(cls, od_config: object) -> H3OptimizationConfig:
        """Resolve only explicit worker configuration; old env cannot select paths."""
        from vllm_omni.diffusion.optimization_config import validate_h3_execution_options

        values = validate_h3_execution_options(od_config)
        layout = values["minimax_h3_rf_reorder_mode"]
        config = cls.from_settings(
            {
                "comm_mode": values["diffusion_attn_head_overlap_mode"],
                "rf_mode": values["diffusion_attn_rf_cache_mode"],
                "rf_layout": layout,
                "graph_mode": values["diffusion_attn_head_overlap_graph_mode"],
                "weight_gate": values["dlo_allgather_gate_mode"] == "first_head",
                "weight_layout": values["dlo_weight_format"],
            }
        )
        return config

    def value(self, name: str, default: str = "0") -> str:
        """Frozen CLI policy for production; legacy readers remain test-compatible."""
        if self.isolated:
            return dict(self.switches).get(name, default)
        return os.environ.get(name, default)

    def execution_options(self) -> dict[str, object]:
        """Explicit public MindIE-SD scheduling policy."""
        if not self.isolated:
            return {}
        return {
            "graph_mode": "graph1" if self.graph_mode == "standard" else self.graph_mode,
            "rf_geometry_cache": self.value("MINDIESD_RF_GEOMETRY_CACHE") == "1",
            "rf_pooling": "legacy",
        }

    @classmethod
    def from_settings(cls, options: Mapping[str, object]) -> H3OptimizationConfig:
        """Internal settings normalizer, independent of the process environment."""
        if not isinstance(options, Mapping):
            raise TypeError("H3 internal settings must be a mapping")
        names = {
            "profile": _PREFIX + "OPTIMIZATION_PROFILE",
            "rf_layout": _LAYOUT_ENV,
            "head_chunk_size": _PREFIX + "MINDIESD_HEAD_CHUNK_SIZE",
            "overlap": _PREFIX + "MINDIESD_OVERLAP",
            "verify_first": _PREFIX + "MINDIESD_VERIFY_FIRST",
            **{name.lower(): _PREFIX + name for name in _GROUP_CHOICES},
        }
        unknown = options.keys() - names.keys()
        if unknown:
            raise ValueError(f"Unknown H3 optimization options: {', '.join(sorted(unknown))}")
        settings = {}
        for name, value in options.items():
            if name in ("weight_gate", "overlap", "verify_first"):
                if not isinstance(value, bool):
                    raise TypeError(f"H3 {name} must be a boolean")
                value = str(int(value))
            elif name == "head_chunk_size":
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("H3 head_chunk_size must be a nonnegative integer")
                value = str(value)
            elif not isinstance(value, str):
                raise TypeError(f"H3 {name} must be a string")
            settings[names[name]] = value
        config = cls.from_env({}, settings=settings)
        # Publish the complete resolved policy once, including disabled defaults,
        # so inherited switches cannot leak back through lower-level consumers.
        return replace(
            config,
            explicit_keys=tuple(
                sorted(
                    set(dict(config.switches))
                    | set(settings)
                    | {
                        _LAYOUT_ENV,
                        _PREFIX + "OPTIMIZATION_PROFILE",
                    }
                )
            ),
            lower_policy_keys=_LOWER_POLICIES,
            isolated=True,
        )

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, *, settings: Mapping[str, str] | None = None
    ) -> H3OptimizationConfig:
        env = dict(os.environ if environ is None else environ)
        explicit = dict(settings or {})
        env.update(explicit)
        # Public name for method 2; retain the historical internal value so
        # existing consumers, environment round trips and receipts stay valid.
        if _LAYOUT_ENV in env:
            layout = env[_LAYOUT_ENV].strip().lower()
            env[_LAYOUT_ENV] = "legacy_input_hoist" if layout == "pre_embed" else layout
        if env.get(_PREFIX + "GRAPH_MODE", "").strip().lower() == "graph1":
            env[_PREFIX + "GRAPH_MODE"] = "standard"
        if env.get("MINDIESD_RF_POOLING") == "fused":
            raise ValueError("MINDIESD_RF_POOLING=fused was removed; use legacy pooling")
        if env.get("MINDIESD_RF_POOLING", "legacy") != "legacy":
            raise ValueError("MINDIESD_RF_POOLING must be legacy")
        if env.get("MINDIESD_ATTENTION_FUSED_INPUT", "0") != "0":
            raise ValueError("Graph input fusion was removed; use MINDIESD_ATTENTION_FUSED_INPUT=0")
        if env.get(_PREFIX + "GRAPH_MODE", "").strip().lower() == "fused":
            raise ValueError("GRAPH_MODE=fused was removed; use standard")
        requested = {}
        for suffix, choices in _GROUP_CHOICES.items():
            name = _PREFIX + suffix
            if name in env:
                value = env[name].strip().lower()
                if value not in choices:
                    raise ValueError(f"{name} must be one of {', '.join(choices)}")
                requested[suffix] = value

        def select(suffix: str, replacements: Mapping[str, str]) -> None:
            """Same-priority contradictions fail; settings override environment.

            Profile defaults are not explicit choices. A group in settings may
            replace old environment values, but never contradictory settings.
            Normalized values remain consistent on repeated initialization.
            """
            group_name = _PREFIX + suffix
            for name, value in replacements.items():
                previous = env.get(name)
                if previous is not None and previous != value:
                    # Historical hoist spellings represent the same boolean.
                    if name == _HOIST_ENV:
                        previous = {"true": "1", "yes": "1", "on": "1", "false": "0", "no": "0", "off": "0"}.get(
                            previous.lower(), previous
                        )
                    layout_override = (
                        suffix == "RF_MODE" and name in (_HOIST_ENV, _POST_ENV) and _LAYOUT_ENV in explicit
                    )
                    if previous != value and (name in explicit or (group_name not in explicit and not layout_override)):
                        raise ValueError(f"H3 {group_name} conflicts with explicit {name}={env[name]}")
                env[name] = value

        if "COMM_MODE" in requested:
            pipeline = requested["COMM_MODE"] == "pipeline"
            replacements = {_PREFIX + "MINDIESD": str(int(pipeline)), _PREFIX + "MINDIESD_BRIDGE": str(int(pipeline))}
            if pipeline:
                replacements.update({_PREFIX + "MINDIESD_OVERLAP": "1", _PREFIX + "MINDIESD_HEAD_CHUNK_SIZE": "1"})
                replacements.update(
                    {"MINDIESD_ATTENTION_" + suffix: "1" for suffix in ("PIPELINE", "PREPACK", "D", "PR658")}
                )
            select("COMM_MODE", replacements)
        if "RF_MODE" in requested:
            mode = requested["RF_MODE"]
            if mode == "optimized":
                layout = env.get(_LAYOUT_ENV)
                if layout is None:
                    hoist = env.get(_HOIST_ENV, "0").lower() in ("1", "true", "yes", "on")
                    layout = "legacy_input_hoist" if hoist else "post_embed"
                if layout == "none":
                    # A same-priority explicit `none` is a contradiction.
                    layout = "post_embed"
            else:
                # Explicit object settings select row layout independently of
                # geometry caching. Preserve historical environment-only groups.
                layout = env[_LAYOUT_ENV] if _LAYOUT_ENV in explicit else "none"
            select(
                "RF_MODE",
                {
                    _LAYOUT_ENV: layout,
                    _HOIST_ENV: str(int(layout == "legacy_input_hoist")),
                    _POST_ENV: str(int(layout == "post_embed")),
                    "MINDIESD_RF_GEOMETRY_CACHE": str(int(mode != "native")),
                    "MINDIESD_RF_POOLING": "legacy",
                },
            )
        if "GRAPH_MODE" in requested:
            mode = requested["GRAPH_MODE"]
            select(
                "GRAPH_MODE",
                {
                    "MINDIESD_ATTENTION_GRAPH": str(int(mode != "off")),
                    "MINDIESD_ATTENTION_GRAPH_PREFIX": str(int(mode == "standard")),
                    "MINDIESD_ATTENTION_GRAPH_SINGLE": str(int(mode == "graph2")),
                    "MINDIESD_ATTENTION_FUSED_INPUT": "0",
                },
            )
        if "WEIGHT_GATE" in requested:
            select("WEIGHT_GATE", {"MINDIESD_FIRST_HEAD_AG_GATE": requested["WEIGHT_GATE"]})
        if "WEIGHT_LAYOUT" in requested:
            select("WEIGHT_LAYOUT", {"MINDIESD_QBMM_NZ": str(int(requested["WEIGHT_LAYOUT"] == "nz"))})
        profile = env.get(_PREFIX + "OPTIMIZATION_PROFILE", "custom").lower()
        if profile not in ("custom", "u8"):
            raise ValueError("H3 OPTIMIZATION_PROFILE must be custom or u8")
        values = {
            name: env.get(
                name, "1" if profile == "u8" and not name.endswith(("VERIFY_FIRST", "GRAPH_SINGLE")) else default
            )
            for name, default in _BOOL_DEFAULTS.items()
        }
        for name, value in values.items():
            _bool(name, value)
        chunk_name = _PREFIX + "MINDIESD_HEAD_CHUNK_SIZE"
        chunk = env.get(chunk_name, "1")
        if not chunk.isdecimal():
            raise ValueError(f"{chunk_name} must be a nonnegative integer")
        values[chunk_name] = chunk

        # The old input hoist accepted these spellings. Preserve that entrance.
        hoist = None
        if _HOIST_ENV in env:
            spelling = env[_HOIST_ENV].strip().lower()
            if spelling not in ("0", "false", "no", "off", "1", "true", "yes", "on"):
                raise ValueError(f"{_HOIST_ENV} must be a boolean")
            hoist = spelling in ("1", "true", "yes", "on")
        post = _bool(_POST_ENV, env[_POST_ENV]) if _POST_ENV in env else None
        if hoist and post and _LAYOUT_ENV not in explicit:
            raise ValueError("H3 input hoist and post-embedding RF layout cannot both be enabled")
        old_choice = "legacy_input_hoist" if hoist else "post_embed" if post else "none"
        layout = env.get(_LAYOUT_ENV)
        if layout is None:
            layout = (
                old_choice if hoist is not None or post is not None else "post_embed" if profile == "u8" else "none"
            )
        elif layout not in ("none", "legacy_input_hoist", "post_embed"):
            raise ValueError("H3 RF_LAYOUT must be none, pre_embed (legacy_input_hoist) or post_embed")
        else:
            for name, value, selected in (
                (_HOIST_ENV, hoist, layout == "legacy_input_hoist"),
                (_POST_ENV, post, layout == "post_embed"),
            ):
                if value is not None and value != selected and (name in explicit or _LAYOUT_ENV not in explicit):
                    raise ValueError(f"H3 RF_LAYOUT conflicts with explicit {name}")
        if layout == "post_embed" and not _bool("H3_RF_PRESERVE_ADALN", env.get("H3_RF_PRESERVE_ADALN", "1")):
            raise ValueError("H3 post-embedding RF layout requires original-owner AdaLN preservation")
        if env.get("H3_RF_EXACT_POOL", "0") != "0":
            raise ValueError("H3_RF_EXACT_POOL is an unvalidated experimental overlay, not a supported profile")
        # Resolve defaults for reporting/validation, without publishing absent
        # policies: the actual consumers retain their historical fallback paths.
        # Native RF has no cache by default; the prepared bridge caches geometry.
        comm_mode = (
            "pipeline" if values[_PREFIX + "MINDIESD"] == values[_PREFIX + "MINDIESD_BRIDGE"] == "1" else "legacy"
        )
        values["MINDIESD_ATTENTION_FUSED_INPUT"] = env.get("MINDIESD_ATTENTION_FUSED_INPUT", "0")
        values["MINDIESD_RF_GEOMETRY_CACHE"] = env.get(
            "MINDIESD_RF_GEOMETRY_CACHE", "1" if comm_mode == "pipeline" else "0"
        )
        values["MINDIESD_RF_POOLING"] = "legacy"
        for name in ("MINDIESD_ATTENTION_FUSED_INPUT", "MINDIESD_RF_GEOMETRY_CACHE"):
            _bool(name, values[name])
        if layout != "none":
            rf_mode = "optimized"
        else:
            rf_mode = "cached" if values["MINDIESD_RF_GEOMETRY_CACHE"] == "1" else "native"
        if values["MINDIESD_ATTENTION_GRAPH"] == "0":
            graph_mode = "off"
        else:
            graph_mode = "graph2" if values["MINDIESD_ATTENTION_GRAPH_SINGLE"] == "1" else "standard"
        if values["MINDIESD_ATTENTION_GRAPH_SINGLE"] == "1" and (
            graph_mode == "off" or values["MINDIESD_ATTENTION_GRAPH_PREFIX"] == "1"
        ):
            raise ValueError("Graph2 requires attention Graph enabled and conflicts with prefix/body Graph1")
        gate = values["MINDIESD_FIRST_HEAD_AG_GATE"] == "1"
        if requested:
            if graph_mode != "off" and comm_mode != "pipeline":
                raise ValueError("H3 Graph requires COMM_MODE=pipeline")
            if graph_mode != "off" and (
                rf_mode not in ("cached", "optimized") or values["MINDIESD_RF_GEOMETRY_CACHE"] != "1"
            ):
                raise ValueError("H3 Graph requires RF_MODE=cached or optimized")
            if gate and comm_mode != "pipeline":
                raise ValueError("H3 WEIGHT_GATE=1 requires COMM_MODE=pipeline")
        return cls(
            profile=profile,
            rf_layout=layout,
            switches=tuple(sorted(values.items())),
            explicit_keys=tuple(sorted(explicit)),
            comm_mode=comm_mode,
            rf_mode=rf_mode,
            graph_mode=graph_mode,
            weight_gate=gate,
            weight_layout="nz" if values["MINDIESD_QBMM_NZ"] == "1" else "nd",
            requested_groups=tuple(sorted(requested.items())),
            lower_policy_keys=tuple(name for name in _LOWER_POLICIES if name in env),
        )

    def activate(self, environ: MutableMapping[str, str] | None = None) -> None:
        """Legacy compatibility API; production workers pass objects directly."""
        if self.isolated:
            raise RuntimeError("CLI optimization settings must be passed directly; environment activation is disabled")
        env = os.environ if environ is None else environ
        if self.requested_groups:
            env.update(
                (name, value)
                for name, value in self.switches
                if name not in _LOWER_POLICIES or name in self.lower_policy_keys
            )
            for suffix, value in self.requested_groups:
                env[_PREFIX + suffix] = value
            env[_LAYOUT_ENV] = self.rf_layout
            env[_HOIST_ENV] = str(int(self.rf_layout == "legacy_input_hoist"))
            env[_POST_ENV] = str(int(self.rf_layout == "post_embed"))
            if _PREFIX + "OPTIMIZATION_PROFILE" in self.explicit_keys:
                env[_PREFIX + "OPTIMIZATION_PROFILE"] = self.profile
            return
        if _PREFIX + "OPTIMIZATION_PROFILE" in self.explicit_keys:
            env[_PREFIX + "OPTIMIZATION_PROFILE"] = self.profile
        for name, value in self.switches:
            if name in _LOWER_POLICIES and name not in self.lower_policy_keys:
                continue
            if name in self.explicit_keys:
                env[name] = value
            elif self.profile == "u8":
                env.setdefault(name, value)
        # An explicit enum is a supported replacement for the old flags.
        if (
            self.profile == "u8"
            or _LAYOUT_ENV in env
            or any(name in self.explicit_keys for name in (_LAYOUT_ENV, _HOIST_ENV, _POST_ENV))
        ):
            env[_LAYOUT_ENV] = self.rf_layout
            env[_HOIST_ENV] = str(int(self.rf_layout == "legacy_input_hoist"))
            env[_POST_ENV] = str(int(self.rf_layout == "post_embed"))

    def enabled(self, suffix: str) -> bool:
        return dict(self.switches)[_PREFIX + suffix] == "1"

    def cli_snapshot(self) -> dict[str, str]:
        """Resolved public options, including Graph2's actual gate release policy."""
        return {
            "diffusion_attn_head_overlap_mode": self.comm_mode,
            "diffusion_attn_rf_cache_mode": "cached" if self.value("MINDIESD_RF_GEOMETRY_CACHE") == "1" else "native",
            "minimax_h3_rf_reorder_mode": "pre_embed" if self.rf_layout == "legacy_input_hoist" else self.rf_layout,
            "diffusion_attn_head_overlap_graph_mode": "graph1" if self.graph_mode == "standard" else self.graph_mode,
            "dlo_allgather_gate_mode": "first_head" if self.weight_gate else "off",
            "dlo_weight_format": self.weight_layout,
            "weight_ag_release": ("block_end_for_sparse_graph2" if self.graph_mode == "graph2" else "first_head")
            if self.weight_gate
            else "immediate",
        }

    def group_snapshot(self) -> dict[str, dict[str, str]]:
        """Report requested groups and resolved policy (not runtime counters)."""
        return {
            "requested": dict(self.requested_groups),
            "effective": {
                "COMM_MODE": self.comm_mode,
                "RF_MODE": self.rf_mode,
                "GRAPH_MODE": self.graph_mode,
                "WEIGHT_GATE": str(int(self.weight_gate)),
                "WEIGHT_LAYOUT": self.weight_layout,
            },
        }
