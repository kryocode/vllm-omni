# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Project-scoped H3 optimization settings, resolved before worker execution."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

_PREFIX = "VLLM_OMNI_MINIMAX_H3_"
_LAYOUT_ENV = _PREFIX + "RF_LAYOUT"
_HOIST_ENV = _PREFIX + "RF_HOIST_REORDER"
_POST_ENV = "H3_SAFE_RF_POST_EMBED"
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
    """

    profile: str
    rf_layout: str
    switches: tuple[tuple[str, str], ...]
    explicit_keys: tuple[str, ...] = ()

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, *, settings: Mapping[str, str] | None = None
    ) -> H3OptimizationConfig:
        env = dict(os.environ if environ is None else environ)
        explicit = dict(settings or {})
        env.update(explicit)
        profile = env.get(_PREFIX + "OPTIMIZATION_PROFILE", "custom").lower()
        if profile not in ("custom", "u8"):
            raise ValueError("H3 OPTIMIZATION_PROFILE must be custom or u8")
        values = {
            name: env.get(name, "1" if profile == "u8" and not name.endswith("VERIFY_FIRST") else default)
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
            layout = old_choice if hoist is not None or post is not None else "post_embed" if profile == "u8" else "none"
        elif layout not in ("none", "legacy_input_hoist", "post_embed"):
            raise ValueError("H3 RF_LAYOUT must be none, legacy_input_hoist or post_embed")
        elif _LAYOUT_ENV not in explicit:
            for name, value, selected in (
                (_HOIST_ENV, hoist, layout == "legacy_input_hoist"),
                (_POST_ENV, post, layout == "post_embed"),
            ):
                if value is not None and value != selected:
                    raise ValueError(f"H3 RF_LAYOUT conflicts with explicit {name}")
        if layout == "post_embed" and not _bool("H3_RF_PRESERVE_ADALN", env.get("H3_RF_PRESERVE_ADALN", "1")):
            raise ValueError("H3 post-embedding RF layout requires original-owner AdaLN preservation")
        if env.get("H3_RF_EXACT_POOL", "0") != "0":
            raise ValueError("H3_RF_EXACT_POOL is an unvalidated experimental overlay, not a supported profile")
        return cls(profile, layout, tuple(sorted(values.items())), tuple(sorted(explicit)))

    def activate(self, environ: MutableMapping[str, str] | None = None) -> None:
        """Publish resolved compatibility values at initialization, never per request."""
        env = os.environ if environ is None else environ
        if _PREFIX + "OPTIMIZATION_PROFILE" in self.explicit_keys:
            env[_PREFIX + "OPTIMIZATION_PROFILE"] = self.profile
        for name, value in self.switches:
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
