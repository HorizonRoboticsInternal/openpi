from __future__ import annotations

import dataclasses
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class EmaSpec:
    decay: float
    key: str
    tag: str


def normalize_decay_key(decay: float) -> str:
    return format(float(decay), ".12g")


def decay_tag(decay: float) -> str:
    return normalize_decay_key(decay).replace(".", "p")


def checkpoint_model_filename_for_decay(decay: float) -> str:
    return f"model_ema_{decay_tag(decay)}.safetensors"


def checkpoint_shadow_filename_for_decay(decay: float) -> str:
    return f"ema_{decay_tag(decay)}.pt"


def build_ema_specs(decays: Iterable[float]) -> tuple[EmaSpec, ...]:
    specs = []
    seen_keys: set[str] = set()
    for decay in decays:
        normalized = float(decay)
        if not (0.0 < normalized < 1.0):
            raise ValueError(
                f"EMA decay must be in (0, 1), got {normalized}")
        key = normalize_decay_key(normalized)
        if key in seen_keys:
            raise ValueError(f"Duplicate EMA decay value detected: {normalized}")
        seen_keys.add(key)
        specs.append(EmaSpec(normalized, key, decay_tag(normalized)))
    if not specs:
        raise ValueError("EMA decay list must not be empty.")
    return tuple(specs)
