# src/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass(frozen=True)
class SimConfig:
    # --- core simulation settings ---
    N: int = 300
    steps: int = 1500
    dt: float = 1.0
    box_size: float = 1.0
    save_every: int = 30

    # --- default model parameters (can be overridden in sweeps) ---
    attraction: float = 0.10
    interaction_range: float = 0.30
    noise: float = 0.03

    # --- regularisation / repulsion ---
    repulsion: float = 0.02
    repulsion_radius: float = 0.05


@dataclass(frozen=True)
class MetricsConfig:
    eps: float = 0.06
    bins: int = 20
    min_size: int = 3
    burn_frac: float = 0.6


def cfg_dict(cfg) -> Dict[str, Any]:
    """Convert dataclass config to plain dict."""
    return asdict(cfg)