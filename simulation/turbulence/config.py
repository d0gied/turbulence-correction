from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TurbulenceConfig:
    turbulence_strength: float = 5.0
    sigma: float = 6.0
    turb_scale: float = 1 / 64
    temporal_scale: float = 2.0
    levels: int = 8
    make_fast: bool = False
    blur_levels: int = 10
    rng_seed: Optional[int] = None
    noise_scale: int = 2
