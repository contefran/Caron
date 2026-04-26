"""Registry of available simulation initial-condition presets."""

from __future__ import annotations

from typing import Callable

from .kelvin_helmholtz import make_initial_condition as make_kelvin_helmholtz
from .riemann_2d import make_initial_condition as make_riemann_2d
from .sod_shock import make_initial_condition as make_sod_shock

InitialConditionFactory = Callable[[object], dict]

_REGISTRY: dict[str, InitialConditionFactory] = {
    "riemann_2d": make_riemann_2d,
    "kelvin_helmholtz": make_kelvin_helmholtz,
    "sod_shock": make_sod_shock,
}


def available_initial_conditions() -> tuple[str, ...]:
    """Return all valid preset names sorted alphabetically."""
    return tuple(sorted(_REGISTRY.keys()))


def load_initial_condition(name: str, args: object) -> dict:
    """Build an initial-condition dictionary from its preset name."""
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(available_initial_conditions())
        raise ValueError(f"Unknown initial condition '{name}'. Available presets: {available}")
    return _REGISTRY[key](args)
