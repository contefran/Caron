"""Sod-like left/right shock tube extended on a 2D grid."""

from __future__ import annotations

import jax.numpy as jnp


def make_initial_condition(args: object) -> dict:
    nx = max(8, int(getattr(args, "sim_size", 400)))
    ny = nx

    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    x0 = 0.5 * (xmin + xmax)

    x = jnp.linspace(xmin, xmax, num=nx)
    y = jnp.linspace(ymin, ymax, num=ny)
    X, _ = jnp.meshgrid(x, y, indexing="ij")

    rho = jnp.where(X < x0, 1.0, 0.125)
    prs = jnp.where(X < x0, 1.0, 0.1)

    vel = jnp.zeros((3, nx, ny))
    prim = jnp.concatenate([rho[None, ...], vel, prs[None, ...]], axis=0)

    return {
        "nx": nx,
        "ny": ny,
        "t_end": 0.2,
        "dt": 0.001,
        "gamma": 1.4,
        "CFL": 0.8,
        "bc_x": "outflow",
        "bc_y": "outflow",
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "coords": (x, y),
        "prim": prim,
    }
