"""A simple Kelvin-Helmholtz-like shear-layer initial condition."""

from __future__ import annotations

import jax.numpy as jnp


def make_initial_condition(args: object) -> dict:
    nx = max(8, int(getattr(args, "sim_size", 200)))
    ny = nx

    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0

    x = jnp.linspace(xmin, xmax, num=nx)
    y = jnp.linspace(ymin, ymax, num=ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    shear = jnp.where((Y > 0.25) & (Y < 0.75), 0.5, -0.5)
    rho = jnp.where((Y > 0.25) & (Y < 0.75), 2.0, 1.0)
    vy = 0.01 * jnp.sin(4.0 * jnp.pi * X)
    prs = jnp.ones_like(rho)

    vel = jnp.zeros((3, nx, ny))
    vel = vel.at[0].set(shear)
    vel = vel.at[1].set(vy)

    prim = jnp.concatenate([rho[None, ...], vel, prs[None, ...]], axis=0)

    return {
        "nx": nx,
        "ny": ny,
        "t_end": 3.6,
        "dt": 0.005,
        "gamma": 1.4,
        "CFL": 0.8,
        "bc_x": "periodic",
        "bc_y": "periodic",
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "coords": (x, y),
        "prim": prim,
    }
