"""Classical 2D quadrant Riemann problem."""

from __future__ import annotations

import jax.numpy as jnp


def make_initial_condition(args: object) -> dict:
    nx = max(8, int(getattr(args, "sim_size", 256)))
    ny = nx

    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    x0 = 0.5 * (xmin + xmax)
    y0 = 0.5 * (ymin + ymax)

    x = jnp.linspace(xmin, xmax, num=nx)
    y = jnp.linspace(ymin, ymax, num=ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    right = X >= x0
    top = Y >= y0

    rho = jnp.where(
        right & top,
        1.0,
        jnp.where(~right & top, 2.0, jnp.where(~right & ~top, 1.0, 3.0)),
    )

    vx = jnp.where(
        right & top,
        0.75,
        jnp.where(~right & top, 0.75, jnp.where(~right & ~top, -0.75, -0.75)),
    )

    vy = jnp.where(
        right & top,
        -0.5,
        jnp.where(~right & top, 0.5, jnp.where(~right & ~top, 0.5, -0.5)),
    )

    prs = jnp.ones_like(rho)

    vel = jnp.zeros((3, nx, ny))
    vel = vel.at[0].set(vx)
    vel = vel.at[1].set(vy)

    prim = jnp.concatenate([rho[None, ...], vel, prs[None, ...]], axis=0)

    return {
        "nx": nx,
        "ny": ny,
        "t_end": 3.6,
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
