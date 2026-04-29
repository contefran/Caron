import jax.numpy as jnp
from functools import partial
from jax import jit
from caron.physics import hllc_flux, conserved_to_primitive
import matplotlib.pyplot as plt
import numpy as np

# PLM reconstruction helpers
# ----------------------------------------------------------------------
def _minmod(a, b):
    """Element-wise minmod slope limiter."""
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.minimum(jnp.abs(a), jnp.abs(b))

def _pad_x(U, bc_x):
    if bc_x == "periodic":
        return jnp.pad(U, ((0, 0), (1, 1), (0, 0)), mode="wrap")
    if bc_x == "outflow":
        return jnp.pad(U, ((0, 0), (1, 1), (0, 0)), mode="edge")
    if bc_x == "reflective":
        left = U[:, :1, :]
        right = U[:, -1:, :]
        left = left.at[1].set(-left[1])   # flip normal momentum mx at x-wall
        right = right.at[1].set(-right[1])
        return jnp.concatenate([left, U, right], axis=1)
    raise ValueError(f"Unknown bc_x='{bc_x}'. Use 'outflow', 'reflective', or 'periodic'.")


def _pad_y(U, bc_y):
    if bc_y == "periodic":
        return jnp.pad(U, ((0, 0), (0, 0), (1, 1)), mode="wrap")
    if bc_y == "outflow":
        return jnp.pad(U, ((0, 0), (0, 0), (1, 1)), mode="edge")
    if bc_y == "reflective":
        bottom = U[:, :, :1]
        top = U[:, :, -1:]
        bottom = bottom.at[2].set(-bottom[2])  # flip normal momentum my at y-wall
        top = top.at[2].set(-top[2])
        return jnp.concatenate([bottom, U, top], axis=2)
    raise ValueError(f"Unknown bc_y='{bc_y}'. Use 'outflow', 'reflective', or 'periodic'.")


def _reconstruct_x(U, bc_x):
    """PLM reconstruction in x: returns (U_L, U_R) at each x-interface, shape (5, nx+1, ny)."""
    U_ext     = _pad_x(U, bc_x)
    dU_m      = U_ext[:, 1:-1, :] - U_ext[:, :-2, :]
    dU_p      = U_ext[:, 2:,   :] - U_ext[:, 1:-1, :]
    sigma     = _minmod(dU_m, dU_p)
    sigma_ext = _pad_x(sigma, bc_x)
    U_L = U_ext[:, :-1, :] + 0.5 * sigma_ext[:, :-1, :]
    U_R = U_ext[:, 1:,  :] - 0.5 * sigma_ext[:, 1:,  :]
    return U_L, U_R


def _reconstruct_y(U, bc_y):
    """PLM reconstruction in y: returns (U_L, U_R) at each y-interface, shape (5, nx, ny+1)."""
    U_ext     = _pad_y(U, bc_y)
    dU_m      = U_ext[:, :, 1:-1] - U_ext[:, :, :-2]
    dU_p      = U_ext[:, :, 2:  ] - U_ext[:, :, 1:-1]
    sigma     = _minmod(dU_m, dU_p)
    sigma_ext = _pad_y(sigma, bc_y)
    U_L = U_ext[:, :, :-1] + 0.5 * sigma_ext[:, :, :-1]
    U_R = U_ext[:, :, 1: ] - 0.5 * sigma_ext[:, :, 1: ]
    return U_L, U_R


def _flux_divergence(U, dx, dy, gamma, bc_x, bc_y):
    """Spatial flux divergence dFdx + dFdy using PLM-reconstructed HLLC fluxes."""
    U_L_x, U_R_x = _reconstruct_x(U, bc_x)
    F_x  = hllc_flux(U_L_x, U_R_x, gamma, direction=0)
    dFdx = (F_x[:, 1:, :] - F_x[:, :-1, :]) / dx

    U_L_y, U_R_y = _reconstruct_y(U, bc_y)
    F_y  = hllc_flux(U_L_y, U_R_y, gamma, direction=1)
    dFdy = (F_y[:, :, 1:] - F_y[:, :, :-1]) / dy

    return dFdx + dFdy


# Time integration
# ----------------------------------------------------------------------
@partial(jit, static_argnums=(5, 6))
def advance_step(U, dt, dx, dy, gamma, bc_x, bc_y):
    """SSP-RK2 (Heun) time integration with PLM reconstruction and HLLC flux.

    U: (5, nx, ny) conserved variables
    bc_x, bc_y: "outflow", "reflective", or "periodic" (JIT recompiles per unique pair)
    """
    L0 = _flux_divergence(U, dx, dy, gamma, bc_x, bc_y)
    U1 = U - dt * L0                                      # Euler predictor
    L1 = _flux_divergence(U1, dx, dy, gamma, bc_x, bc_y)
    return 0.5 * (U + U1 - dt * L1)                      # Heun corrector


def solve_euler_2d(U0, t_end, dx, dy, gamma, coords, dt, bc_x="outflow", bc_y="outflow"):
    """
    Python-side time loop with live plotting.

    - If ny == 1: 1D line plots of rho, p, vx.
    - If ny > 1:  2D colormap of rho.
    """
    U, t, step = U0, 0.0, 0
    x, y = coords

    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    while t < t_end:
        dt = float(min(dt, t_end - t))
        U  = advance_step(U, dt, dx, dy, gamma, bc_x, bc_y)
        t += dt
        step += 1
        rho = conserved_to_primitive(U, gamma)[0]

        ax.cla()
        ax.imshow(np.array(rho).T, origin="lower", aspect="auto")
        ax.set_title(f"Density — t={t:.3f}")
        plt.pause(0.01)

    plt.ioff()
    return U
