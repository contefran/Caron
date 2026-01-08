import jax.numpy as jnp
from jax import jit, lax
from physics import rusanov_flux, conserved_to_primitive
import matplotlib.pyplot as plt
import numpy as np
from time import time


def solve_euler_2d(U0,t_end,dx,dy,gamma,coords,dt):
    """
    Python-side time loop with live plotting.

    - If ny == 1: 1D line plots of rho, p, vx.
    - If ny > 1:  2D colormap of rho.
    """
    U, t, step = U0, 0.0, 0
    x, y = coords
    nx, ny = U.shape[1], U.shape[2]

    plt.ion()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    X, Y = np.meshgrid(np.array(x), np.array(y), indexing="ij")
    while t < t_end:
        dt = float(min(dt, t_end - t))  # to keep the Python loop happy
        U = advance_step(U, dt, dx, dy, gamma)
        t += dt
        step += 1

        rho = conserved_to_primitive(U, gamma)[1]

        ax.cla()
        ax.plot(x, rho[:, nx//2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Density — t={t:.3f}")


        plt.pause(0.01)


    plt.ioff()

    return U


@jit
def advance_step(U, dt, dx, dy, gamma):
    """
    Advance the solution by one step using Rusanov flux
    in both x and y directions.

    U: (5, nx, ny)
    """
    # --- x-direction update ---
    # pad in x (axis=1)
    U_ext_x = jnp.pad(U, ((0, 0), (1, 1), (0, 0)), mode="edge")
    U_L_x = U_ext_x[:, :-1, :]
    U_R_x = U_ext_x[:, 1:, :]
    F_x = rusanov_flux(U_L_x, U_R_x, gamma, direction=0)  # (5, nx+1, ny)
    dFdx = F_x[:, 1:, :] - F_x[:, :-1, :]                 # (5, nx, ny)
    dFdx = dFdx / dx

    # --- y-direction update ---
    # pad in y (axis=2)
    U_ext_y = jnp.pad(U, ((0, 0), (0, 0), (1, 1)), mode="edge")
    U_L_y = U_ext_y[:, :, :-1]
    U_R_y = U_ext_y[:, :, 1:]
    F_y = rusanov_flux(U_L_y, U_R_y, gamma, direction=1)  # (5, nx, ny+1)
    dFdy = F_y[:, :, 1:] - F_y[:, :, :-1]                 # (5, nx, ny)
    dFdy = dFdy / dy

    # Total divergence
    return U - dt * (dFdx + dFdy)