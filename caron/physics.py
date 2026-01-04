
import jax.numpy as jnp
from jax import jit
@jit
def primitive_to_conserved(prim, gamma):
    """Converts primitive variables to conserved variables."""

    rho = prim[0]
    vel = prim[1:4]
    prs = prim[4]
   
    mom = rho * vel                   # (3, nx, ny)
    kinetic = 0.5 * rho * jnp.sum(vel**2, axis=0)
    E = prs / (gamma - 1.0) + kinetic   # (nx, ny)
    return jnp.concatenate(
            [rho[None, ...], mom, E[None, ...]],
            axis=0
        )

@jit
def conserved_to_primitive(cons, gamma):
    """
    cons2prim conversion for 2D Euler with 3D velocity:
    U: (5, nx, ny) with
        U[0] = rho
        U[1] = rho * vx
        U[2] = rho * vy
        U[3] = rho * vz
        U[4] = E

    Returns rho, p, vel where
        rho, p: (nx, ny)
        vel:    (3, nx, ny)  -> (vx, vy, vz)
    """
    rho = cons[0]
    mom = cons[1:4]      # (3, nx, ny)
    E = cons[4]

    vel = mom / rho   # broadcast, (3, nx, ny)
    kinetic = 0.5 * rho * jnp.sum(vel**2, axis=0)
    prs = (gamma - 1.0) * (E - kinetic)

    return jnp.concatenate(
            [rho[None, ...], vel, prs[None, ...]],
            axis=0
        )

@jit
def physical_flux(U, gamma, direction: int):
    """
    Directional physical flux for 2D Euler with 3D velocity:
    direction = 0 -> x
    direction = 1 -> y

    U: (5, nx, ny)
    Returns F_dir(U): (5, nx, ny)
    """
    rho, p, vel = conserved_to_primitive(U, gamma)
    mom = U[1:4]          # (3, nx, ny)
    E = U[4]

    u_n = vel[direction]  # normal velocity in this direction, (nx, ny)

    # Mass flux
    F_rho = rho * u_n

    # Momentum flux: rho v_i u_n + p δ_{i,dir}
    F_mom = mom * u_n
    F_mom = F_mom.at[direction].add(p)

    # Energy flux: (E + p) u_n
    F_E = (E + p) * u_n

    return jnp.concatenate(
        [F_rho[None, ...], F_mom, F_E[None, ...]],
        axis=0
    )


@jit
def rusanov_flux(U_L, U_R, gamma, direction: int):
    """
    Rusanov (local Lax-Friedrichs) flux in given direction.

    U_L, U_R: (5, nx, ny) on two sides of an interface.
    """
    F_L = physical_flux(U_L, gamma, direction)
    F_R = physical_flux(U_R, gamma, direction)

    rho_L, p_L, vel_L = conserved_to_primitive(U_L, gamma)
    rho_R, p_R, vel_R = conserved_to_primitive(U_R, gamma)

    u_L = vel_L[direction]
    u_R = vel_R[direction]

    c_L = jnp.sqrt(gamma * p_L / rho_L)
    c_R = jnp.sqrt(gamma * p_R / rho_R)

    s_max = jnp.maximum(jnp.abs(u_L) + c_L, jnp.abs(u_R) + c_R)
    return 0.5 * (F_L + F_R) - 0.5 * s_max * (U_R - U_L)