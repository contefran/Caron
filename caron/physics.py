
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
    prim = conserved_to_primitive(U, gamma)
    rho, vel, p = prim[0], prim[1:4], prim[4]
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

    prim = conserved_to_primitive(U_L, gamma)
    rho_L, p_L, vel_L = prim[0], prim[4], prim[1:4]
    prim = conserved_to_primitive(U_R, gamma)
    rho_R, p_R, vel_R = prim[0], prim[4], prim[1:4]

    u_L = vel_L[direction]
    u_R = vel_R[direction]

    c_L = jnp.sqrt(gamma * p_L / rho_L)
    c_R = jnp.sqrt(gamma * p_R / rho_R)

    s_max = jnp.maximum(jnp.abs(u_L) + c_L, jnp.abs(u_R) + c_R)
    return 0.5 * (F_L + F_R) - 0.5 * s_max * (U_R - U_L)


@jit
def hllc_flux(U_L, U_R, gamma, direction: int):
    """
    HLLC Riemann solver in given direction (Toro 1994).

    U_L, U_R: (5, nx, ny) conserved variables on two sides of an interface.

    The HLLC flux adds the contact/shear wave (S*) missing from Rusanov,
    giving exact resolution of contact discontinuities and shear waves.
    """
    prim_L = conserved_to_primitive(U_L, gamma)
    prim_R = conserved_to_primitive(U_R, gamma)

    rho_L, p_L = prim_L[0], prim_L[4]
    rho_R, p_R = prim_R[0], prim_R[4]
    vel_L = prim_L[1:4]   # (3, ...)
    vel_R = prim_R[1:4]
    u_L   = vel_L[direction]
    u_R   = vel_R[direction]
    E_L   = U_L[4]
    E_R   = U_R[4]

    c_L = jnp.sqrt(gamma * p_L / rho_L)
    c_R = jnp.sqrt(gamma * p_R / rho_R)

    # Roe-averaged wave speed estimates (Davis 1988 simple bounds)
    S_L = jnp.minimum(u_L - c_L, u_R - c_R)
    S_R = jnp.maximum(u_L + c_L, u_R + c_R)

    # Contact wave speed (Toro eq. 10.37)
    num   = p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
    denom = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    S_M   = num / (denom + 1e-30)   # avoid divide-by-zero in smooth flow

    # HLLC star states  U*_K = rho_K * (S_K - u_K)/(S_K - S_M) * [...]
    def star_state(_U_K, rho_K, u_K, p_K, vel_K, E_K, S_K):
        chi   = rho_K * (S_K - u_K) / (S_K - S_M + 1e-30)
        # Tangential velocity components stay unchanged
        mom_star = chi * vel_K                                   # (3, ...)
        mom_star = mom_star.at[direction].set(chi * S_M)        # normal momentum
        E_star   = chi * (E_K / rho_K + (S_M - u_K) * (S_M + p_K / (rho_K * (S_K - u_K + 1e-30))))
        return jnp.concatenate([
            (chi)[None, ...],
            mom_star,
            E_star[None, ...],
        ], axis=0)

    U_star_L = star_state(U_L, rho_L, u_L, p_L, vel_L, E_L, S_L)
    U_star_R = star_state(U_R, rho_R, u_R, p_R, vel_R, E_R, S_R)

    F_L = physical_flux(U_L, gamma, direction)
    F_R = physical_flux(U_R, gamma, direction)

    # HLLC flux selection (Toro eq. 10.26)
    F_hllc_L = F_L + S_L * (U_star_L - U_L)
    F_hllc_R = F_R + S_R * (U_star_R - U_R)

    use_L      = S_L >= 0.0
    use_star_L = (S_L < 0.0) & (S_M >= 0.0)
    use_star_R = (S_M < 0.0) & (S_R >= 0.0)
    # else: use_R (S_R < 0)

    return jnp.where(use_L,      F_L,
           jnp.where(use_star_L, F_hllc_L,
           jnp.where(use_star_R, F_hllc_R,
                                  F_R)))