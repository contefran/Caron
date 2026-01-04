"""Simulation class producing data for the visualization."""
from typing import Any

import jax.numpy as jnp

from caron.data import Data
from caron.physics import primitive_to_conserved, conserved_to_primitive
from jax import jit
from solver import solve_euler_2d

class Simulation:
    """Produces 2D frames and pushes them into the Data buffer."""

    def __init__(self, data: Data, n_frames: int, size: int, init: str | None = None, **kwargs: Any) -> None:

        self.data = data
        self.n_frames = n_frames
        self.size = size
        self.kwargs = kwargs
        if init is None:
            self.import_init()

    def import_init(self) -> None:
        """Imports the initial conditions. Right now a 2D Sod shock tube is initialized, later more options will be added."""

        # Numerical parameters
        self.nx: int = 400
        self.ny: int = 1     # ny = 1 => effectively 1D
        self.t_end: float = 0.2
        self.dt = 0.001
        self.gamma: float = 1.4
        self.CFL: float = 0.8

        # Domain
        self.xmin: float = 0.0
        self.xmax: float = 1.0
        self.ymin: float = 0.0
        self.ymax: float = 1.0

        x0 = 0.5 * (self.xmin + self.xmax)

        x = jnp.linspace(self.xmin, self.xmax, num=self.nx)
        y = jnp.linspace(self.ymin, self.ymax, num=self.ny)
        X, _ = jnp.meshgrid(x, y, indexing="ij")

        rho = jnp.where(X < x0, 1.0, 0.125)
        prs = jnp.where(X < x0, 1.0, 0.1)

        # 3 velocity components: vx, vy, vz
        vel = jnp.zeros((3, self.nx, self.ny))

        self.coords = (x, y)
        self.prim = jnp.concatenate(
            [rho[None, ...], vel, prs[None, ...]],
            axis=0
        )
        self.data.coords = self.coords
        self.data.push_frame(self.prim[0])  # Push initial density frame to data buffer

    @property
    def dx(self):
        """
        Grid spacing in the x-direction.
        """
        return (self.xmax - self.xmin) / self.nx

    @property
    def dy(self):
        """
        Grid spacing in the y-direction.
        If ny = 1, this will be the full domain height (1D case).
        """
        # guard against ny = 0; normally ny >= 1
        return (self.ymax - self.ymin) / max(self.ny, 1)

    def run(self) -> None:
        self.cons = primitive_to_conserved(self.prim, self.gamma)

        raise NotImplementedError("Simulation.run is not implemented yet.")

    def run_no_viz(self) -> None:
        self.cons = primitive_to_conserved(self.prim, self.gamma)
        self.cons = solve_euler_2d(
            U0=self.cons,
            t_end=self.t_end,
            dx = self.dx,
            dy = self.dy,
            gamma=self.gamma,
            coords=self.coords,
            dt = self.dt,
        )
        self.prim = conserved_to_primitive(self.cons, self.gamma)