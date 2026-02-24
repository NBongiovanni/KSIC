from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

G = 9.81

class Plant:
    """
    Base class for plants (continuous or discrete).
    Enforces a single dynamics signature: f(t, x, u) -> x_dot or x_next.
    """

    def __init__(self, dt: float, discrete_or_continuous: str = "continuous"):
        self.dt = float(dt)
        self.discrete_or_continuous = discrete_or_continuous  # "continuous" or "discrete"

        # to be set by subclasses
        self.drone_dim: int | None = None
        self.x_dim: int | None = None
        self.u_dim: int | None = None

    def _dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        If continuous: return x_dot.
        If discrete: return x_next.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def update_state(self, x_k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        x_k = np.asarray(x_k, dtype=float).reshape(-1)
        u_k = np.asarray(u_k, dtype=float).reshape(-1)

        if self.x_dim is not None:
            assert x_k.shape[0] == self.x_dim, f"x has dim {x_k.shape[0]} but expected {self.x_dim}"
        if self.u_dim is not None:
            assert u_k.shape[0] == self.u_dim, f"u has dim {u_k.shape[0]} but expected {self.u_dim}"

        if self.discrete_or_continuous == "continuous":
            # Integrate over one step
            sol = solve_ivp(
                fun=lambda t, x: self._dynamics(t, x, u_k),
                t_span=(0.0, self.dt),
                y0=x_k,
                method="RK45",
                rtol=1e-7,
                atol=1e-9,
            )
            return sol.y[:, -1]

        if self.discrete_or_continuous == "discrete":
            # Discrete step (ignore t)
            return np.asarray(self._dynamics(0.0, x_k, u_k), dtype=float).reshape(-1)

        raise ValueError(f"Unknown discrete_or_continuous={self.discrete_or_continuous}")


