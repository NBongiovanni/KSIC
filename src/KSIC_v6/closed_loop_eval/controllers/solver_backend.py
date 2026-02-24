from pathlib import Path

from .mpc_problem import MPCProblem
import numpy as np


class SolverBackend:
    def __init__(self):
        self.sqp_iters: list = []

    def build(self, problem: MPCProblem): ...        # crée le modèle/ocp interne

    def set_initial_condition(self, z0, u_guess=None): ...

    def set_tvp_provider(self, provider): ...

    def make_step(self, z_k): ...         # -> u_k

    def define_constraints(self, u_min_scaled, u_max_scaled):
        pass

    def _set_mpc_cost(self, problem: MPCProblem):
        pass

    def _set_up(self) -> None:
        pass

    def define_init_conditions(self, problem: MPCProblem) -> np.ndarray:
        pass

    def _set_inputs_constraints(self) -> None:
        pass

    def scale_and_encode(self, x_k: np.ndarray, x_km1: np.ndarray) -> np.ndarray:
        pass

    def reset(self) -> None:
        pass
