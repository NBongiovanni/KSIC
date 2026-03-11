from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from KSIC.models import BaseKoopModel
from .solver_backend import SolverBackend
from .mpc_problem import MPCProblem

from .casadi_dynamics import build_latent_dynamics_function

class MPCControllerBase(ABC):
    """
    Base MPC controller: contains all MPC / backend logic.
    Subclasses implement only `encode_to_z(...)`.
    """
    def __init__(
        self,
        model_params: dict,
        control_params: dict,
        solver_backend: SolverBackend,
        koop_model: BaseKoopModel,
        u_scaler: StandardScaler,
        x_scaler: Optional[StandardScaler] = None,  # used only for the sensor modality
    ):
        super().__init__()
        self.model_params = model_params
        self.control_params = control_params
        self.backend = solver_backend
        self.koop_model = koop_model

        self.drone_dim = model_params["drone"]["dim"]
        self.u_dim = self.koop_model.u_dim
        self.device = next(self.koop_model.parameters()).device
        self.z_dim = int(self.model_params["z_dynamics"]["z_dim"])

        self.u_scaler = u_scaler
        self.x_scaler = x_scaler
        self.control_runs_dir = self.control_params["control_runs_dir"]

        self.state_ref_traj = None
        self.z_ref_traj = None
        self.im_ref_traj = None

        # Logs
        self.z_traj: list[np.ndarray] = []
        self.u_scaled_traj: list[np.ndarray] = []
        self.u_physical_traj: list[np.ndarray] = []

        self.problem: Optional[MPCProblem] = None

    def build(self) -> None:
        Q, Qf, R = self._set_cost_matrices()
        constraints = self.control_params["constraints"]
        Fmin, Fmax = constraints["force_limits"]
        txmin, txmax = constraints["torque_limits"]
        tymin, tymax = constraints["torque_limits"]
        tzmin, tzmax = constraints["torque_limits"]

        if self.u_dim == 2:
            u_min = np.array([Fmin, txmin], dtype=float)
            u_max = np.array([Fmax, txmax], dtype=float)
        elif self.u_dim == 4:
            u_min = np.array([Fmin, txmin, tymin, tzmin], dtype=float)
            u_max = np.array([Fmax, txmax, tymax, tzmax], dtype=float)
        else:
            raise ValueError(f"Unsupported u_dim={self.u_dim}")

        u_min_s = self.u_scaler.transform(u_min.reshape(1, -1))[0]
        u_max_s = self.u_scaler.transform(u_max.reshape(1, -1))[0]

        # Initial guess in scaled space
        # TODO: refaire cette partie plus proprement et dynamiquement
        if self.u_dim == 2:
            u_guess = self.u_scaler.transform(np.array([[9.81, 0.0]], dtype=float))[0]
        elif self.u_dim == 4:
            u_guess = self.u_scaler.transform(np.array([[0.26487, 0.0, 0.0, 0.0]], dtype=float))[0]
        else:
            raise ValueError(f"Unsupported u_dim={self.u_dim}")

        f_discrete = build_latent_dynamics_function(
            z_dynamics_model=self.model_params["z_dynamics"]["model"],
            z_dim=self.z_dim,
            u_dim=self.u_dim,
            koop_model=self.koop_model,
            augment_actuated=self.model_params["z_dynamics"]["affine_term"],
        )

        self.problem = MPCProblem(
            dt=self.control_params["dt"],
            N=self.control_params["num_steps_horizon"],
            z_dim=self.z_dim,
            u_dim=self.u_dim,
            Q=Q,
            Qf=Qf,
            R=R,
            S=np.diag(self.control_params["cost"]["S"]),
            use_inputs_constraints=constraints["use_inputs_constraints"],
            u_min=u_min_s,
            u_max=u_max_s,
            f_discrete=f_discrete,
            tvp_provider=self._make_tvp_provider(),
            u_guess=u_guess,
        )
        self.backend.build(self.problem)

    def reset(self) -> None:
        self.z_traj.clear()
        self.u_physical_traj.clear()
        self.u_scaled_traj.clear()
        self.state_ref_traj = None
        self.z_ref_traj = None
        self.backend.reset()

    def set_reference(
            self,
            state_ref_traj: np.ndarray,
            z_ref_traj: np.ndarray,
            im_ref_traj,
    ) -> None:
        z_ref = np.asarray(z_ref_traj, dtype=float)
        if z_ref.ndim == 1:
            z_ref = z_ref[None, :]  # (1, z_dim)

        self.state_ref_traj = state_ref_traj
        self.z_ref_traj = z_ref
        self.im_ref_traj = im_ref_traj

        if hasattr(self.backend, "set_reference"):
            self.backend.set_reference(z_ref)

    def set_initial_conditions(self, x_init: np.ndarray) -> None:
        assert self.problem is not None, "Call build() before set_initial_conditions()."

        with torch.no_grad():
            z_init = self.encode_to_z(x_init, x_init)
            self.z_traj.append(z_init)

            # backend expects (z0, u0_guess) in do-mpc style
            self.backend.set_initial_condition(z_init, self.problem.u_guess)

    def compute_control(self, x_k: np.ndarray, x_km1: np.ndarray) -> np.ndarray:
        """
        Real control from measurements (sensor) or states (vision->render).
        """
        assert self.problem is not None, "Call build() before compute_control()."

        with torch.no_grad():
            z_k = self.encode_to_z(x_k, x_km1)
            self.z_traj.append(z_k)

            path = Path(self.control_runs_dir) / "solver_stdout.txt"
            with open(path, "a") as f, contextlib.redirect_stdout(f):
                u_k_scaled = self.backend.make_step(z_k)

        return self._process_control(u_k_scaled)

    def _process_control(self, u_k_scaled: np.ndarray) -> np.ndarray:
        u_k_scaled = np.asarray(u_k_scaled).reshape(-1)
        self.u_scaled_traj.append(u_k_scaled)

        u_k_physical = self.u_scaler.inverse_transform(u_k_scaled.reshape(1, -1))[0]
        u_k_physical = np.asarray(u_k_physical).reshape(-1)

        self.u_physical_traj.append(u_k_physical)
        return u_k_physical

    def _set_cost_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        half = int(self.z_dim / 2)
        q_pos = float(self.control_params["cost"]["Q_positions"])
        q_vel = float(self.control_params["cost"]["Q_velocities"])
        Q = np.zeros((self.z_dim, self.z_dim))
        Q[:half, :half] = q_pos * np.eye(half)
        Q[half:, half:] = q_vel * np.eye(half)

        p_pos = float(self.control_params["cost"]["P_positions"])
        p_vel = float(self.control_params["cost"]["P_velocities"])
        P = np.zeros((self.z_dim, self.z_dim))
        P[:half, :half] = p_pos * np.eye(half)
        P[half:, half:] = p_vel * np.eye(half)


        R = np.diag(self.control_params["cost"]["R"])
        return Q, P, R

    def _make_tvp_provider(self):
        dt = float(self.control_params["dt"])
        N = int(self.control_params["num_steps_horizon"])

        def provider(t_now, template):
            assert self.z_ref_traj is not None, "Call set_reference() before running MPC."
            z_ref = self.z_ref_traj
            k0 = int(float(t_now) / dt)

            for k in range(N + 1):
                idx = min(k0 + k, z_ref.shape[0] - 1)
                template["_tvp", k, "z_ref"] = z_ref[idx]
            return template
        return provider

    # ----------------- subclass hook -----------------
    @abstractmethod
    def encode_to_z(self, x_k: np.ndarray, x_km1: np.ndarray) -> np.ndarray:
        """
        Return z_k as shape (z_dim,).
        """
        raise NotImplementedError
