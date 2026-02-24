from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .ref_traj_builder_base import ReferenceTrajBuilderBase


class ReferenceTrajBuilderSensor(ReferenceTrajBuilderBase):
    """
    Builds z_ref directly from sensor signals (no images).
    """
    def __init__(self, *args, x_scaler: StandardScaler, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_scaler = x_scaler

    def build(self):
        print("Generating reference trajectories (sensor)...")
        state_ref_traj = self.build_state_ref()

        # IMPORTANT:
        # Here you must define how to obtain sensor measurements from state_ref_traj.
        # If your "state_ref_traj" is already in sensor space, then x_ref_traj = state_ref_traj.
        x_ref_traj = state_ref_traj  # <-- adjust if needed

        # scale -> torch -> project
        x_ref_traj = np.asarray(x_ref_traj)
        x_scaled = self.x_scaler.transform(x_ref_traj).astype(np.float32)  # (T, x_dim)

        device = next(self.koop_model.parameters()).device
        x_t = torch.as_tensor(x_scaled, dtype=torch.float32, device=device)  # (T, x_dim)

        with torch.no_grad():
            z_t = self.koop_model.project(x_t)  # expected (T, z_dim)

        z_ref_traj = z_t.detach().cpu().numpy()
        im_ref_traj = None  # no images in sensor

        return state_ref_traj, im_ref_traj, z_ref_traj
