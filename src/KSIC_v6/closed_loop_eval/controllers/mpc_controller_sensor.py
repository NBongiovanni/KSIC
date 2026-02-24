import numpy as np
import torch

from .mpc_controller_base import MPCControllerBase


class SensorMPCController(MPCControllerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.x_scaler is None:
            raise ValueError("SensorMPCController requires x_scaler (StandardScaler).")

    def encode_to_z(self, x_k: np.ndarray, x_km1: np.ndarray) -> np.ndarray:
        # x_km1 unused in sensor modality
        x_k = np.asarray(x_k).reshape(-1)

        # scaler -> float32
        x_scaled = self.x_scaler.transform(x_k[None, :])[0].astype(np.float32)

        x_t = torch.as_tensor(x_scaled, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, x_dim)

        z_t = self.koop_model.project(x_t)  # expected (1, z_dim)
        z = z_t.detach().cpu().numpy().squeeze().reshape(-1)

        return z
