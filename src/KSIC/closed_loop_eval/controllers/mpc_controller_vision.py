import numpy as np
import torch

from .mpc_controller_base import MPCControllerBase
from ..state_renderer import StateRenderer

class VisionMPCController(MPCControllerBase):
    def __init__(self, *args, render_resolution: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_renderer = StateRenderer(render_resolution)
        self.im_ref_traj = None

    def encode_to_z(self, x_k: np.ndarray, x_km1: np.ndarray) -> np.ndarray:
        im_k = self.state_renderer.pipeline(x_k).astype(np.float32) / 255.0
        im_km1 = self.state_renderer.pipeline(x_km1).astype(np.float32) / 255.0

        im_k = torch.from_numpy(im_k)       # [1, H, W] presumably
        im_km1 = torch.from_numpy(im_km1)

        y = torch.stack((im_km1, im_k), dim=0).unsqueeze(0)  # [1, 2, H, W]
        y = y.to(self.device)

        z_t = self.koop_model.project(y)  # (1, z_dim)
        z = z_t.detach().cpu().numpy().squeeze().reshape(-1)
        return z
