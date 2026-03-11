import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from KSIC.models import VisionKoopModel

class ZTrajGen:
    def __init__(
            self,
            num_steps: int,
            koop_model: VisionKoopModel,
            drone_dim: int,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.koop_model = koop_model
        self.drone_dim = drone_dim
        self.x_ref_traj = self.z_ref_traj = None

    def pipeline(self, im_traj) -> np.ndarray:
        n_steps = self.num_steps
        z_traj = []
        z_k = self.encode(im_traj[0], im_traj[0])
        z_traj.append(z_k)

        for i in tqdm(range(1, n_steps)):
            z_k = self.encode(im_traj[i], im_traj[i-1])
            z_traj.append(z_k)
        return np.stack(z_traj)

    def encode(self, im_k: Tensor, im_km1: Tensor) -> np.ndarray:
        y_k = torch.stack((im_km1, im_k), dim=0).unsqueeze(0)  # [1, 2, H, W]
        z_k = self.koop_model.project(y_k)
        z_k = z_k.detach().cpu().numpy()
        return np.squeeze(z_k)