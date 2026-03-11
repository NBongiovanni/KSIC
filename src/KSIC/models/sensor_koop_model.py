import torch
from torch import Tensor
import numpy as np

from .base_koop_model import BaseKoopModel
from KSIC.models.outputs.sensor_outputs import Pred
from KSIC.models.nn.mlp_blocks import build_mlp

class SensorKoopModel(BaseKoopModel):
    def __init__(self, model_params: dict):
        super().__init__(model_params)
        self.x_dim = model_params["z_dynamics"]["x_dim"]
        self.u_dim = model_params["z_dynamics"]["u_dim"]
        self.z_dim = model_params["z_dynamics"]["z_dim"]
        self.activation = model_params["activation"]
        self.parameters_to_prune = self.get_prunable_params()

        self.encoder = build_mlp(
            self.x_dim,
            model_params["dim_hidden_layers"],
            self.z_dim,
            model_params["num_hidden_layers"],
            self.activation,
        )
        self.decoder = build_mlp(
            self.z_dim,
            model_params["dim_hidden_layers"],
            self.x_dim,
            model_params["num_hidden_layers"],
            self.activation,
        )
    # -------- AE I/O (vision) ----------
    def project(self, y: Tensor) -> Tensor:
        return self.encoder(y)

    def reconstruct(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def batch_projection(self, x_gt: Tensor) -> Tensor:
        b_size = x_gt.size(0)
        n_steps = x_gt.size(1)

        x_gt_flat = torch.reshape(x_gt, (b_size * n_steps, -1))
        z_proj_flat = self.project(x_gt_flat)
        return torch.reshape(z_proj_flat, (b_size, n_steps, self.z_dim))

    def forward(
            self,
            x_init: Tensor,
            u_traj: Tensor,
            num_steps: int
    ) -> tuple[Tensor, Pred]:

        device = next(self.parameters()).device
        batch_size = u_traj.shape[0]
        dtype = x_init.dtype

        x_pred = torch.zeros(
            (batch_size, num_steps, self.x_dim),
            device=device,
            dtype=dtype
        )
        z_pred = torch.zeros(
            (batch_size, num_steps, self.z_dim),
            device=device,
            dtype=dtype
        )

        z_init_k = self.project(x_init.float())
        x_rec_k = self.reconstruct(z_init_k)

        z_pred[:, 0] = z_init_k
        x_pred[:, 0] = x_rec_k
        x_rec = x_rec_k

        z_pred_k = z_init_k
        for i in range(1, num_steps):
            z_pred_kp1 = self.z_dynamics_step(z_pred_k,u_traj[:, i - 1].to(device))
            z_pred[:, i] = z_pred_kp1
            w_pred_kp1_ = self.reconstruct(z_pred_kp1)
            x_pred[:, i] = w_pred_kp1_
            z_pred_k = z_pred_kp1
        return x_rec, Pred(state=x_pred, z=z_pred)