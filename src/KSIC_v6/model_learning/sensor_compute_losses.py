from typing import Dict, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn

from KSIC_v6.models.outputs.sensor_outputs import SensorValForwardOutputs

class SensorsSubLosses(TypedDict):
    rec: float
    pred_z: float
    pred_x: float
    pred_position: float

SensorFullLoss = tuple[float, SensorsSubLosses]

class SensorLossComputer:
    def __init__(self, base: dict):
        self.base = base
        self.mse = nn.MSELoss()

    def compute(
            self,
            models_outputs: SensorValForwardOutputs,
            device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        alphas = self.base

        x_rec = models_outputs.rec
        z_proj = models_outputs.proj
        z_pred = models_outputs.pred.z
        x_pred = models_outputs.pred.state
        x_gt = models_outputs.state_gt_scaled

        loss_fn = nn.MSELoss()

        loss_x_rec = loss_fn(x_rec, x_gt[:, 0].to(device))
        loss_pred_x = loss_fn(x_pred[:, 1:], x_gt[:, 1:].to(device))
        loss_position = loss_fn(x_pred[:, 1:, :3], x_gt[:, 1:, :3].to(device))
        loss_z_pred = loss_fn(z_pred[:, 1:], z_proj[:, 1:])

        full_loss = (
                alphas["rec"] * loss_x_rec +
                alphas["pred_z"] * loss_z_pred +
                alphas["pred_x"] * loss_pred_x
        )

        named_losses = {
            "pred_x": float(loss_pred_x.detach().cpu().item()),
            "pred_z": float(loss_z_pred.detach().cpu().item()),
            "rec": float(loss_x_rec.detach().cpu().item()),
            "pred_position": float(loss_position.detach().cpu().item()),
        }
        return full_loss, named_losses


def _mean_sub_losses(items):
    keys = ("rec", "pred_x", "pred_z", "pred_position")
    out = {}
    for k in keys:
        out[k] = float(np.mean([d[k] for d in items], dtype=np.float64))
    return out