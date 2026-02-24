# KSIC_v6/model_learning/trainer/vision_losses.py
from typing import Dict, Tuple, TypedDict
import numpy as np
import torch
from torch import nn, Tensor
from KSIC_v6.models import ForwardOutputs, GroundTruth

CENTROIDS_TO_METERS = 0.015
RAD_TO_DEG = 180 / np.pi

class VisionSubLosses(TypedDict, total=False):
    y_pred: float
    y_rec: float
    z: float
    c: float
    angle: float
    horizontal: float
    vertical: float
    iou: float

VisionFullLoss = tuple[float, VisionSubLosses]


class VisionLossComputer:
    def __init__(self, base: dict):
        self.base = base
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    @torch.no_grad()
    def _iou_loss(self, im_pred: Tensor, target: Tensor) -> Tensor:
        return 1 - compute_iou(im_pred, target)

    def _check_num_views(self, num_views: int) -> None:
        if num_views not in (1, 2):
            raise ValueError(f"Invalid number of views {num_views}")

    @staticmethod
    def _zero_like(x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def _maybe_right(
            self,
            left_like: Tensor,
            right_value: Tensor,
            num_views: int
    ) -> Tensor:
        return right_value if num_views == 2 else self._zero_like(left_like)

    def _get_targets(self, gt: GroundTruth, num_views: int) -> dict:
        """
        Returns a dict of target tensors with explicit names.
        Shapes:
          - gtL0: (B,1,H,W), gtL: (B,T-1,1,H,W)
          - gtR0/gtR present only if num_views==2
        """
        gtL0 = gt.y_left[:, 0]
        gtL = gt.y_left[:, 1:]
        if num_views == 2:
            gtR0 = gt.y_right[:, 0]
            gtR = gt.y_right[:, 1:]
        else:
            gtR0 = None
            gtR = None
        return {"gtL0": gtL0, "gtL": gtL, "gtR0": gtR0, "gtR": gtR}

    @torch.no_grad()
    def _metrics(
            self,
            outputs: ForwardOutputs,
            t: dict,
            gt: GroundTruth,
            num_views: int
    ) -> tuple[Tensor, Tensor, Tensor]:

        iouL = self._iou_loss(outputs.pred.y_left, t["gtL"])
        if num_views == 2:
            iouR = self._iou_loss(outputs.pred.y_right, t["gtR"])
            iou = 0.5 * (iouL + iouR)
        else:
            iou = iouL

        # Horizontal / vertical centroid errors (MSE; scaled later)
        hl = self.mse(outputs.pred.centroids_left[:, :, 0, 0], gt.centroids_left[:, 1:, 0, 0])
        vl = self.mse(outputs.pred.centroids_left[:, :, 0, 1], gt.centroids_left[:, 1:, 0, 1])

        hr = self.mse(outputs.pred.centroids_right[:, :, 0, 0], gt.centroids_right[:, 1:, 0, 0])
        vr = self.mse(outputs.pred.centroids_right[:, :, 0, 1], gt.centroids_right[:, 1:, 0, 1])

        hr = self._maybe_right(hl, hr, num_views)
        vr = self._maybe_right(vl, vr, num_views)

        horizontal = hl + hr
        vertical = vl + vr
        return iou, horizontal, vertical

    def compute(
            self,
            outputs: ForwardOutputs,
            z_proj: Tensor,
            ground_truth: GroundTruth,
            phases_active: list[bool],
            effective_weight,
            num_views: int = 2,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        self._check_num_views(num_views)
        t = self._get_targets(ground_truth, num_views)

        # --- losses with grad ---
        im_rec_loss = self._image_rec_loss(outputs, t, num_views)
        im_pred_loss = self._image_pred_loss(outputs, t, num_views)
        z_loss = self._latent_loss(outputs, z_proj)
        centroid_loss, angle_loss = self._geom_losses(outputs, ground_truth)

        # --- metrics (no grad) ---
        metrics = self._metrics(outputs, t, ground_truth, num_views)
        loss_iou, horizontal_loss, vertical_loss = metrics

        # --- weights ---
        if len(phases_active) > 1 and phases_active[1]:
            w_y_rec = 0.6 * self.base["y_rec"]
            w_z = 2.0 * self.base["z"]
        else:
            w_y_rec = self.base["y_rec"]
            w_z = self.base["z"]

        if len(phases_active) > 2 and phases_active[2]:
            w_c = 0.0
            w_a = 0.0
        else:
            w_c = effective_weight(self.base["c"], "c")
            w_a = effective_weight(self.base["a"], "a")

        weights = {
            "y_pred": self.base["y_pred"],
            "y_rec": w_y_rec,
            "z": w_z,
            "c": w_c,
            "a": w_a,
        }

        full = (
            weights["y_pred"] * im_pred_loss
            + weights["y_rec"] * im_rec_loss
            + weights["z"] * z_loss
            + weights["c"] * centroid_loss
            + weights["a"] * angle_loss
        )

        sub = {
            "y_pred": float(np.sqrt(float(im_pred_loss.detach().cpu().item()))),
            "y_rec": float(np.sqrt(float(im_rec_loss.detach().cpu().item()))),
            "z": float(np.sqrt(float(z_loss.detach().cpu().item()))),
            "c": float(np.sqrt(float(centroid_loss.detach().cpu().item()))),
            "angle": float(np.sqrt(float(angle_loss.detach().cpu().item())) * RAD_TO_DEG),
            "iou": float(loss_iou.detach().cpu().item()),
            "horizontal": float(np.sqrt(float(horizontal_loss.detach().cpu().item())) * CENTROIDS_TO_METERS),
            "vertical": float(np.sqrt(float(vertical_loss.detach().cpu().item())) * CENTROIDS_TO_METERS),
        }
        return full, sub

    def _image_rec_loss(
            self, outputs: ForwardOutputs, t: dict, num_views: int
    ) -> Tensor:
        recL = self.bce(outputs.rec.y_logits_left, t["gtL0"])
        if num_views == 2:
            recR = self.bce(outputs.rec.y_logits_right, t["gtR0"])
        else:
            recR = self._zero_like(recL)
        return recL + recR

    def _image_pred_loss(
            self, outputs: ForwardOutputs, t: dict, num_views: int
    ) -> Tensor:
        predL = self.bce(outputs.pred.y_logits_left, t["gtL"])
        if num_views == 2:
            predR = self.bce(outputs.pred.y_logits_right, t["gtR"])
        else:
            predR = self._zero_like(predL)
        return predL + predR

    def _latent_loss(self, outputs: ForwardOutputs, z_proj: Tensor) -> Tensor:
        return self.mse(outputs.pred.z, z_proj[:, 1:])

    def _geom_losses(
            self, outputs: ForwardOutputs, gt: GroundTruth
    ) -> tuple[Tensor, Tensor]:
        # centroid_loss, angle_loss
        cL = self.mse(outputs.pred.centroids_left, gt.centroids_left[:, 1:])
        aL = self.mse(outputs.pred.angles_left, gt.angles_left[:, 1:])
        cR = self.mse(outputs.pred.centroids_right, gt.centroids_right[:, 1:])
        aR = self.mse(outputs.pred.angles_right, gt.angles_right[:, 1:])
        return (cL + cR), (aL + aR)


def _mean_sub_losses_vision(items: list[VisionSubLosses]) -> VisionSubLosses:
    keys = ("y_rec", "y_pred", "z", "c", "angle", "iou", "horizontal", "vertical")
    out = {}
    for k in keys:
        out[k] = float(np.mean([d[k] for d in items], dtype=np.float64))
    return out


def compute_iou(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    # Predictions and targets must have the same shape:
    assert pred.shape == target.shape
    # Expected shape: (B, N, 1, H, W):

    pred_bin = (pred >= 0.4)
    target_bin = (target >= 0.4)

    # (B, N)
    intersection = (pred_bin & target_bin).float().sum(dim=(2, 3, 4))
    union = (pred_bin | target_bin).float().sum(dim=(2, 3, 4))
    iou = intersection / (union + eps)  # (B, N)
    return iou.mean()
