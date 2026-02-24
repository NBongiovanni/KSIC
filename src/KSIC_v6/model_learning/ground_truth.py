from __future__ import annotations

import torch
from torch import Tensor

from KSIC_v6.data_pipeline import compute_angles, compute_centroids_gt
from KSIC_v6.models import GroundTruth


def get_channel_slices(num_views: int):
    # ordre : [v0_t, v1_t, ..., v0_tp1, v1_tp1, ...]
    slices = {}
    for v in range(num_views):
        slices[f"view{v}_k"]   = slice(v, v + 1)
        slices[f"view{v}_kp1"] = slice(v + num_views, v + num_views + 1)
    return slices


def build_ground_truth_from_images(
        y_data: Tensor,
        x_data: Tensor,
        drone_dim: int
) -> GroundTruth:

    num_views = 2 if drone_dim == 3 else 1
    idx = get_channel_slices(num_views)
    left_kp1 = idx["view0_kp1"]

    y_left_kp1 = y_data[:, :, left_kp1]
    centroids_gt_left_kp1 = compute_centroids_gt(y_data[:, :, left_kp1])
    angles_gt_left_kp1 = compute_angles(y_data[:, :, left_kp1])
    # centroids_gt_left_kp1  = ema_filter_time(centroids_gt_left_kp1,  alpha=0.9, time_dim=1)

    if drone_dim == 3:
        right_kp1 = idx["view1_kp1"]
        y_right_kp1 = y_data[:, :, right_kp1]
        centroids_gt_right_kp1 = compute_centroids_gt(y_data[:, :, right_kp1])
        angles_gt_right_kp1 = compute_angles(y_data[:, :, right_kp1])
         # centroids_gt_right_kp1 = ema_filter_time(centroids_gt_right_kp1, alpha=0.9, time_dim=1)
    elif drone_dim == 2:
        y_right_kp1 = torch.zeros_like(y_left_kp1)
        centroids_gt_right_kp1 = torch.zeros_like(centroids_gt_left_kp1)
        angles_gt_right_kp1 = torch.zeros_like(angles_gt_left_kp1)
    else:
        raise ValueError(f"Invalid number of views {drone_dim}")

    return GroundTruth(
        y_left=y_left_kp1,
        y_right=y_right_kp1,
        centroids_left=centroids_gt_left_kp1,
        angles_left=angles_gt_left_kp1,
        centroids_right=centroids_gt_right_kp1,
        angles_right=angles_gt_right_kp1,
        x_data=x_data
    )


def ema_filter_time(x: Tensor, alpha: float, time_dim: int = 1) -> Tensor:
    """
    Exponential moving average along time dimension.
    y[t] = alpha * y[t-1] + (1-alpha) * x[t]
    alpha in [0,1). Higher alpha => smoother.
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0,1), got {alpha}")

    # Move time dim to 1 for simplicity
    if time_dim != 1:
        x = x.transpose(time_dim, 1)

    y = torch.empty_like(x)
    y[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        y[:, t] = alpha * y[:, t - 1] + (1.0 - alpha) * x[:, t]

    if time_dim != 1:
        y = y.transpose(time_dim, 1)
    return y
