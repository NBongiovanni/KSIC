import numpy as np
from KSIC.models import ForwardOutputs, GroundTruth

CENTROIDS_TO_METERS = 0.015
RAD_TO_DEG = 180 / np.pi

def compute_loss_nrmse_fit(outputs: ForwardOutputs, ground_truth: GroundTruth) -> dict:
    """This class computes the NRMSE Fit (%) loss for the angle and centroids.
    It is used to evaluate the performance of the model on the ground truth trajectories.
    """
    angles_pred = outputs.pred.angles_left * RAD_TO_DEG
    angles_gt = ground_truth.angles_left[:, 1:] * RAD_TO_DEG

    angle_fit = compute_nrmse_fit_batch(
        angles_pred.detach().cpu().numpy(),
        angles_gt.detach().cpu().numpy()
    )

    centroids_pred = outputs.pred.centroids_left * CENTROIDS_TO_METERS
    centroids_gt = ground_truth.centroids_left[:, 1:] * CENTROIDS_TO_METERS

    horizontal_fit = compute_nrmse_fit_batch(
        centroids_pred[:, :, 0, 0:1].detach().cpu().numpy(),
        centroids_gt[:, :, 0, 0:1].detach().cpu().numpy()
    )
    vertical_fit = compute_nrmse_fit_batch(
        centroids_pred[:, :, 0, 1:2].detach().cpu().numpy(),
        centroids_gt[:, :, 0, 1:2].detach().cpu().numpy()
    )
    return {"angle": angle_fit, "horizontal": horizontal_fit, "vertical": vertical_fit}


def compute_nrmse_fit_batch(
    pred: np.ndarray,
    true: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    # Batch version of NRMSE Fit (%).
    #
    # Args:
    #     pred: (B, T, D)
    #     true: (B, T, D)
    #     eps
    #
    # Returns:
    #     fit: (B, D)  Fit percentage per trajectory and per dimension
    # """
    # pred = np.asarray(pred)
    # true = np.asarray(true)
    #
    # assert pred.shape == true.shape
    # assert pred.ndim == 3, "Expected (B, T, D)"
    #
    # # Error
    # err = pred - true                              # (B, T, D)
    # num = np.linalg.norm(err, axis=1)              # (B, D)
    #
    # # Centered ground truth
    # true_centered = true - np.mean(true, axis=1, keepdims=True)
    # denom = np.linalg.norm(true_centered, axis=1)  # (B, D)
    #
    # # Avoid division by zero
    # nrmse = num / (denom + eps)
    # fit = (1.0 - nrmse) * 100.0
    #
    # # For constant signals → fit = 0
    # fit[denom < eps] = 0.0
    # return fit.mean(axis=0)
    return 0
