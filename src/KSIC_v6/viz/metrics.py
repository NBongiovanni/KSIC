import numpy as np


def compute_nrmse_fit(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Compute the normalized RMSE and Fit (%) as used in
    system identification (MATLAB-style definition):

        NRMSE = ||ŷ - y|| / ||y - mean(y)||
        Fit = (1 - NRMSE) * 100

    Args:
        pred: numpy array of shape (N, D)
        true: numpy array of shape (N, D)

    Returns:
        float: Fit percentage
    """
    pred = np.asarray(pred)
    true = np.asarray(true)

    # Flatten along the time dimension (vector form)
    err = pred - true
    num = np.linalg.norm(err)

    # Centering of the true signal
    true_centered = true - np.mean(true, axis=0)
    denom = np.linalg.norm(true_centered)

    if denom == 0:
        # fichier constant : RMSE irrelevant
        return 0.0

    nrmse = num / denom
    fit = (1 - nrmse) * 100
    return fit


def compute_rmse(pred, true):
    err = pred - true
    return np.sqrt(np.mean(err**2))