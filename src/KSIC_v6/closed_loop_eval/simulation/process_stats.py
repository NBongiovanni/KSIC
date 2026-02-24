from typing import Optional, Sequence

import numpy as np

from .results import SimResults
from .simulation_statistics import SimulationStatistics

def process_statistics(
        simulation_results: SimResults,
        sqp_iters: Optional[Sequence[float]],
        *,
        div_threshold: float = 1.2,
) -> SimulationStatistics:

    t = simulation_results.time # (T,)
    x_data = simulation_results.x_data # (T, n)
    z_data = simulation_results.z_data

    mse_z = np.mean(z_data.error) if z_data is not None else 0.0
    if sqp_iters and len(sqp_iters) > 0:
        sqp_iter = float(np.nanmean(sqp_iters))
    else:
        sqp_iter = 0.0
    mse_x_pos = 0.0
    terminal_error = 0.0
    has_diverged = False
    overshoot_1 = overshoot_2 = 0.0
    settling_time_1 = settling_time_2 = np.nan

    if x_data is not None:
        x_traj = x_data.traj  # (T, n)
        x_ref_traj = x_data.ref_traj  # (T, 2)
        x_traj_error = x_data.error

        mse_x_pos = np.mean(x_traj_error[:, :2])
        terminal_error = np.linalg.norm(x_traj[-1, :2] - x_ref_traj[-1, :2])
        max_deviation = np.max(np.abs(x_traj[:, :2]))
        has_diverged = max_deviation > div_threshold # divergence detection

        overshoot_1 = _overshoot_pct_1d(x_traj[:, 0], x_ref_traj[:, 0])
        overshoot_2 = _overshoot_pct_1d(x_traj[:, 1], x_ref_traj[:, 1])
        settling_time_1 = _settling_time_5pct_1d(t, x_traj[:, 0], x_ref_traj[:, 0])
        settling_time_2 = _settling_time_5pct_1d(t, x_traj[:, 1], x_ref_traj[:, 1])

    return SimulationStatistics(
        error_x_pos=mse_x_pos,
        error_z=mse_z,
        terminal_error=terminal_error,
        sqp_iter=sqp_iter,
        has_diverged=has_diverged,
        overshoot_1=overshoot_1,
        overshoot_2=overshoot_2,
        settling_time_1=settling_time_1,
        settling_time_2=settling_time_2,
    )


def _overshoot_pct_1d(y: np.ndarray, y_ref: np.ndarray, eps_abs: float = 1e-6) -> float:
    """
    Overshoot (%) pour un échelon 1D: max excès au-delà de la valeur finale,
    normalisé par l'amplitude d'échelon.
    Retourne 0 si amplitude ~ 0 ou si pas d'excès.
    """
    y0 = y[0]
    yss = y_ref[-1]
    step = max(abs(yss - y0), eps_abs)
    sign = np.sign(yss - y0) or 1.0

    # excursion au-delà de la valeur finale, du bon côté
    exc = (y - yss) * sign
    exc_max = np.max(exc)
    return max(0.0, 100.0 * exc_max / step)


def _settling_time_5pct_1d(
        t: np.ndarray,
        y: np.ndarray,
        y_ref: np.ndarray,
        eps_abs: float = 1e-6,
        tol_rel: float = 0.05,
        window_ratio: float = 0.1,
) -> float:
    """
    Temps de réponse à 5% (settling time) pour un signal 1D.

    Paramètres
    ----------
    t : (T,) ndarray
        Temps croissant (en s).
    y : (T,) ndarray
        Signal simulé.
    y_ref : (T,) ndarray
        Référence (par ex. consigne ou valeur finale).
    eps_abs : float
        Seuil minimal pour la normalisation afin d'éviter les divisions par zéro.
    tol_rel : float
        Tolérance relative (par défaut 5%).
    window_ratio : float
        Fraction de la fin de la trajectoire utilisée pour estimer la valeur finale.

    Retourne
    --------
    ts : float
        Temps de réponse (le plus petit t où |y - y_ss|/|y_ss| < tol_rel
        et la condition reste vraie jusqu’à la fin).
        np.nan si non atteint.
    """
    T = len(t)
    if T < 2:
        return np.nan

    # Valeur finale estimée (moyenne des 10% derniers points de la ref)
    k = max(5, int(window_ratio * T))
    y_ss = np.mean(y_ref[-k:])
    y0 = y[0]

    # Erreur relative
    denom = max(abs(y_ss), eps_abs)
    err_rel = np.abs(y - y_ss) / denom

    # Indices où erreur > tolérance
    idx_out = np.where(err_rel > tol_rel)[0]
    if idx_out.size == 0:
        # jamais sorti au-delà de la tolérance (déjà à l’équilibre)
        return t[0]

    # Dernier instant où on est en dehors de la tolérance
    last_out = idx_out[-1]

    # Vérifie qu'il reste au moins un échantillon après
    if last_out + 1 < T:
        return float(t[last_out + 1])
    else:
        return np.nan