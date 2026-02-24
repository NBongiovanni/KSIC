from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from KSIC_v6.viz.style import GT_COLOR

def plot_x(
        axes: Sequence[Axes],
        time: np.ndarray,
        labels: list[str],
        x: np.ndarray,
        x_gt: np.ndarray,
        show_legend: bool = True,
        label_x: str = "Predicted trajectory",
        label_gt: str = "True trajectory"
) -> None:
    """
    Trace l'état réel et prédit.
    Convention :
      - len(axes) == nombre de dimensions à tracer
      - x, x_gt ont la forme (T, n_dim) avec n_dim == len(axes)
    """
    n_dim = len(axes)

    assert x.shape[1] == n_dim
    assert x_gt.shape[1] == n_dim
    assert len(labels) == n_dim

    for i, ax in enumerate(axes):
        ax.plot(
            time,
            x_gt[:, i],
            color=GT_COLOR,
            linestyle = "--",
            label=label_gt
        )
        ax.plot(time, x[:, i], label=label_x)
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("Time [s]")
    if show_legend:
        axes[0].legend()

def plot_x_gt(
        axes: Sequence[Axes],
        time: np.ndarray,
        labels: list[str],
        x_gt: np.ndarray,
        show_legend: bool = True,
        label_gt: str = "True trajectory"
) -> None:
    """
    Trace l'état réel et prédit.
    Convention :
      - len(axes) == nombre de dimensions à tracer
      - x, x_gt ont la forme (T, n_dim) avec n_dim == len(axes)
    """
    n_dim = len(axes)

    assert x_gt.shape[1] == n_dim
    assert len(labels) == n_dim

    for i, ax in enumerate(axes):
        ax.plot(
            time,
            x_gt[:, i],
            color=GT_COLOR,
            linestyle = "--",
            label=label_gt
        )
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("Time [s]")
    if show_legend:
        axes[0].legend()

def plot_u(
        axes: Sequence[Axes],
        time: np.ndarray,
        u_traj: np.ndarray,
        labels: list
) -> None:
    u_dim = u_traj.shape[1]
    assert len(labels) == u_traj.shape[1], f"labels ({len(labels)}) != u_dim ({u_traj.shape[1]})"

    for dim in range(u_dim):
        axes[dim].plot(time, u_traj[:, dim])
        axes[dim].set_ylabel(labels[dim])
    axes[-1].set_xlabel("Time [s]")


def plot_z(
        axes: Sequence[Axes],
        time: np.ndarray,
        z_proj: np.ndarray,
        z_pred: np.ndarray,
) -> None:
    """
    Trace la variable latente réelle et projetée.
    """
    for dim in range(z_pred.shape[1]):
        axes[dim].plot(time, z_proj[:, dim], label="z_proj")
        axes[dim].plot(time, z_pred[:, dim], label="z_pred")
        axes[dim].set_ylabel(rf"$z_{{{dim+1}}}$")
    axes[-1].set_xlabel("Time [s]")


def plot_and_save_eigs_compare(
        eigs_a,
        eigs_b,
        label_a="Linear",
        label_b="Bilinear",
        title="Eigenvalues comparison",
        savepath=None,
        dpi=300,
        show_unit_circle=True,
        grid=True,
        equal_aspect=True,
):
    """
    Plot and (optionally) save a comparison of two eigenvalue sets in the complex plane.

    Parameters
    ----------
    eigs_a, eigs_b : array_like of complex, shape (n,)
        Eigenvalues to compare.
    label_a, label_b : str
        Legend labels.
    title : str
        Figure title.
    savepath : str or Path or None
        If provided, saves the figure to this path (e.g., .pdf or .png).
    dpi : int
        Save resolution (for raster formats like PNG).
    show_unit_circle : bool
        Draw unit circle if True.
    grid : bool
        Show grid if True.
    equal_aspect : bool
        Enforce equal axis aspect ratio if True.

    Returns
    -------
    fig, ax
    """
    eigs_a = np.asarray(eigs_a).astype(np.complex128).ravel()
    eigs_b = np.asarray(eigs_b).astype(np.complex128).ravel()

    fig, ax = plt.subplots()

    # Linear: croix (couleur explicitée)
    ax.scatter(
        eigs_a.real,
        eigs_a.imag,
        marker="x",
        s=45,
        linewidths=1.5,
        c="C0",
        label=label_a,
        zorder=3,
    )

    # Bilinear: cercles vides (CONTOUR forcé)
    ax.scatter(
        eigs_b.real,
        eigs_b.imag,
        marker="o",
        s=55,
        facecolors="none",
        edgecolors="C1",
        linewidths=1.5,
        label=label_b,
        zorder=4,
    )

    # Unit circle
    if show_unit_circle:
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=1)

    # Axes through origin
    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)

    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title(title)
    ax.legend()

    if grid:
        ax.grid(True)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    # Common limits based on both sets + unit circle
    r = np.max(np.r_[1.0, np.abs(eigs_a), np.abs(eigs_b)])
    pad = 0.1 * r
    lim = r + pad
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    fig.tight_layout()

    # Save (if requested)
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, ax

