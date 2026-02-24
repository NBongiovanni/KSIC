from __future__ import annotations
from typing import Iterable, Sequence

from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import numpy as np

from KSIC_v6.viz.metrics import compute_nrmse_fit

def plot_state_multi(
        axes: Sequence[Axes],
        runs: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, str]],  # (t, x, name)
        labels: Sequence[str],
        colors: Sequence[str],
        ref_dim: int,
        gt_label: str,
        show_fit: bool = True,
        run_label: str = "{name}",
) -> None:
    runs = list(runs)
    assert len(runs) > 0

    # --- build a single shared GT (xref) ---
    t0 = runs[0][0]
    x_ref0 = runs[0][1][:, :ref_dim]

    # If open-loop provides one x_gt per run, ensure they're consistent
    for (t, x_ref, _, tag) in runs[1:]:
        if t.shape != t0.shape or not np.allclose(t, t0, atol=0.0, rtol=0.0):
            raise ValueError("Time vectors differ across runs; cannot use a single GT.")
        if x_ref.shape[0] != x_ref0.shape[0] or not np.allclose(x_ref[:, :ref_dim], x_ref0, atol=1e-9, rtol=1e-9):
            raise ValueError("GT differs across runs; expected a single shared GT trajectory.")

    x_ref = x_ref0
    x_runs = [(t, x, f"x ({tag})") for (t, _x_ref, x, tag) in runs]

    t0, x0, _ = x_runs[0]
    x_dim = int(x0.shape[1])
    assert len(axes) >= x_dim, f"Need >= {x_dim} axes, got {len(axes)}"
    assert len(labels) >= x_dim, f"Need >= {x_dim} labels, got {len(labels)}"

    # --- reference once ---

    for i in range(ref_dim):
        axes[i].plot(
            t0,
            x_ref[:, i],
            color="black",
            linestyle="--",
            label=(gt_label if i == 0 else None),
        )

    # --- runs ---
    for ridx, (t, x, name) in enumerate(x_runs):
        c = colors[ridx]
        T = x.shape[0]
        for i in range(x_dim):
            if show_fit and i < ref_dim:
                # compute fit for THIS coordinate only
                fit_i = compute_nrmse_fit(
                    pred=x[:T, i:i+1],
                    true=x_ref[:T, i:i+1],
                )
                if i == 0:
                    label = f"{run_label.format(name=name)} — {fit_i:.1f}\%"
                else:
                    label = f"{fit_i:.1f}\%"
            axes[i].plot(t[:T], x[:T, i], color=c, label=label)

    for i in range(x_dim):
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.2)
        axes[i].legend(loc="best")

    axes[x_dim - 1].set_xlabel("Time [s]")


def plot_u_multi_2d(
        axes: Sequence[Axes],
        runs_u: Iterable[tuple[np.ndarray, np.ndarray, str]],
        u_labels: Sequence[str],
        *,
        start_idx: int = 0,
        show_run_legend: bool = False,
) -> int:
    runs_u = list(runs_u)
    assert len(runs_u) > 0, "runs_u is empty"

    t0, u0, _ = runs_u[0]
    u_dim = int(u0.shape[1])
    assert u_dim == 2, f"plot_u_multi_2d expects u_dim=2, got {u_dim}"
    assert len(u_labels) >= 2

    u_axes = axes[start_idx:start_idx + 2]
    assert len(u_axes) == 2, "Need 2 axes for u_dim=2"

    # plot all runs in black
    for (t, u, name) in runs_u:
        assert u.shape[1] == 2, f"Expected u_dim=2, got {u.shape[1]}"
        u_axes[0].plot(t, u[:, 0], color="black", label=(name if show_run_legend else None))
        u_axes[1].plot(t, u[:, 1], color="black", label=(name if show_run_legend else None))

    u_axes[0].set_ylabel(u_labels[0])
    u_axes[1].set_ylabel(u_labels[1])

    for ax in u_axes:
        ax.grid(True, alpha=0.2)
        if show_run_legend:
            handles, labs = ax.get_legend_handles_labels()
            if labs:
                ax.legend(loc="best")

    u_axes[-1].set_xlabel("Time [s]")
    return 2


def plot_u_multi_3d(
        axes: Sequence[Axes],
        runs_u: Iterable[tuple[np.ndarray, np.ndarray, str]],
        u_labels: Sequence[str],
        *,
        start_idx: int = 0,
        grouped_ylabel: str = r"Moments [N.m]",
        group_legend_labels: Sequence[str] = (r"$\tau_1$", r"$\tau_2$", r"$\tau_3$"),
) -> int:
    runs_u = list(runs_u)
    assert len(runs_u) > 0, "runs_u is empty"

    t0, u0, _ = runs_u[0]
    u_dim = int(u0.shape[1])
    assert u_dim == 4, f"plot_u_multi_3d expects u_dim=4, got {u_dim}"
    assert len(u_labels) >= 4

    u_axes = axes[start_idx:start_idx + 2]
    assert len(u_axes) == 2, "Need 2 axes for u_dim=4 (1 + grouped last 3)"

    ax0, axM = u_axes
    styles = ["-", "--", ":"]  # fixed for the 3 moments

    for (t, u, _name) in runs_u:
        assert u.shape[1] == 4, f"Expected u_dim=4, got {u.shape[1]}"

        # first input alone
        ax0.plot(t, u[:, 0], color="black")

        # last 3 grouped (moments)
        for k, j in enumerate([1, 2, 3]):
            axM.plot(t, u[:, j], color="black", linestyle=styles[k])

    ax0.set_ylabel(u_labels[0])
    axM.set_ylabel(grouped_ylabel)

    # legend on grouped axis: τ1 τ2 τ3 (black handles)
    handles = [Line2D([0], [0], color="black", linestyle=styles[k]) for k in range(3)]
    axM.legend(handles, list(group_legend_labels), loc="best")

    for ax in u_axes:
        ax.grid(True, alpha=0.2)

    u_axes[-1].set_xlabel("Time [s]")
    return 2


def plot_u_multi(
        axes: Sequence[Axes],
        runs_u: Iterable[tuple[np.ndarray, np.ndarray, str]],
        u_labels: Sequence[str],
        *,
        start_idx: int = 0,
        grouped_ylabel: str = r"Moments [N.m]",
        group_legend_labels: Sequence[str] = (r"$\tau_1$", r"$\tau_2$", r"$\tau_3$"),
) -> int:
    runs_u = list(runs_u)
    assert len(runs_u) > 0
    u_dim = int(runs_u[0][1].shape[1])

    if u_dim == 2:
        return plot_u_multi_2d(
            axes=axes,
            runs_u=runs_u,
            u_labels=u_labels,
            start_idx=start_idx,
            show_run_legend=False,
        )
    elif u_dim == 4:
        return plot_u_multi_3d(
            axes=axes,
            runs_u=runs_u,
            u_labels=u_labels,
            start_idx=start_idx,
            grouped_ylabel=grouped_ylabel,
            group_legend_labels=group_legend_labels,
        )
    else:
        raise ValueError(f"Unsupported u_dim={u_dim}. Expected 2 or 4.")