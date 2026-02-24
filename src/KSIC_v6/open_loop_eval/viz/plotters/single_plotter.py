from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Callable, Optional

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib import rc_context

from KSIC_v6.viz.primitives_single import plot_x, plot_u
from KSIC_v6.viz.labels import get_u_labels, get_x_labels
from KSIC_v6.viz.axes_layout import get_shared_ylim_groups_state, apply_shared_ylims
from KSIC_v6.viz.style import DEFAULT_RC_PARAMS, save_figure

GetArray = Callable[[Any], np.ndarray]
GetOptArray = Callable[[Any], Optional[np.ndarray]]

@dataclass(frozen=True)
class SinglePlotExtractors:
    """
    Defines how to extract arrays to plot from an arbitrary rollout output object.
    All arrays are expected to be numpy arrays.
    """
    get_x_gt: GetArray
    get_x_pred: GetOptArray
    get_u: GetArray


class SingleStateInputPlotter:
    """
    Plot ONE rollout (single trajectory):
      - states: x_gt vs x_pred (aligned)
      - inputs: u

    The class is modality-agnostic. You inject extractors at init.
    """
    WIDTH_FIGURES: float = 5.5
    HEIGHT_FIGURES: float = 1.0

    def __init__(
            self,
            drone_dim: int,
            dt: float,
            layout: str,
            only_position: bool,
            path: Path,
            extractors: SinglePlotExtractors,
    ) -> None:
        assert drone_dim in (2, 3)
        assert layout in ("two_columns", "single_column"), f"Unknown layout='{layout}'"

        self.drone_dim = int(drone_dim)
        self.dt = float(dt)
        self.layout = layout
        self.only_position = bool(only_position)
        self.path = Path(path)
        self.extractors = extractors

        self.rc_params = DEFAULT_RC_PARAMS
        self.output: Any = None

    # -------------------------
    # Public API
    # -------------------------
    def pipeline(self, output: Any) -> None:
        """
        Run a full visualization pipeline for a single trajectory.
        """
        self.output = output
        save_dir = self.path
        save_dir.mkdir(parents=True, exist_ok=True)

        with rc_context(self.rc_params):
            self.save_sim_result(
                self.output,
                save_dir / "results",
                self.extractors
            )
            self.plot_states(save_dir)
            self.plot_inputs(save_dir)

    # -------------------------
    # States
    # -------------------------
    def plot_states(self, save_dir: Path) -> None:
        if self.layout == "two_columns":
            self.plot_x_two_columns(save_dir)
        elif self.layout == "single_column":
            self.plot_x_single_column(save_dir)
        else:
            raise ValueError(f"Unknown layout '{self.layout}'")

    def plot_x_two_columns(self, save_dir: Path) -> None:
        x_gt, x_pred_aligned, time, labels = self._prepare_state_plot_data()

        x_dim = x_gt.shape[1]
        assert x_dim % 2 == 0, f"x_dim doit être pair pour 2 colonnes, reçu {x_dim}"
        n_rows = x_dim // 2
        n_cols = 2

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self.WIDTH_FIGURES, n_rows * self.HEIGHT_FIGURES),
            constrained_layout=True,
            squeeze=False,
        )
        axes = np.asarray(axes, dtype=object)

        first = True
        for i in range(x_dim):
            # fill column-by-column (like your previous code)
            row = i % n_rows
            col = i // n_rows
            ax: Axes = axes[row, col]

            plot_x(
                [ax],
                time,
                [labels[i]],
                x_pred_aligned[:, i],
                x_gt[:, i],
                first,
            )
            first = False

            if row < n_rows - 1:
                ax.set_xlabel("")

        groups = get_shared_ylim_groups_state(self.drone_dim, self.only_position)
        apply_shared_ylims(
            axes,
            x_gt,
            x_pred_aligned,
            groups_1based=groups,
            pad_frac=0.05,
        )
        save_figure(fig, save_dir, "states.pdf")

    def plot_x_single_column(self, save_dir: Path) -> None:
        x_gt, x_pred_aligned, time, labels = self._prepare_state_plot_data()
        x_dim = x_gt.shape[1]

        if self.only_position:
            x_dim_disp = int(x_dim/2)
        else:
            x_dim_disp = x_dim

        fig, axes = plt.subplots(
            x_dim_disp,
            1,
            figsize=(self.WIDTH_FIGURES, x_dim_disp * self.HEIGHT_FIGURES),
            constrained_layout=True,
            squeeze=False,
        )
        axes = np.asarray(axes).flatten()

        first = True
        for i in range(x_dim_disp):
            ax: Axes = axes[i]
            plot_x(
                [ax],
                time,
                [labels[i]],
                None if x_pred_aligned is None else x_pred_aligned[:, i:i + 1],
                x_gt[:, i:i + 1],
                first,
            )
            first = False

            if i < x_dim_disp - 1:
                ax.set_xlabel("")

        # groups = get_shared_ylim_groups_state(self.drone_dim, self.only_position)
        # apply_shared_ylims(
        #     axes.reshape(-1, 1),
        #     x_gt,
        #     x_pred_aligned,
        #     groups_1based=groups,
        #     pad_frac=0.05,
        # )
        save_figure(fig, save_dir, "states_single_column.pdf")

    def _prepare_state_plot_data(self):
        x_gt = self.extractors.get_x_gt(self.output)
        x_pred = self.extractors.get_x_pred(self.output)
        labels = get_x_labels(self.drone_dim, self.only_position)
        num_steps = x_pred.shape[0]
        time = np.arange(num_steps) * self.dt
        return x_gt, x_pred, time, labels

    # -------------------------
    # Inputs
    # -------------------------
    def plot_inputs(self, save_dir: Path) -> None:
        u = self.extractors.get_u(self.output)  # (T-1, u_dim)
        labels = get_u_labels(self.drone_dim,)
        u_dim = u.shape[1]
        time_u = np.arange(u.shape[0]) * self.dt

        fig, axes = plt.subplots(
            u_dim, 1,
            figsize=(self.WIDTH_FIGURES, max(1, u_dim) * self.HEIGHT_FIGURES * 2.0),
            constrained_layout=True,
            squeeze=False,
        )
        axes = np.asarray(axes).flatten()
        plot_u(axes, time_u, u, labels)
        save_figure(fig, save_dir, "inputs.pdf")

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def save_sim_result(result: Any, path: Path, extractors: SinglePlotExtractors) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "x_gt": extractors.get_x_gt(result),
            "x_pred": extractors.get_x_pred(result),
            "u": extractors.get_u(result),
        }

        np.savez_compressed(path.with_suffix(".npz"), **data)