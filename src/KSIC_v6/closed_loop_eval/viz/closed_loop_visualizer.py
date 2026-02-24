from pathlib import Path
import pickle

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib import rc_context

from KSIC_v6.viz.axes_layout import get_shared_ylim_groups_state, apply_shared_ylims

from KSIC_v6 import utils
from KSIC_v6.closed_loop_eval.simulation.results import SimResults
from KSIC_v6.viz import (
    save_figure,
    DEFAULT_RC_PARAMS,
    get_x_labels,
    get_u_labels,
    get_angle_indexes
)
from KSIC_v6.viz.primitives_single import plot_x, plot_u

class ClosedLoopVisualizer:
    WIDTH_FIGURES: float = 5.5
    HEIGHT_FIGURES: float = 1.0

    def __init__(
            self,
            drone_dim: int,
            results: SimResults,
            run_dir: Path,
            dt: float,
            nominal_control: bool,
            only_positions: bool,
            layout: str,
    ):
        self.run_dir = run_dir
        self.dt = dt
        self.plot_dir = run_dir
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.results = results
        self.drone_dim = drone_dim
        self.nominal_control = nominal_control
        self.layout = layout
        self.only_positions = only_positions

        self.rc_params = DEFAULT_RC_PARAMS
        self.angles_indexes = get_angle_indexes(drone_dim)
        self.x_dim, self.u_dim, self.x_ref_dim = utils.get_dimensions(drone_dim)
        assert self.layout in ("two_columns", "single_column"), f"Unknown layout='{self.layout}'"

    def visualize(self) -> None:
        with rc_context(self.rc_params):
            self.save_sim_result(self.results, self.run_dir / "results.pkl")
            self.plot_states()
            self.plot_u()

    def plot_states(self) -> None:
        x = self.results.x_data.traj  # (T, x_dim)
        x_ref = self.results.x_data.ref_traj  # (T, x_dim)
        time = self.results.time

        labels = get_x_labels(self.drone_dim, self.only_positions)

        x_dim = x.shape[1]
        x_dim_displayed = x_dim if not self.only_positions else int(x_dim/2)
        x_disp = x.copy()
        x_ref_disp = x_ref.copy()

        if self.layout == "two_columns":
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
                row = i % n_rows
                col = i // n_rows
                ax: Axes = axes[row, col]

                plot_x(
                    [ax],
                    time,
                    [labels[i]],
                    x_disp[:, i:i + 1],
                    x_ref_disp[:, i:i + 1],
                    show_legend=first,
                )
                first = False

                if row < n_rows - 1:
                    ax.set_xlabel("")

            groups = get_shared_ylim_groups_state(self.drone_dim, self.only_positions)
            apply_shared_ylims(
                axes,
                x_ref_disp,  # "gt"
                x_disp,  # "pred"
                groups_1based=groups,
                pad_frac=0.05,
            )
            save_figure(fig, self.plot_dir, "states.pdf")

        else:  # single_column
            fig, axes = plt.subplots(
                x_dim_displayed,
                1,
                figsize=(self.WIDTH_FIGURES, x_dim * self.HEIGHT_FIGURES),
                constrained_layout=True,
                squeeze=False,
            )
            axes = np.asarray(axes).flatten()

            first = True
            for i in range(x_dim_displayed):
                ax: Axes = axes[i]
                plot_x(
                    [ax],
                    time,
                    [labels[i]],
                    x_disp[:, i:i + 1],
                    x_ref_disp[:, i:i + 1],
                    show_legend=first,
                )
                first = False
                if i < x_dim_displayed - 1:
                    ax.set_xlabel("")

            groups = get_shared_ylim_groups_state(self.drone_dim, self.only_positions)
            apply_shared_ylims(
                axes.reshape(-1, 1),
                x_ref_disp,
                x_disp,
                groups_1based=groups,
                pad_frac=0.05,
            )
            save_figure(fig, self.plot_dir, "states_single_column.pdf")

    def plot_u(self) -> None:
        time = self.results.time
        u_physical_traj = self.results.inputs_data.u_physical
        labels = get_u_labels(self.drone_dim)

        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(12, 4),
            squeeze=False,
            constrained_layout=True
        )
        axes = axes.ravel()
        plot_u(axes, time[:-1], u_physical_traj, labels)
        save_figure(fig, self.plot_dir, "inputs.pdf")

    @staticmethod
    def save_sim_result(result: SimResults, path: Path) -> None:
        """Sauvegarde via pickle."""
        with open(path, "wb") as f:
            pickle.dump(result, f)
