from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_context
import matplotlib.gridspec as gridspec

from KSIC import utils
from KSIC.viz import (
    plot_u_multi,
    plot_state_multi,
    save_figure,
    DEFAULT_RC_PARAMS_MULTI,
    get_u_labels,
    get_x_labels,
    get_angle_indexes,
)

WIDTH_FIGURES = 3.5
HEIGHT_FIGURES = 1.5

class ClosedLoopVisualizerMulti:
    """
    Version simplifiée : plus de paramètre 'tags'.
    Les légendes sont automatiquement déduites du nom du fichier
    ou d'un identifiant de type run_0, run_1, etc.
    """
    def __init__(
            self,
            drone_dim: int,
            plot_dir: Path,
            dt: float,
            names: list[str],
            colors: list[str],
    ):
        self.plot_dir = plot_dir
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.drone_dim = drone_dim
        self.dt = dt
        self.results_list = None
        self.names = names
        self.colors = colors
        self.x_dim, self.u_dim, self.x_ref_dim = utils.get_dimensions(drone_dim)

    # ------------------------------------------------------------------
    def visualize(self) -> None:
        """Affiche les figures multi-run."""
        with rc_context(DEFAULT_RC_PARAMS_MULTI):
            self._plot_simulation_two_columns()
    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    def _plot_simulation_two_columns(self) -> None:
        """
        Paper figure:
          - full state x on TWO columns (x_dim must be even)
          - control u below (one subplot per u component)
        """
        # --- state dimension ---
        first = next(r for r in self.results_list if r.x_data is not None)
        x_dim = int(first.x_data.traj.shape[1])
        assert x_dim % 2 == 0, "x_dim must be even to display states on 2 columns"

        labels_x = get_x_labels(self.drone_dim, False)
        labels_u = get_u_labels(self.drone_dim)

        # --- build runs for x ---
        runs_x = []
        for name, r in zip(self.names, self.results_list):
            runs_x.append((r.time, r.x_data.ref_traj, r.x_data.traj, name))

        # --- build runs for u (scaled) ---
        runs_u = []
        for name, r in zip(self.names, self.results_list):
            t_u = r.time[:-1]
            runs_u.append((t_u, r.inputs_data.u_physical, name))

        # -------------------------
        # Layout: 2 columns for x, then u spans both columns
        # -------------------------
        x_rows = x_dim // 2
        group_last_u = 3  # <- les 3 dernières composantes de u sur le même graphe
        u_axes_count = (self.u_dim - group_last_u) + 1  # +1 pour le subplot groupé final

        total_rows = x_rows + u_axes_count

        fig = plt.figure(
            figsize=(WIDTH_FIGURES * 2.0, HEIGHT_FIGURES * (x_rows + 0.7 * u_axes_count))
        )

        gs = gridspec.GridSpec(
            nrows=total_rows,
            ncols=2,
            figure=fig,
            height_ratios=[1.0] * x_rows + [0.7] * u_axes_count,
            wspace=0.25,
        )

        # --- state axes (x_rows x 2) ---
        x_axes_grid = np.empty((x_rows, 2), dtype=object)
        for r in range(x_rows):
            for c in range(2):
                x_axes_grid[r, c] = fig.add_subplot(gs[r, c])

        x_axes = list(x_axes_grid[:, 0]) + list(x_axes_grid[:, 1])

        plot_state_multi(
            x_axes,
            runs_x,
            labels_x,
            self.colors,
            self.x_ref_dim,
            "Reference trajectory",
        )

        u_axes = []
        for i in range(u_axes_count):
            rr = x_rows + i
            u_axes.append(fig.add_subplot(gs[rr, :]))

        plot_u_multi(
            axes=u_axes,
            runs_u=runs_u,
            u_labels=labels_u,
            start_idx=0,
            group_last=group_last_u,
            grouped_ylabel=r"Moments [N.m]",
            colors=self.colors,
            group_line_styles=["solid", "dotted", "dashed"],
            group_legend_labels=[r"$\tau_1$", r"$\tau_2$", r"$\tau_3$"],
            group_legend_mode="black",
            legend_loc="upper right",
        )

        all_axes = x_axes + u_axes
        fig.align_ylabels(all_axes)
        save_figure(fig, self.plot_dir, "closed_loop_simulation.pdf")

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    @staticmethod
    def load_sim_result(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_results(self, *paths: Path) -> None:
        """Charge plusieurs fichiers pickle et crée les noms automatiquement."""
        loaded = []
        for p in paths:
            loaded.append(self.load_sim_result(p))
        self.results_list = loaded