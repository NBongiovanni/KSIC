from __future__ import annotations

from pathlib import Path
import pickle
from typing import Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar, cast

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc_context

from KSIC.utils import to_numpy, get_dimensions
from KSIC.viz.labels import get_u_labels, get_x_labels
from KSIC.viz.axes_layout import apply_shared_ylims, get_shared_ylim_groups_state
from KSIC.viz.style import DEFAULT_RC_PARAMS_MULTI, save_figure
from KSIC.viz.primitives_multi import plot_u_multi, plot_state_multi

TOutput = TypeVar("TOutput")

class BaseRolloutPlotterMulti(Generic[TOutput]):
    WIDTH_FIGURES_1COL: float = 3.5
    WIDTH_FIGURES_2COL: float = 3.5 * 2
    HEIGHT_FIGURES: float = 1.2

    def __init__(
            self,
            drone_dim: int,
            only_position: bool,
            plot_dir: Path,
            names: Sequence[str],
            dt: float,
            colors: Sequence[str],
            layout: str = "two_columns",
            filename: str = "open_loop_simulation.pdf",
    ) -> None:
        assert drone_dim in (2, 3)
        assert layout in ("single_column", "two_columns"), f"Invalid layout={layout}"

        self.drone_dim = int(drone_dim)
        self.only_position = bool(only_position)
        self.plot_dir = plot_dir
        self.names = list(names)
        self.dt = float(dt)
        self.layout = layout

        self.x_dim, self.u_dim, self.x_ref_dim = get_dimensions(drone_dim)
        self.filename = filename
        self.colors = colors

        self.rc_params = DEFAULT_RC_PARAMS_MULTI
        self.results_list: Optional[List[TOutput]] = None

    def load_results(self, *paths: Path) -> None:
        loaded: List[TOutput] = []
        for p in paths:
            loaded.append(self.load_sim_result(p))
        self.results_list = loaded

    def pipeline(self) -> None:
        assert len(self.results_list) == len(self.names)

        with rc_context(self.rc_params):
            self._plot()

    def _iter_runs_x(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        raise NotImplementedError

    def _iter_runs_u(self) -> Iterable[tuple[np.ndarray, np.ndarray, str]]:
        assert self.results_list is not None
        for name, out in zip(self.names, self.results_list):
            u_ph = getattr(out, "inputs_physical")
            u = to_numpy(u_ph)
            t = np.arange(u.shape[0]) * self.dt
            yield t, u, name

    def _plot(self) -> None:
        runs_x = list(self._iter_runs_x())
        runs_u = list(self._iter_runs_u())

        x_labels = get_x_labels(self.drone_dim, self.only_position)
        u_labels = get_u_labels(self.drone_dim, self.only_position)

        # slice states if needed
        x_dim_disp = len(x_labels)
        _, x_gt0, _, _ = runs_x[0]
        x_dim_data = x_gt0.shape[1]
        if x_dim_data != x_dim_disp:
            runs_x = [
                (t, x_gt[:, :x_dim_disp], x_pr[:, :x_dim_disp], name)
                for (t, x_gt, x_pr, name) in runs_x
            ]

        if self.layout == "single_column":
            self._plot_one_column(
                runs_x,
                runs_u,
                x_labels,
                u_labels,
                self.x_dim
            )
        else:
            self._plot_two_columns(
                runs_x,
                runs_u,
                x_labels,
                u_labels,
                self.x_dim,
            )

    def _plot_one_column(
            self,
            runs_x: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
            runs_u: Sequence[tuple[np.ndarray, np.ndarray, str]],
            x_labels: Sequence[str],
            u_labels: Sequence[str],
            n_ref_disp: int,
    ) -> None:
        _, x_gt0, _, _ = runs_x[0]
        x_dim = x_gt0.shape[1]

        height_ratios = [1.0] * x_dim + [0.6] * self.u_dim
        fig, axes = plt.subplots(
            nrows=x_dim + self.u_dim,
            ncols=1,
            figsize=(self.WIDTH_FIGURES_1COL, self.HEIGHT_FIGURES * (x_dim + 0.6 * self.u_dim)),
            gridspec_kw={"height_ratios": height_ratios},
            constrained_layout=True,
        )
        axes = np.asarray(axes).ravel()

        plot_state_multi(
            axes[:x_dim],
            runs_x,
            x_labels,
            colors=self.colors,
            ref_dim=n_ref_disp,
            gt_label="True trajectory"
        )

        plot_u_multi(
            axes=axes,
            runs_u=runs_u,
            u_labels=u_labels,
            start_idx=x_dim,
            grouped_ylabel=r"Moments [N.m]",
        )

        fig.align_ylabels(axes)
        save_figure(fig, self.plot_dir, self.filename)

    def _plot_two_columns(
            self,
            runs_x: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
            runs_u: Sequence[tuple[np.ndarray, np.ndarray, str]],
            x_labels: Sequence[str],
            u_labels: Sequence[str],
            n_ref_disp: int,
    ) -> None:
        _, x_gt0, _, _ = runs_x[0]
        x_dim = x_gt0.shape[1]
        u_axes_count = 2 if self.u_dim == 4 else self.u_dim

        assert x_dim % 2 == 0, "x_dim must be even to display states on 2 columns"
        x_rows = x_dim // 2
        total_rows = x_rows + u_axes_count

        fig = plt.figure(
            figsize=(self.WIDTH_FIGURES_2COL, self.HEIGHT_FIGURES * (x_rows + 0.7 * self.u_dim))
        )
        gs = gridspec.GridSpec(
            nrows=total_rows,
            ncols=2,
            figure=fig,
            height_ratios=[1.0] * x_rows + [0.7] * u_axes_count,
            wspace=0.25,
        )

        x_axes_grid = np.empty((x_rows, 2), dtype=object)
        for r in range(x_rows):
            for c in range(2):
                x_axes_grid[r, c] = fig.add_subplot(gs[r, c])

        x_axes = list(x_axes_grid[:, 0]) + list(x_axes_grid[:, 1])

        plot_state_multi(
            x_axes,
            runs_x,
            x_labels,
            colors=self.colors,
            ref_dim=n_ref_disp,
            gt_label="True trajectory",
        )

        groups = get_shared_ylim_groups_state(self.drone_dim, self.only_position)
        # concaténer toutes les GT et prédictions
        x_gt_all = np.concatenate([r[1] for r in runs_x], axis=0)
        x_pred_all = np.concatenate([r[2] for r in runs_x], axis=0)

        apply_shared_ylims(
            axes=x_axes_grid,
            x_gt=x_gt_all,
            x_pred=x_pred_all,
            groups_1based=groups,
            pad_frac=0.05,
        )

        u_axes = []
        for i in range(u_axes_count):
            rr = x_rows + i
            ax = fig.add_subplot(gs[rr, :])
            u_axes.append(ax)
            ax.grid(True, alpha=0.2)

        all_axes = x_axes + u_axes

        plot_u_multi(
            axes=all_axes,
            runs_u=runs_u,
            u_labels=u_labels,
            start_idx=x_dim,
            grouped_ylabel=r"Moments [N.m]",
        )

        fig.align_ylabels(all_axes)
        save_figure(fig, self.plot_dir, self.filename)

    @staticmethod
    def load_sim_result(path: Path) -> TOutput:
        with open(path, "rb") as f:
            return cast(TOutput, pickle.load(f))