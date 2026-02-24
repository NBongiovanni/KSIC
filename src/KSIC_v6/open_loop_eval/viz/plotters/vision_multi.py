from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from KSIC_v6.models.outputs.vision_outputs import VisionValForwardOutputs
from KSIC_v6.open_loop_eval.viz.plotters.base_multi import BaseRolloutPlotterMulti
from KSIC_v6.utils import to_numpy

class VisionStateRolloutPlotterMulti(BaseRolloutPlotterMulti[VisionValForwardOutputs]):
    """
    Multi-run vision comparison on ONE trajectory.
    Only responsibility: define how to extract x_gt/x_pred.
    """
    def __init__(
            self,
            drone_dim: int,
            only_position: bool,
            plot_dir: Path,
            names: Sequence[str],
            dt: float,
            colors: Sequence[str],
            layout: str = "two_columns",
            filename: str = "open_loop_experiment.pdf",
    ) -> None:
        super().__init__(
            drone_dim=drone_dim,
            only_position=only_position,
            plot_dir=plot_dir,
            names=names,
            dt=dt,
            layout=layout,
            filename=filename,
            colors=colors
        )

    def _iter_runs_x(self) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        for name, out in zip(self.names, self.results_list):
            x_gt = to_numpy(out.g_t.state)  # (T, x_dim)
            x_pred = to_numpy(out.pred.state)  # (T-1, x_dim)

            T = x_gt.shape[0]
            t = np.arange(1, T) * self.dt
            yield t, x_gt[1 : T], x_pred[:T-1], name