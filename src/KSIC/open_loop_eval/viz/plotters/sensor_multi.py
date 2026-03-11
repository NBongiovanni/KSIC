from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from KSIC.models.outputs.sensor_outputs import SensorValForwardOutputs
from KSIC.open_loop_eval.viz.plotters.base_multi import BaseRolloutPlotterMulti
from KSIC.utils import to_numpy

class SensorStateRolloutPlotterMulti(BaseRolloutPlotterMulti[SensorValForwardOutputs]):
    """
    Multi-run sensor comparison on ONE trajectory.
    Only responsibility: define how to extract x_gt/x_pred.
    """
    def __init__(
            self,
            drone_dim: int,
            only_position: bool,
            plot_dir,
            names: Sequence[str],
            dt: float,
            colors: Sequence[str],
            layout: str,
            filename: str = "open_loop_simulation.pdf",
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
            x_gt = to_numpy(out.state_gt_physical)  # (T, x_dim)
            x_pred = to_numpy(out.pred.state)  # (T-1, x_dim)

            T = x_gt.shape[0]
            L = min(T - 1, x_pred.shape[0])

            t = np.arange(1, 1 + L) * self.dt
            yield t, x_gt[1 : 1 + L], x_pred[:L], name