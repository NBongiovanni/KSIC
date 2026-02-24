#!/usr/bin/env python
"""
Main script for executing MPC control simulations.

Execution modes:
- Single simulation with visualization
- Multiple simulations with statistics computation
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from dataclasses import dataclass
from KSIC_v6.closed_loop_eval import ClosedLoopVisualizerMulti
from KSIC_v6 import utils

# ============================================================================
# CONFIGURATION
# ============================================================================
DRONE_DIM = 3

@dataclass(frozen=True)
class ModelSimuConfig:
    id: int
    color: str
    label: str

@dataclass(frozen=True)
class ClosedLoopComparisonConfig:
    num_simulations: int = 10
    modality: str = "sensor"  # "vision" or "sensor"
    dt: float = 0.01
    case_1: ModelSimuConfig = ModelSimuConfig(
        id=5,
        label="Bilinear Koopman",
        color="tab:orange"
    )
    case_2: ModelSimuConfig = ModelSimuConfig(
        id=6,
        label="Linear Koopman",
        color="tab:blue"
    )

# ============================================================================
# ENTRY POINT
# ============================================================================
def main() -> None:
    cfg = ClosedLoopComparisonConfig()
    logger = utils.setup_logging()
    run_status = "final"

    cases = utils.load_cases(cfg.modality)
    case_1 = cases[cfg.case_1.id]
    case_2 = cases[cfg.case_2.id]

    output_dir = Path("/home/bongiovanni/Desktop/PycharmProjects/KSIC_v6/outputs")
    run_dir_1 = output_dir / run_status / "sensor" / f"{case_1.drone_dim}d" / "models" / case_1.stamp
    run_dir_2 = output_dir / run_status / "sensor" / f"{case_2.drone_dim}d" / "models" / case_2.stamp

    names = [cfg.case_1.label, cfg.case_2.label]
    colors = [cfg.case_1.color, cfg.case_2.color]

    stamp_simulation_1 = case_1.closed_loop_simulations["setpoint_tracking"]
    stamp_simulation_2 = case_2.closed_loop_simulations["setpoint_tracking"]

    run_dir_1 = run_dir_1 / "eval" / "closed_loop" / stamp_simulation_1 / "run_0" / "results.pkl"
    run_dir_2 = run_dir_2 / "eval" / "closed_loop" / stamp_simulation_2 / "run_0" / "results.pkl"

    comparisons_dir = Path("comparisons") / "closed_loop"
    base_ctrl_runs_dir_save = output_dir / run_status / cfg.modality / f"{DRONE_DIM}d" / comparisons_dir
    comparison_stamp = utils.make_timestamped_dir(logger)
    run_dir = base_ctrl_runs_dir_save / comparison_stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(run_dir)

    ctrl_visualizer = ClosedLoopVisualizerMulti(
        DRONE_DIM,
        run_dir,
        cfg.dt,
        names,
        colors,
    )
    ctrl_visualizer.load_results(run_dir_1, run_dir_2)
    ctrl_visualizer.visualize()

if __name__ == '__main__':
    main()