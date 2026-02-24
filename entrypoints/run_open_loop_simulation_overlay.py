from dataclasses import dataclass
from pathlib import Path

from KSIC_v6 import utils
from KSIC_v6.open_loop_eval import SensorStateRolloutPlotterMulti, VisionStateRolloutPlotterMulti

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class ModelSimuConfig:
    id: int
    label: str
    color: str

@dataclass(frozen=True)
class OpenLoopComparisonConfig:
    num_traj: int = 10
    modality: str = "sensor"
    dt: float = 0.01
    case_1: ModelSimuConfig = ModelSimuConfig(
        id=5,
        color="tab:orange",
        label="Bilinear Koopman",
    )
    case_2: ModelSimuConfig = ModelSimuConfig(
        id=6,
        color="tab:blue",
        label="Linear Koopman",
    )

def main() -> None:
    cfg = OpenLoopComparisonConfig()
    logger = utils.setup_logging()

    cases = utils.load_cases(cfg.modality)
    case_1 = cases[cfg.case_1.id]
    case_2 = cases[cfg.case_2.id]
    run_status = "final"

    output_dir = Path("/home/bongiovanni/Desktop/PycharmProjects/KSIC_v6/outputs")
    base_run_dir_1 = output_dir / run_status / "sensor" / f"{case_1.drone_dim}d" / "models" / case_1.stamp
    base_run_dir_2 = output_dir / run_status / "sensor" / f"{case_2.drone_dim}d" / "models" / case_2.stamp

    names = [cfg.case_1.label, cfg.case_2.label]
    colors = [cfg.case_1.color, cfg.case_2.color]

    stamp_simulation_1 = case_1.open_loop_simulations["setpoint_tracking"]
    stamp_simulation_2 = case_2.open_loop_simulations["setpoint_tracking"]

    stamp_out = utils.make_timestamped_dir(logger)

    run_dir_1 = base_run_dir_1 / "eval" / "open_loop" / stamp_simulation_1 / "rollout_0" / "results.pkl"
    run_dir_2 = base_run_dir_2 / "eval" / "open_loop" / stamp_simulation_2 / "rollout_0" / "results.pkl"

    # ----------------------------
    # Traj comparison
    # ----------------------------
    interim_dir = output_dir / run_status / cfg.modality / f"{case_1.drone_dim}d"
    open_loop_cmp_dir = interim_dir / "comparisons" / "open_loop"
    comparison_dir = open_loop_cmp_dir / stamp_out

    for i in range(cfg.num_traj):
        traj_out_dir = comparison_dir / f"rollout_{i}"
        traj_out_dir.mkdir(parents=True, exist_ok=True)

        if cfg.modality == "vision":
            visualizer = VisionStateRolloutPlotterMulti(
                drone_dim=case_1.drone_dim,
                only_position=True,
                plot_dir=traj_out_dir,
                names=names,
                dt=cfg.dt,
                layout="single_column",  # ou "two_columns"
                colors=colors
            )
        elif cfg.modality == "sensor":
            visualizer = SensorStateRolloutPlotterMulti(
                drone_dim=case_1.drone_dim,
                only_position=False,
                plot_dir=traj_out_dir,
                names=names,
                dt=cfg.dt,
                colors=colors,
                layout="two_columns"
            )
        else:
            raise ValueError(f"Unknown modality='{cfg.modality}'. Expected 'vision' or 'sensor'.")
        visualizer.load_results(run_dir_1, run_dir_2)
        visualizer.pipeline()

if __name__ == "__main__":
    main()
