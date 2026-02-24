import multiprocessing
import matplotlib
matplotlib.use("Agg")

from KSIC_v6 import utils
from KSIC_v6.open_loop_eval import (
    open_loop_simulation_sensor_pipeline,
    extract_one_rollout_sensor,
    render_open_loop_rollouts,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
MODALITY = "sensor"
DRONE_DIM = 3
SEED = 3
CASE_ID = 5
NUM_TRAJ = 10
NUM_STEPS = 150
PHASE = "val_2"
DT = 0.01

# ============================================================================
# ENTRY POINT
# ============================================================================
def main(case_id: int) -> None:
    cases = utils.load_cases(MODALITY)
    case = cases[case_id]
    logger = utils.setup_logging()
    stamp_open_loop = utils.make_timestamped_dir(logger)

    simulation_output = open_loop_simulation_sensor_pipeline(
        case,
        PHASE,
        NUM_STEPS,
        MODALITY,
        DRONE_DIM,
        stamp_open_loop,
        SEED,
    )

    render_open_loop_rollouts(
        modality=MODALITY,
        dt=DT,
        drone_dim=DRONE_DIM,
        layout="two_columns",
        only_position=False,
        output=simulation_output["models_outputs"],
        phase=PHASE,
        num_rollouts=NUM_TRAJ,
        epoch=case.epoch,
        eval_dir=simulation_output["open_loop_eval_dir"],
        extract_one_rollout=extract_one_rollout_sensor,
        render_images=False,
        label="Sensor model",
        num_steps=NUM_STEPS,
    )

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main(CASE_ID)
