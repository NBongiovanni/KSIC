import multiprocessing
import matplotlib
matplotlib.use("Agg")

from KSIC import utils
from KSIC.open_loop_eval import (
    extract_one_rollout_vision,
    render_open_loop_rollouts,
    RenderOpenLoopConfig,
    open_loop_simulation_vision_pipeline,
    label_from_case
)
# ============================================================================
# CONFIGURATION
# ============================================================================
MODALITY = "vision"
SEED = 3
CASE_ID = 21
NUM_SIMULATIONS = 10
NUM_STEPS = 10
PHASE = "val_2"
dataset_version = 33

# ============================================================================
# ENTRY POINT
# ============================================================================
def main(case_id: int) -> None:
    cases = utils.load_cases(MODALITY)
    case = cases[case_id]
    logger = utils.setup_logging()
    stamp_open_loop = utils.make_timestamped_dir(logger)

    simulation_output = open_loop_simulation_vision_pipeline(
        case,
        PHASE,
        MODALITY,
        NUM_STEPS,
        SEED,
        stamp_open_loop,
        case.dt,
        dataset_version,
    )
    eval_dir = simulation_output.run_dir / "eval" / "open_loop" / stamp_open_loop

    config = RenderOpenLoopConfig(
        modality=MODALITY,
        drone_dim=case.drone_dim,
        dt=case.dt,
        phase=PHASE,
        epoch=case.epoch,
        layout="single_column",
        only_position=True,
        num_rollouts=NUM_SIMULATIONS,
        render_images=True,
        snapshots=True,
        num_steps=NUM_STEPS,
        label=label_from_case(case),
    )

    render_open_loop_rollouts(
        config=config,
        output=simulation_output.val_output,
        eval_dir=eval_dir,
        extract_one_rollout=extract_one_rollout_vision,
    )

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main(CASE_ID)
