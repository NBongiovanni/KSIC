#!/usr/bin/env python
"""
Main script for executing MPC control simulations.

Execution modes:
- Single simulation with visualization
- Multiple simulations with statistics computation
"""
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import numpy as np

from KSIC_v6 import utils
from KSIC_v6.closed_loop_eval import (
    prepare_for_closed_loop_eval,
    redirect_output_to_file,
    run_closed_loop_simulations,
    run_closed_loop_visualization
)
from KSIC_v6.utils.cases_loader import load_cases

# ============================================================================
# CONFIGURATION
# ============================================================================
MODALITY = "vision"
DRONE_DIM = 2
SEED = 3
CASE_ID = 21

# ============================================================================
# ENTRY POINT
# ============================================================================

def main(case_id: int) -> None:
    """Main entry point for the control script."""
    random.seed(SEED)
    np.random.seed(SEED)
    logger = utils.setup_logging()
    cases = load_cases(MODALITY)
    case = cases[case_id]

    # Config loading:
    control_env = prepare_for_closed_loop_eval(
        MODALITY,
        logger,
        case.run_status,
        case.stamp,
        DRONE_DIM,
        case.control_config,
        case.epoch,
        case.geom_losses,
        SEED
    )

    # Run simulations with output redirection
    log_file_path: Path = control_env.control_params["control_runs_dir"] / "simulation.log"
    with redirect_output_to_file(log_file_path, also_print=True):
        simulation_results = run_closed_loop_simulations(
            DRONE_DIM,
            control_env.num_simulations,
            control_env.model_params,
            control_env.control_params,
            control_env.solver_backend,
            control_env.koop_model,
            control_env.x_scaler,
            control_env.u_scaler
        )

        run_closed_loop_visualization(
            control_env.control_params["plot"],
            control_env.control_params["render_images"],
            list(range(control_env.num_simulations)),
            control_env.model_params,
            control_env.control_params,
            control_env.control_params["control_runs_dir"],
            simulation_results,
            control_env.control_params["num_steps_simulation"],
            "Bilinear model",
            True,
            "single_column",
        )

    print(f"\n✅ Simulation complete. Full output saved to: {log_file_path}")


if __name__ == '__main__':
    main(CASE_ID)