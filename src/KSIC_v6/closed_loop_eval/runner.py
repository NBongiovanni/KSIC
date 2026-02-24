#!/usr/bin/env python
"""
Main script for executing MPC control simulations.

Execution modes:
- Single simulation with visualization
- Multiple simulations with statistics computation
"""
from pathlib import Path
import copy
from sklearn.preprocessing import StandardScaler

from KSIC_v6 import utils
from KSIC_v6.closed_loop_eval import (
    SolverBackend,
    create_state_init_conditions,
    process_statistics,
    MultiRunMetrics,
    create_simulator,
    ReferenceTrajBuilderVision,
    ReferenceTrajBuilderSensor,
    VisionMPCController,
    SensorMPCController,
    create_plant,
)
from KSIC_v6.models import VisionKoopModel
from KSIC_v6.rendering import render_trajectory_sequence, create_snapshots_figure
from .simulation.results import SimResults
from .viz.closed_loop_visualizer import ClosedLoopVisualizer

def run_closed_loop_simulations(
        drone_dim: int,
        num_simulations: int,
        model_params: dict,
        control_params: dict,
        solver_backend: SolverBackend,
        koop_model: VisionKoopModel,
        x_scaler: StandardScaler,
        u_scaler: StandardScaler
) -> list:
    """
    Execute multiple simulations and compute statistics.

    Typically used for:
    - Evaluating controller robustness
    - Computing average metrics
    - Analyzing performance variance

    Args:
        modality: Sensor or vision modalities
        drone_dim: Drone dimension
        num_simulations: Number of simulations to run
        model_params: Model configuration parameters
        control_params: Control configuration parameters
        solver_backend: MPC solver backend (Acados or do-mpc)
        koop_model: Learned Koopman model
        x_scaler: State input scaler
        u_scaler: Control input scaler

    Returns:
        tuple: (mse_x_mean, mse_z_mean) - Average MSE across all simulations
    """
    modality = control_params["modality"]
    metrics = MultiRunMetrics()
    simulation_results: list = []

    A, B = koop_model.construct_koop_matrices()
    # analyze_AB(A, B, paths.closed_loop_eval_dir)
    print(f"A:\n{A}")
    print(f"B:\n{B}")

    x_dim, u_dim, x_ref_dim = utils.get_dimensions(drone_dim)

    if modality == "sensor":
        controller = SensorMPCController(
            model_params=model_params,
            control_params=control_params,
            solver_backend=solver_backend,
            koop_model=koop_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
        )
    else:
        controller = VisionMPCController(
            model_params=model_params,
            control_params=control_params,
            solver_backend=solver_backend,
            koop_model=koop_model,
            u_scaler=u_scaler,
            render_resolution=512,
        )
    controller.build()

    plant = create_plant(
        drone_dim,
        control_params["use_nominal_plant"],
        control_params["dt"],
        koop_model,
    )
    simulator = create_simulator(control_params, plant, controller)

    for i in range(num_simulations):
        print(f"\n{'=' * 60}")
        print(f"Simulation {i + 1}/{num_simulations}")
        print('=' * 60)
        x_init = create_state_init_conditions(x_dim, control_params)

        if modality == "sensor":
            ref_builder = ReferenceTrajBuilderSensor(
                control_params=control_params,
                koop_model=koop_model,
                specs=copy.deepcopy(control_params["x_ref"]["specs"]),
                drone_dim=drone_dim,
                x_scaler=x_scaler
            )
        elif modality == "vision":
            ref_builder = ReferenceTrajBuilderVision(
                control_params=control_params,
                koop_model=koop_model,
                specs=copy.deepcopy(control_params["x_ref"]["specs"]),
                drone_dim=drone_dim,
                )
        else:
            raise ValueError(f"Modality {modality} not supported.")
        state_ref_traj, im_ref_traj, z_ref_traj = ref_builder.build()

        controller.reset()
        controller.set_reference(state_ref_traj, z_ref_traj, im_ref_traj)
        simulation_output = simulator.run(x_init)
        simulation_results.append(simulation_output)
        sqp_iters = controller.backend.sqp_iters
        stats = process_statistics(simulation_output, sqp_iters)
        print(stats.short_summary())
        metrics.add(stats)

    metrics.display(num_simulations)
    return simulation_results


def render_rollout_images(simulation_results: SimResults, rollout_dir: Path) -> None:
    controlled_images = simulation_results.im_data.traj
    ref_images = simulation_results.im_data.ref_traj
    out_dir = rollout_dir / "images"
    render_trajectory_sequence(
        pred_imgs=controlled_images,
        gt_imgs=ref_images,
        path_results=out_dir,
    )


def run_closed_loop_visualization(
        plot: bool,
        render_images: bool,
        simulation_indexes: list,
        model_params: dict,
        control_params: dict,
        base_ctrl_runs_dir: Path,
        simulation_results: list[SimResults],
        num_steps: int,
        label: str,
        only_positions: bool,
        layout: str,
) -> None:

    if plot:
        for i in simulation_indexes:
            run_dir = base_ctrl_runs_dir / f"run_{i}"
            run_dir.mkdir(parents=True, exist_ok=True)
            ctrl_visualizer = ClosedLoopVisualizer(
                model_params["drone"]["dim"],
                simulation_results[i],
                run_dir,
                control_params["dt"],
                control_params["use_nominal_plant"],
                only_positions,
                layout,
            )
            ctrl_visualizer.visualize()
            if render_images:
                render_rollout_images(simulation_results[i], run_dir)
                im_dir_i = run_dir / "images"
                out_dir_i = run_dir / "snapshots.png"
                create_snapshots_figure(
                    im_dir=im_dir_i,
                    out_path=out_dir_i,
                    step_count=120,
                    label=label,
                    n_rows=2,
                    stride=6
                )
