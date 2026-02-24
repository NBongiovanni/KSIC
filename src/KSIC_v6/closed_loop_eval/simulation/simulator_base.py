import time

import numpy as np
import torch
from tqdm import tqdm

from KSIC_v6.viz import get_angle_indexes
from KSIC_v6.closed_loop_eval.plants.plant import Plant
from KSIC_v6.closed_loop_eval.simulation.results import (
    SimResults, TrajectoryData, InputsData
)
from KSIC_v6.closed_loop_eval.controllers.mpc_controller_base import MPCControllerBase
from ..state_renderer import StateRenderer
from ..vision_observer import VisionObserver

class ControlSimulator:
    """Base class for control simulation."""
    def __init__(
            self,
            control_params: dict,
            plant: Plant,
            controller: MPCControllerBase
    ):
        self.control_params = control_params
        self.plant = plant
        self.controller = controller
        self.n_steps_simulation = control_params["num_steps_simulation"]
        self.dt = control_params["dt"]

        state_renderer = StateRenderer(512) # TODO: define dynamically
        self.observer = VisionObserver(state_renderer)

    def run(self, x_init: np.ndarray) -> SimResults:
        """Run the simulation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")

    def _run_simulation_loop(
            self,
            initial_state: np.ndarray,
    ) -> tuple[list, list]:
        """
        Common simulation loop logic.
        Args:
            initial_state: Initial state vector
        Returns:
            tuple: (state_traj, im_traj)
        """
        start = time.time()
        state_k = np.asarray(initial_state).reshape(-1)

        state_traj: list[np.ndarray] = [state_k.copy()]
        state_km1 = state_k.copy()
        state_traj.append(state_k.copy())
        im_traj: list = [self.observer.observe(state_k)]

        iterator = tqdm(range(self.n_steps_simulation - 1))
        for _ in iterator:
            u_k = self.controller.compute_control(state_k,state_km1)
            state_kp1 = self.plant.update_state(state_k, u_k)
            im_kp1 = self.observer.observe(state_kp1)

            state_traj.append(state_kp1.copy())
            im_traj.append(im_kp1)
            state_km1 = state_k.copy()
            state_k = state_kp1.copy()

        self.compute_total_time(start)
        return state_traj, im_traj

    def process_output(
            self,
            x_traj: list,
            x_ref_traj,
            z_traj: list,
            z_ref_traj,
            im_traj: list,
            im_ref_traj: list,
    ) -> SimResults:
        """Process simulation output into SimResults object."""
        time_vector = np.arange(self.n_steps_simulation+1) * self.dt
        x_traj_array = np.stack([np.asarray(x).reshape(-1) for x in x_traj])


        # z_traj.append(z_traj[-1]) # TODO: handle this à gérer
        z_traj_array = np.stack([np.asarray(z).reshape(-1) for z in z_traj])

        angles_indexes = get_angle_indexes(self.plant.drone_dim)
        x_data = TrajectoryData(
            traj=self.convert_rad_to_deg_np(x_traj_array, angles_indexes),
            ref_traj=self.convert_rad_to_deg_np(x_ref_traj, angles_indexes),
            error=np.abs(x_traj_array - x_ref_traj)
        )
        z_data = TrajectoryData(
            traj=z_traj_array,
            ref_traj=z_ref_traj,
            error=z_traj_array - z_ref_traj
        )
        im_data = TrajectoryData(
            traj=torch.stack(im_traj).detach().cpu(),
            ref_traj=np.stack(im_ref_traj) if im_ref_traj is not None else None,
            error=None,
        )

        u_physical_traj = self.controller.u_physical_traj
        u_physical_traj.append(u_physical_traj[-1])
        u_scaled_traj = self.controller.u_scaled_traj
        u_scaled_traj.append(u_scaled_traj[-1])

        control = InputsData(
            u_physical=np.stack(u_physical_traj),
            u_scaled=np.stack(u_scaled_traj)
        )

        simulation_results = SimResults(
            time=time_vector,
            x_data=x_data,
            z_data=z_data,
            im_data=im_data,
            inputs_data=control,
        )
        return simulation_results

    @staticmethod
    def convert_rad_to_deg_np(x, idxs: list[int]) -> np.ndarray:
        x = np.asarray(x).copy()
        x[..., idxs] *= (180.0 / np.pi)
        return x

    def compute_total_time(self, start: float) -> None:
        """Compute and print execution time statistics."""
        end = time.time()
        time_control = end - start
        time_per_step = time_control / (self.n_steps_simulation - 1)
        print(f"\nTime in control: {time_control:.3f} seconds")
        print(f"Time per step: {time_per_step:.3f} seconds")