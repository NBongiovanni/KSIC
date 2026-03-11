import numpy as np

from KSIC.closed_loop_eval.simulation.results import SimResults
from .simulator_base import ControlSimulator

class RealControlSimulator(ControlSimulator):
    """
    Simulator using the real physical plant (PlanarQuad).

    The controller uses the learned Koopman model while the plant dynamics
    are the true physical equations. This allows testing robustness to
    model mismatch.

    State space: x (physical states: position, velocity, angle, etc.)
    """
    def run(self, x_init: np.ndarray) -> SimResults:
        self.controller.set_initial_conditions(x_init)
        x_traj, im_traj = self._run_simulation_loop(x_init)

        x_ref_traj = self.controller.state_ref_traj
        z_traj = self.controller.z_traj  # liste de (D,)

        z_ref_traj = self.controller.z_ref_traj  # (T, D) déjà normalisé
        im_ref_traj = self.controller.im_ref_traj

        processed_outputs = self.process_output(
            x_traj,
            x_ref_traj,
            z_traj,
            z_ref_traj,
            im_traj,
            im_ref_traj,
        )
        return processed_outputs
