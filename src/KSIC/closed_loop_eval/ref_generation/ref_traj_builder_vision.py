from __future__ import annotations

from .ref_traj_builder_base import ReferenceTrajBuilderBase
from .im_traj_gen import ImTrajGenerator
from .z_traj_gen import ZTrajGen


class ReferenceTrajBuilderVision(ReferenceTrajBuilderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        print("Generating reference trajectories (vision)...")
        state_ref_traj = self.build_state_ref()

        im_ref_traj = ImTrajGenerator(False, self.debug).pipeline(state_ref_traj)
        z_ref_traj = ZTrajGen(
            self.num_steps_simulation,
            self.koop_model,
            self.drone_dim,
        ).pipeline(im_ref_traj)

        return state_ref_traj, im_ref_traj, z_ref_traj
