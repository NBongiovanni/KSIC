from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from .state_traj_generator import StateTrajGenerator
from KSIC.models import BaseKoopModel


class ReferenceTrajBuilderBase(ABC):
    def __init__(
        self,
        control_params: dict,
        koop_model: BaseKoopModel,
        specs: dict,
        drone_dim: int,
        debug: bool = False,
    ):
        self.koop_model = koop_model
        self.specs = specs
        self.drone_dim = drone_dim
        self.debug = debug
        self.num_steps_simulation = int(control_params["num_steps_simulation"])
        self.dt = float(control_params["dt"])

    def build_state_ref(self) -> np.ndarray:
        for axis_idx in [0, 1]:
            if self.specs[axis_idx]["type"] == "constant":
                spec = self.specs[axis_idx]
                if spec["rand"]:
                    new_value = np.random.uniform(spec["min"], spec["max"])
                    self.specs[axis_idx]["value"] = new_value
                    print(f"[INFO] Axis {axis_idx} setpoint: {new_value:.3f}")

        gen = StateTrajGenerator(
            self.drone_dim,
            self.num_steps_simulation+1,
            self.specs,
            self.dt
        )
        return gen.pipeline()

    @abstractmethod
    def build(self):
        raise NotImplementedError
