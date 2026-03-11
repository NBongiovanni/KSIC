#!/usr/bin/env python
from logging import Logger
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from KSIC import utils
from KSIC.models import (
    load_koop_model_for_eval,
    VisionKoopModel,
)
from .controllers import AcadosBackend, SolverBackend
from .simulation import RealControlSimulator, NominalControlSimulator
from .plants import LearnedModel, Quad3D, PlanarQuad

@dataclass
class ControlEnvironment:
    koop_model: VisionKoopModel
    solver_backend: SolverBackend
    x_scaler: StandardScaler
    u_scaler: StandardScaler
    model_params: dict
    control_params: dict
    num_simulations: int

def prepare_for_closed_loop_eval(
        modality: str,
        logger: Logger,
        run_status: str,
        stamp_run: str,
        drone_dim: int,
        name_config: str,
        epoch: int,
        geom_losses: bool,
        seed: int,
) -> ControlEnvironment:

    stamp_control = utils.make_timestamped_dir(logger)

    paths = utils.build_run_paths(
        modality,
        drone_dim,
        run_status,
        stamp_run,
        None,
        stamp_control,
    )
    paths.closed_loop_eval_dir.mkdir(parents=True, exist_ok=True)
    sys_params = utils.load_checkpoint_config(paths)
    sys_params = utils.process_checkpoint_config(sys_params, paths, seed)
    model_params = sys_params["model_params"]

    koop_model, x_scaler, u_scaler = load_koop_model_for_eval(
        modality,
        model_params,
        epoch,
        paths.run_dir,
    )

    control_params = utils.load_base_configs(
        name_config,
        "control",
        modality,
        drone_dim,
        geom_losses
    )
    control_params = utils.process_control_params(
        sys_params,
        control_params,
        epoch,
        paths,
    )
    utils.save_config_yaml(
        control_params,
        paths.closed_loop_eval_dir,
        "control_params.yaml",
    )
    solver_backend = AcadosBackend(control_params)

    return ControlEnvironment(
        koop_model=koop_model,
        solver_backend=solver_backend,
        x_scaler=x_scaler,
        u_scaler=u_scaler,
        model_params=sys_params["model_params"],
        control_params=control_params,
        num_simulations=control_params["num_simulations"]
    )


def create_state_init_conditions(x_dim: int, control_params: dict) -> np.ndarray:
    x_init = np.zeros(x_dim, dtype=float)
    for j in range(x_dim):
        if control_params["x_init"][j]["rand"]:
            init_min = control_params["x_init"][j]["min"]
            init_max = control_params["x_init"][j]["max"]
            x_init[j] = np.random.uniform(init_min, init_max)
        else:
            x_init[j] = control_params["x_init"][j]["value"]
    print(f"x_init: {x_init}")
    return x_init


def create_plant(
        drone_dim: int,
        use_nominal_plant: bool,
        dt: float,
        koop_model: VisionKoopModel,
):
    """Crée la plante (nominale ou réelle)."""
    if use_nominal_plant:
        return LearnedModel(dt, koop_model)
    else:
        if drone_dim == 2:
            return PlanarQuad(dt)
        elif drone_dim == 3:
            return Quad3D(dt)
        else:
            raise ValueError(f"Drone dimension inconnue: {drone_dim}")


def create_simulator(control_params, plant, controller):
    if control_params["use_nominal_plant"]:
        return NominalControlSimulator(control_params, plant, controller)
    else:
        return RealControlSimulator(control_params, plant, controller)

