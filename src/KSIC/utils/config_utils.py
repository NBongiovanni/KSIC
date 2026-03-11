import argparse
import copy
from logging import Logger
import os
from pathlib import Path

import numpy as np
import yaml

from .path_utils import find_project_root, RunPaths


def get_dimensions(drone_dim: int) -> tuple[int, int, int]:
    if drone_dim == 1:
        x_dim = 2
        u_dim = 1
        x_ref_dim = 1
    elif drone_dim == 2:
        x_dim = 6
        u_dim = 2
        x_ref_dim = 2
    elif drone_dim == 3:
        x_dim = 12
        u_dim = 4
        x_ref_dim = 3
    else:
        raise ValueError(f"Drone dimension {drone_dim} not supported.")
    return x_dim, u_dim, x_ref_dim


def build_arg_parser_vision() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/Resume Koopman model (vision)")
    help_resume = "Resume from a snapshot (otherwise: train from scratch)."
    p.add_argument("--mode", type=str)
    p.add_argument("--id", type=str)
    p.add_argument("--resume", action="store_true", help=help_resume)
    p.add_argument("--resume_stamp", type=str)
    p.add_argument("--resume_epoch", type=int)
    p.add_argument("--drone_dim", type=int)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--geom_losses",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable geometric losses (--no-geom_losses if we want to disable this option)."
    )
    p.add_argument("--dynamics", type=str)
    return p


def build_arg_parser_sensors() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/Resume Koopman model (sensors)")
    p.add_argument("--mode", type=str)
    p.add_argument("--id", type=str)
    p.add_argument("--drone_dim", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dynamics", type=str)
    p.add_argument(
        "--geom_losses",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable geometric losses (--no-geom_losses if we want to disable this option)."
    )
    return p

def build_arg_parser_data_generation() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/Resume Koopman model (sensors)")
    p.add_argument("--drone_dim", type=int)
    return p


def process_checkpoint_config(params: dict, paths: RunPaths, seed: int) -> dict:
    dataset_params = params["dataset_params"]
    training_params = params["training_params"]
    model_params = params["model_params"]

    x_dim, u_dim, x_ref_dim = get_dimensions(model_params["drone"]["dim"])
    if model_params["drone"]["dim"] == 2:
        num_views = 1
    elif model_params["drone"]["dim"] == 3:
        num_views = 2
    else:
        raise ValueError(f"Invalid drone dimension {model_params['drone']['dim']}")

    model_params["drone"]["num_views"] = num_views
    model_params["z_dynamics"]["x_dim"] = x_dim
    model_params["z_dynamics"]["u_dim"] = u_dim
    training_params["dt"] = model_params["dt"]
    dataset_params["dt"] = model_params["dt"]

    training_params["run_dir"] = paths.run_dir
    training_params["log_dir"] = paths.log_dir
    training_params["checkpoints_dir"] = paths.run_dir / "checkpoints"
    training_params["seed"] = seed
    dataset_params["dim"] = model_params["drone"]["dim"]
    return params


def save_config_yaml(
        params: dict, path: Path | None = None, name="config.yaml"
) -> None:
    """
    Save config dict as a YAML file.

    Parameters
    ----------
    params : dict
        Configuration dictionary.
    path : Path | None, optional
        Directory where the config file will be saved.
        If None, uses params["training_params"]["run_dir"].
    name : str, optional
        Name of the YAML file (default: "config.yaml").
    """
    if path is None:
        save_path = Path(params["training_params"]["run_dir"])
    else:
        save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / name, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            make_serializable(params),
            f,
            sort_keys=False,
            indent=2,
            allow_unicode=True
        )


def process_control_params(
        sys_params: dict,
        control_params: dict,
        epoch: int,
        runs_dir: RunPaths
) -> dict:
    control_params["run_dir"] =  runs_dir.closed_loop_eval_dir
    control_params["control_runs_dir"] = runs_dir.closed_loop_eval_dir
    model_params = sys_params["model_params"]
    control_params["z_dim"] = model_params["z_dynamics"]["z_dim"]
    control_params["z_dynamics_model"] = model_params["z_dynamics"]["model"]
    control_params["epoch"] = epoch
    return control_params


def prepare_params_from_checkpoint(
        args: argparse.Namespace,
        logger: Logger,
        drone_dim: int,
        child_stamp: str,
        jean_zay: bool
) -> tuple[str, dict, str, dict]:

    logger.info(f"resume :{args.resume_stamp}")
    parent_stamp = args.resume_stamp

    base_configs_dir = Path("outputs") / "interim" / "learning" / "vision"
    if not jean_zay:
        root = find_project_root()
    else:
        root = Path(os.environ["SCRATCH"])
    path_config = root / base_configs_dir / f"{drone_dim}d" / "models" / parent_stamp / "config.yaml"

    with open(path_config, "r", encoding="utf-8") as f:
        parent_params = yaml.safe_load(f)

    child_stamp = f"{child_stamp}_from_{parent_stamp}"
    child_params = copy.deepcopy(parent_params)
    child_params["training_params"]["parent_stamp"] = str(parent_stamp)
    child_params["training_params"]["parent_epoch"] = args.resume_epoch
    return child_stamp, child_params, parent_stamp, parent_params


def load_base_configs(
        config: str,
        task: str,
        modality: str,
        drone_dim: int,
        geom_losses: bool = False,
) -> dict:
    base_configs_dir = Path("configs") / task / modality
    root = find_project_root()
    geom_losses_dir = "geom_losses" if geom_losses else "no_geom_losses"

    if modality == "vision":
        config_subdir = Path(f"{drone_dim}d") / geom_losses_dir / f"config_{config}.yaml"
    else:
        config_subdir = Path(f"{drone_dim}d") / f"config_{config}.yaml"
    path_config = root / base_configs_dir / config_subdir
    with open(path_config, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params


def load_checkpoint_config(path: RunPaths) -> dict:
    path_config = path.run_dir / "config.yaml"
    with open(path_config, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params


def make_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj