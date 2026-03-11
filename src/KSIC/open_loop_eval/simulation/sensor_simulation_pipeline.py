from __future__ import annotations
import random
from typing import Any

import numpy as np
import torch

from KSIC import utils
from KSIC.data_pipeline import StateInputsDatasetBuilder
from KSIC.models import load_sensor_koop_model_for_eval
from KSIC.models.outputs.sensor_outputs import SensorValForwardOutputs

# TODO: handle the 2d case
ANGLE_INDEXES = [3, 4, 5]
ANGULAR_RATE_INDEXES = [9, 10, 11]
RAD2DEG_INDEXES = ANGLE_INDEXES + ANGULAR_RATE_INDEXES

def open_loop_simulation_sensor_pipeline(
        case: utils.CaseConfig,
        phase: str,
        num_steps: int,
        modality: str,
        drone_dim: int,
        stamp_open_loop: str,
        seed: int,
):
    """Perform open-loop forward simulation and return everything needed for visualization."""
    random.seed(seed)
    np.random.seed(seed)

    paths = utils.build_run_paths(
        modality,
        drone_dim,
        case.run_status,
        case.stamp,
        stamp_open_loop
    )
    params = utils.load_checkpoint_config(paths)
    params = utils.process_checkpoint_config(params, paths, seed)

    ckpt_dir = paths.run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = utils.load_device()

    model_params = params["model_params"]
    training_params = params["training_params"]
    dataset_params = params["dataset_params"]
    training_params["results_dir"] = paths.run_dir
    training_params["log_dir"] = paths.log_dir

    state_inputs_dataset_builder = StateInputsDatasetBuilder(
        dataset_params,
        drone_dim
    )
    data_loader = state_inputs_dataset_builder.data_loader

    koop_model, x_scaler, u_scaler = load_sensor_koop_model_for_eval(
        model_params,
        case.epoch,
        paths.run_dir,
    )

    A, B = koop_model.construct_koop_matrices()
    print(f"A:\n{A}")
    print(f"B:\n{B}")

    with torch.no_grad():
        batch = get_one_batch(data_loader[phase])
        x_gt_scaled, u_traj = batch

        rec, pred = koop_model.forward(
            x_gt_scaled[:, 0].to(device),
            u_traj.to(device),
            num_steps,
        )
        rec_physical = unscale_data(rec.detach().cpu().numpy(), x_scaler)
        x_gt_physical = unscale_data(x_gt_scaled.detach().cpu().numpy(), x_scaler)
        pred_x_physical = unscale_data(pred.state.detach().cpu().numpy(), x_scaler)
        u_traj_physical = torch.from_numpy(unscale_data(u_traj.detach().cpu().numpy(), u_scaler)).float()

        rec_deg = torch.from_numpy(convert_rad_to_deg_np(rec_physical, RAD2DEG_INDEXES)).float()
        pred.state = torch.from_numpy(convert_rad_to_deg_np(pred_x_physical, RAD2DEG_INDEXES)).float()
        x_gt_physical_deg = torch.from_numpy(convert_rad_to_deg_np(x_gt_physical, RAD2DEG_INDEXES)).float()
        z_proj = koop_model.batch_projection(x_gt_scaled.to(device))

        models_outputs = SensorValForwardOutputs(
            rec=rec_deg,
            pred=pred,
            proj=z_proj,
            state_gt_scaled=x_gt_scaled,
            state_gt_physical=x_gt_physical_deg,
            inputs_scaled=u_traj,
            inputs_physical=u_traj_physical
        )

    simu_output = {
        "models_outputs": models_outputs,
        "x_scaler": x_scaler,
        "u_scaler": u_scaler,
        "run_dir": paths.run_dir,
        "open_loop_eval_dir": paths.open_loop_eval_dir,
    }
    return simu_output

def get_one_batch(dataloader: torch.utils.data.DataLoader) -> Any:
    it = iter(dataloader)
    try:
        return next(it)
    except StopIteration as e:
        # Diagnostic utile
        ds_len = len(dataloader.dataset) if hasattr(dataloader, "dataset") else "unknown"
        bs = getattr(dataloader, "batch_size", "unknown")
        drop_last = getattr(dataloader, "drop_last", "unknown")
        raise RuntimeError(
            f"DataLoader is empty: len(dataset)={ds_len}, batch_size={bs}, drop_last={drop_last}. "
            f"Likely causes: wrong phase key, dataset split empty, or drop_last=True with dataset < batch_size."
        ) from e


def scale_data(data, scaler) -> np.ndarray:
    if scaler is None:
        return np.asarray(data)

    x = np.asarray(data)
    if x.ndim == 1:
        return scaler.transform(x.reshape(1, -1)).reshape(-1)
    if x.ndim == 2:
        return scaler.transform(x)
    if x.ndim == 3:
        b, t, f = x.shape
        x2 = x.reshape(-1, f)
        y2 = scaler.transform(x2)
        return y2.reshape(b, t, f)
    raise ValueError(f"scale_data: expected 1D/2D/3D, got shape={x.shape}")

def unscale_data(data, scaler) -> np.ndarray:
    if scaler is None:
        return np.asarray(data)

    x = np.asarray(data)
    if x.ndim == 1:
        return scaler.inverse_transform(x.reshape(1, -1)).reshape(-1)
    if x.ndim == 2:
        return scaler.inverse_transform(x)
    if x.ndim == 3:
        b, t, f = x.shape
        x2 = x.reshape(-1, f)
        y2 = scaler.inverse_transform(x2)
        return y2.reshape(b, t, f)
    raise ValueError(f"unscale_data: expected 1D/2D/3D, got shape={x.shape}")

def convert_rad_to_deg_np(x, idxs: list[int]) -> np.ndarray:
    x = np.asarray(x).copy()
    x[..., idxs] *= (180.0 / np.pi)
    return x
