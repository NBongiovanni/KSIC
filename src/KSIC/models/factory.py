from __future__ import annotations
from typing import TypeAlias

import torch
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler

from KSIC import utils
from KSIC.models.nn.auto_encoder import AutoEncoder
from .vision_koop_model import VisionKoopModel
from .sensor_koop_model import SensorKoopModel

TrainingComponents: TypeAlias = tuple[Optimizer, StepLR, SummaryWriter]
VisionModelAndTC: TypeAlias = tuple[VisionKoopModel, TrainingComponents]
SensorModelAndTC: TypeAlias = tuple[SensorKoopModel, TrainingComponents]
VisionModelAndScaler: TypeAlias = tuple[VisionKoopModel, StandardScaler]
SensorModelAndScaler: TypeAlias = tuple[SensorKoopModel, StandardScaler, StandardScaler]

def init_vision_koop_model(
        model_params: dict,
        training_params: dict
) -> VisionModelAndTC:
    """
    Initialize a fresh Koopman model and its training components.

    Args:
        model_params (dict)
        training_params (dict)

    Returns:
        (VisionKoopModel, (Optimizer, _LRScheduler, SummaryWriter)):
            The model on the selected device and its optimizer, scheduler,
            and TensorBoard writer.

    Notes:
        - Calls `_build_model`, creates an AdamW optimizer with named param-groups,
          a StepLR scheduler (epoch-wise), and a SummaryWriter at `log_dir`.
        - Prints an `auto_encoder` summary via `torchinfo.summary`.
    """
    model, device = _build_vision_model(model_params)
    optimizer = _make_optimizer_vision(training_params["optimizer"], model)
    scheduler = StepLR(optimizer, step_size=1, gamma=training_params["lr_decay"])
    writer = SummaryWriter(log_dir=str(training_params["log_dir"]))
    model.train()
    return model, (optimizer, scheduler, writer)


def init_sensor_koop_model(
        model_params: dict,
        training_params: dict
) -> SensorModelAndTC:
    model, device = _build_sensor_model(model_params)
    writer = SummaryWriter(log_dir=str(training_params["log_dir"]))

    model_params = model.parameters()
    l2_reg = training_params["optimizer"]["l2_reg"]
    lr = training_params["optimizer"]["lr"]
    lr_decay = training_params["lr_decay"]
    optimizer = Adam(params=model_params, lr=lr, weight_decay=l2_reg)
    scheduler = StepLR(optimizer, step_size=1, gamma=lr_decay)
    return model, (optimizer, scheduler, writer)


def _build_vision_model(
        model_params: dict,
        device: torch.device | None = None
) -> tuple[VisionKoopModel, torch.device]:
    """
    Build a fresh VisionKoopModel (architecture only), moved to `device`.

    This is meant to be used by BOTH:
      - init_vision_koop_model (fresh training)
      - load_vision_koop_model_for_train/eval (restore from checkpoint)

    Returns:
        (model, device)
    """
    z_dim = model_params["z_dynamics"]["z_dim"]
    drone_dim = model_params["drone"]["dim"]

    device = utils.load_device() if device is None else device

    auto_encoder = AutoEncoder(
        model_params["auto_encoder"],
        z_dim,
        drone_dim
    ).to(device)

    model = VisionKoopModel(model_params, auto_encoder).to(device)
    return model, device


def _build_sensor_model(
        params: dict,
        device: torch.device | None = None
) -> tuple[SensorKoopModel, torch.device]:
    """
    Build a fresh SensorKoopModel (architecture only), moved to `device`.

    This is meant to be used by BOTH:
      - init_sensor_koop_model
      - load_sensor_koop_model_for_eval (and possible future load_for_train)

    Returns:
        (model, device)
    """
    device = utils.load_device() if device is None else device

    model = SensorKoopModel(params).to(device)
    return model, device


def _get_dynamics_parameters(model: VisionKoopModel):
    """
    Rétrocompatible:
    - nouveau: model.get_dynamics_parameters()
    - ancien: model.z_drift + model.z_act
    """
    if hasattr(model, "get_dynamics_parameters"):
        return list(model.get_dynamics_parameters())
    # fallback ancien comportement
    return list(model.z_drift.parameters()) + list(model.z_act.parameters())


def _make_optimizer_vision(hparams: dict, model: VisionKoopModel) -> Optimizer:
    lr_ae = hparams["lr"]["ae"]
    lr_ab = hparams["lr"]["ab"]
    wd_ae = hparams["weight_decay"]["ae"]
    wd_ab = hparams["weight_decay"]["ab"]

    a_e = model.auto_encoder
    params_ae_cnn = list(a_e.encoder_cnn.parameters()) + list(a_e.decoder_cnn.parameters())
    params_ae_mlp = list(a_e.encoder_mlp.parameters()) + list(a_e.decoder_mlp.parameters())

    params_ab = _get_dynamics_parameters(model)

    return torch.optim.Adam([
        {"params": params_ae_cnn, "lr": lr_ae, "weight_decay": wd_ae, "name": "ae"},
        {"params": params_ae_mlp, "lr": lr_ae, "weight_decay": wd_ae, "name": "mlp"},
        {"params": params_ab, "lr": lr_ab, "weight_decay": wd_ab, "name": "ab"},
    ])