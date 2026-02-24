#!/usr/bin/env python

import numpy as np
import torch

from KSIC_v6 import utils
from KSIC_v6.closed_loop_eval import (
    StateRenderer,
    compute_pose_features_denorm
)
from KSIC_v6.models import load_vision_koop_model_for_eval

STAMP = "2025-09-24_15-44-42"
EPOCH = 200
RUN_STATUS = "final"

# STAMP = "2025-09-26_14-17-50"
# EPOCH = 190
DIM = 2
NANE_CONFIG = "2d"


def pixel_to_world(u, v, W, H) -> tuple:
    x_min = y_min = -1
    x_max = y_max = 1
    x = x_min + (u / (W - 1)) * (x_max - x_min)
    y = y_min + (1.0 - v / (H - 1)) * (y_max - y_min)
    return x, y

def world_to_pixel(x, y, W, H) -> tuple:
    x_min = y_min = -1
    x_max = y_max = 1
    u = (x - x_min) / (x_max - x_min) * (W - 1)
    v = (1.0 - (y - y_min) / (y_max - y_min)) * (H - 1)
    return u, v

def main():
    logger = utils.setup_logging()
    stamp_control = utils.make_timestamped_dir(logger)

    # Config loading:
    paths = utils.ensure_training_run_paths("vision", STAMP, RUN_STATUS, logger)
    sys_params = utils.load_checkpoint_config(STAMP, RUN_STATUS, logger, DIM)
    sys_params = utils.process_checkpoint_config(sys_params, paths)
    koop_model = load_vision_koop_model_for_eval(sys_params, EPOCH)

    ctrl_runs_dir = utils.create_control_runs_dir("vision", stamp_control, "interim")
    ctrl_params = utils.load_base_configs(NANE_CONFIG, "control", logger)
    ctrl_params = utils.process_control_params(sys_params, ctrl_params, ctrl_runs_dir)

    utils.save_config_yaml(ctrl_params, ctrl_runs_dir, "control_params.yaml")
    model_params = sys_params["model_params"]
    pose_in_z = model_params["z_dynamics"]["include_pose_in_z"]
    dataset_params = sys_params["dataset_params"]


    state_renderer = StateRenderer(512, 128)
    device = next(koop_model.parameters()).device
    x_k = np.array([-0.5, -0.5, 0, 0, 0, 0])

    im_k = state_renderer.pipeline(x_k)
    im_k = im_k.astype(np.float32) / 255.0
    im_k = torch.from_numpy(im_k)  # [1, H, W]

    im_km1 = state_renderer.pipeline(x_k)
    im_km1 = im_km1.astype(np.float32) / 255.0
    im_km1 = torch.from_numpy(im_km1)  # [1, H, W]

    c_k, a_k = compute_pose_features_denorm(im_k.unsqueeze(0),device)
    x_k, y_k = pixel_to_world(c_k[0, 0], c_k[0, 1], 128, 128)

if __name__ == '__main__':
    main()
