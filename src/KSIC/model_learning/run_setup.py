from logging import Logger
import random
import argparse

import numpy as np
import torch

from KSIC import utils

def prepare_training_from_scratch(
        modality: str,
        args: argparse.Namespace,
        logger: Logger,
) -> tuple[dict, utils.RunPaths]:

    stamp = utils.make_timestamped_dir(logger)

    if args.dynamics == "bilinear":
        stamp = "bilin_" + args.id + "_" + stamp
    elif args.dynamics == "linear":
        stamp = "lin_" + args.id + "_" + stamp
    else:
        raise ValueError("Problem in the dynamics name")

    paths = utils.build_run_paths(
        modality,
        args.drone_dim,
        "interim",
        stamp,
    )
    config = args.dynamics
    params = utils.load_base_configs(
        config,
        "learning",
        modality,
        args.drone_dim,
        args.geom_losses,
    )
    utils.make_unique_dir(paths.run_dir)
    utils.make_unique_dir(paths.log_dir)
    ckpt_dir = paths.run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return params, paths


def define_seeds(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False