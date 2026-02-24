#!/usr/bin/env python
import joblib

import matplotlib
matplotlib.use("Agg")

from KSIC_v6 import utils
from KSIC_v6.model_learning import (
    VisionTrainer,
    prepare_training_from_scratch,
    define_seeds,
)
from KSIC_v6.models import init_vision_koop_model
from KSIC_v6.data_pipeline import ImageDatasetBuilder, StateInputsDatasetBuilder

MODALITY = "vision"


def main() -> None:
    """
    Training of a Koopman model with a visual dataset.
    """
    args = utils.build_arg_parser_vision().parse_args()
    logger = utils.setup_logging()
    define_seeds(args.seed)

    params, paths = prepare_training_from_scratch(MODALITY,args, logger)
    params = utils.process_checkpoint_config(params, paths, args.seed)
    utils.save_config_yaml(params)

    training_params = params["training_params"]
    model_params = params["model_params"]
    koop_model, (optimizer, scheduler, writer) = init_vision_koop_model(
        model_params,
        training_params,
    )
    # Data loading
    dataset_params = params["dataset_params"]
    state_inputs_dataset_builder = StateInputsDatasetBuilder(
        dataset_params,
        args.drone_dim
    )
    processed_states_inputs = state_inputs_dataset_builder.processed

    for phase, d in processed_states_inputs.items():
        n_sensor = _n_traj(d["x"])
        print(phase, n_sensor)
    u_scaler = state_inputs_dataset_builder.u_scaler
    joblib.dump(u_scaler, paths.run_dir / "u_scaler.pkl")

    im_dataset_builder = ImageDatasetBuilder(
        dataset_params["dataset_version"],
        2,
        128,
        dataset_params["batch_size"],
        processed_states_inputs,
        dataset_params["num_workers"],
        args.drone_dim,
        args.seed,
        True
    )
    im_dataset_loader = im_dataset_builder.pipeline()

    trainer = VisionTrainer(
        params["training_params"],
        koop_model,
        im_dataset_loader,
        dataset_params["train"]["num_steps_pred"],
        dataset_params["val_datasets"][0]["num_steps_pred"],
        dataset_params["val_datasets"][1]["num_steps_pred"],
        optimizer,
        scheduler,
        writer,
        params["model_params"]["drone"]["num_views"]
    )
    koop_model.train()

    trainer.train_model()

def _n_traj(arr):
    # arr typiquement shape (N, T, dim) ou liste de trajs
    return len(arr)


if __name__ == '__main__':
    main()