#!/usr/bin/env python
import joblib

from KSIC_v6 import utils
from KSIC_v6.model_learning import (
    SensorTrainer, prepare_training_from_scratch, define_seeds
)
from KSIC_v6.models import init_sensor_koop_model
from KSIC_v6.data_pipeline import StateInputsDatasetBuilder

MODALITY = "sensor"

def main() -> None:
    args = utils.build_arg_parser_sensors().parse_args()
    logger = utils.setup_logging()
    define_seeds(args.seed)

    params, paths = prepare_training_from_scratch(MODALITY,args, logger)
    params = utils.process_checkpoint_config(params, paths, args.seed)
    utils.save_config_yaml(params)

    training_params = params["training_params"]
    model_params = params["model_params"]
    dataset_params = params["dataset_params"]

    koop_model, (optimizer, scheduler, writer) = init_sensor_koop_model(
        model_params,
        training_params,
    )
    # Data loading
    params = utils.process_checkpoint_config(params, paths, args.seed)
    utils.save_config_yaml(params)

    state_inputs_dataset_builder = StateInputsDatasetBuilder(
        dataset_params,
        args.drone_dim
    )
    data_loader = state_inputs_dataset_builder.data_loader
    x_scaler = state_inputs_dataset_builder.x_scaler
    u_scaler = state_inputs_dataset_builder.u_scaler
    joblib.dump(u_scaler, paths.run_dir / "u_scaler.pkl")
    joblib.dump(x_scaler, paths.run_dir / "x_scaler.pkl")

    trainer = SensorTrainer(
        training_params,
        koop_model,
        data_loader,
        dataset_params["train"]["num_steps_pred"],
        dataset_params["val_datasets"][0]["num_steps_pred"],
        dataset_params["val_datasets"][1]["num_steps_pred"],
        optimizer,
        scheduler,
        writer,
        paths.run_dir
    )
    trainer.train_model()  # Training


if __name__ == '__main__':
    main()