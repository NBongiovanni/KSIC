#!/usr/bin/env python
import matplotlib
matplotlib.use("Agg")

from KSIC import utils
from KSIC.data_pipeline import (
    ImDatasetGenerator,
    StateInputsDatasetLoader,
    ImageProcessorMmap,
    DatasetParams
)
from KSIC.utils import find_project_root

def main() -> None:
    args = utils.build_arg_parser_data_generation().parse_args()
    root_dir = find_project_root()
    path_config = root_dir / "configs" / f"dataset_generation_{args.drone_dim}d.yaml"
    dataset_params = DatasetParams.from_yaml(path_config)
    im_data_dir = root_dir / "datasets" / "images_processed" / f"{args.drone_dim}d"
    dataset_params.image_dataset_dir = im_data_dir

    # State-inputs dataset processing:
    data_loader = StateInputsDatasetLoader(
        dataset_params.dataset_version,
        dataset_params.drone_dim,
        dataset_params.train,
        dataset_params.val_datasets[0],
        dataset_params.val_datasets[1],
        dataset_params.decimation_factor,
    )
    raw_data = data_loader.load_raw_sensor_data()

    phase_list = ["train", "val_1", "val_2"]
    num_trajs = [
        dataset_params.train["num_trajs"],
        dataset_params.val_datasets[0]["num_trajs"],
        dataset_params.val_datasets[1]["num_trajs"]
    ]
    num_steps_pred = [
        dataset_params.train["num_steps_pred"],
        dataset_params.val_datasets[0]["num_steps_pred"],
        dataset_params.val_datasets[1]["num_steps_pred"]
    ]

    for i in range(3):
        phase = phase_list[i]
        im_generator = ImDatasetGenerator(
            dataset_params,
            phase,
            raw_data[phase],
            num_trajs[i]
        )
        im_generator.generate_raw_images()
        im_preprocessor = ImageProcessorMmap(dataset_params, phase)
        im_preprocessor.pipeline(
            num_trajs[i],
            dataset_params.resolution,
            num_steps_pred[i]
        )


if __name__ == '__main__':
    main()