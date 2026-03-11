from .processor import Processor
from .loader import Loader

class Builder:
    def __init__(self, dataset_params: dict, drone_dim: int):
        self.loader = Loader(
            dataset_params["dataset_version"],
            drone_dim,
            dataset_params["train"],
            dataset_params["val_datasets"][0],
            dataset_params["val_datasets"][1],
            dataset_params["decimation_factor"],
            True
            )
        raw_data = self.loader.load_raw_sensor_data()
        batch_size = dataset_params["batch_size"]

        self.processor = Processor(
            batch_size,
            dataset_params["train"],
            dataset_params["val_datasets"][0],
            dataset_params["val_datasets"][1],
            raw_data,
            dataset_params["scaler"],
            dataset_params["delay"],
        )
        self.processor.generate_u_scaler()
        self.processor.generate_x_scaler()

        self.processor.process_datasets()
        self.u_scaler = self.processor.u_scaler
        self.x_scaler = self.processor.x_scaler

        self.processor.build_data_loader()
        self.data_loader = self.processor.data_loader

    @property
    def processed(self):
        return self.processor.processed_datasets