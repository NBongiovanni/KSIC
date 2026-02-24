from pathlib import Path
import os

import numpy as np

from KSIC_v6.data_pipeline.vision_data.quad_drawer.quad_drawer_2d import QuadDrawer2D

class StateRenderer(QuadDrawer2D):
    def __init__(self, img_size: int):
        super().__init__(img_size, 128)

    def pipeline(self, state: np.ndarray) -> np.ndarray:
        save_dir = Path("results")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        name_fig = Path("raw_images.png")
        img = self._gen_raw_image(state, save_dir / name_fig, True, False)
        return img
