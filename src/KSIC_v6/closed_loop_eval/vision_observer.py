import cv2
import numpy as np
import torch

from .state_renderer import StateRenderer

class VisionObserver:
    def __init__(self, state_renderer: StateRenderer):
        self.state_renderer = state_renderer

    def observe(self, x_k: np.ndarray, debug: bool = False) -> torch.Tensor:
        im_k = self.state_renderer.pipeline(np.asarray(x_k))
        if debug:
            cv2.imshow("Rendered state", im_k)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord('q')):
                cv2.destroyAllWindows()
                raise KeyboardInterrupt
        im_k = im_k.astype(np.float32) / 255.0
        return torch.from_numpy(im_k)
