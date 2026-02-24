import cv2
import numpy as np
import torch
from tqdm import tqdm

from KSIC_v6.closed_loop_eval.state_renderer import StateRenderer


class ImTrajGenerator:
    def __init__(self, old: bool, debug: bool = False):
        self.debug = debug
        self.state_renderer = StateRenderer(512)

    def pipeline(self, x_ref_traj) -> torch.Tensor:
        im_traj = self.render_traj(x_ref_traj)
        return im_traj

    def render_traj(self, x_traj: np.ndarray) -> torch.Tensor:
        im_traj = []

        for i in tqdm(range(x_traj.shape[0])):
            x_k = x_traj[i]
            im_k = self.state_renderer.pipeline(x_k)

            if self.debug:
                cv2.imshow("Reference image trajectory", im_k)
                key = cv2.waitKey(0)  & 0xFF

                # ESC ou 'q' pour quitter
                if key in (27, ord('q')):
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt("Arrêt demandé par l'utilisateur")

            im_k = im_k.astype(np.float32) / 255.0
            im_k = torch.from_numpy(im_k)  # [1, H, W]
            im_traj.append(im_k)
        return torch.stack(im_traj)