import torch
import numpy as np


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().squeeze(0).numpy()


def load_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch device: {}".format(device))
    return device