# pruning.py
from dataclasses import dataclass
from torch.nn.utils import prune
import torch

@dataclass(frozen=True)
class PruningConfig:
    enabled: bool
    start_epoch: int
    every_n_epochs: int
    threshold: float

    def should_prune(self, epoch: int) -> bool:
        if not self.enabled or epoch < self.start_epoch:
            return False
        return (epoch - self.start_epoch) % self.every_n_epochs == 0


class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"
    def __init__(self, threshold: float):
        self.threshold = threshold
    def compute_mask(self, tensor, default_mask):
        return (torch.abs(tensor) > self.threshold) & default_mask.bool()


def apply_global_threshold_pruning(model, threshold: float) -> None:
    prune.global_unstructured(
        parameters=model.parameters_to_prune,
        pruning_method=ThresholdPruning,
        threshold=threshold,
    )


def de_prune_state_dict(state_dict: dict) -> dict:
    sd = dict(state_dict)
    for k in list(sd.keys()):
        if k.endswith("weight_orig"):
            base = k[:-len("_orig")]
            mask_k = base + "_mask"
            if mask_k in sd:
                sd[base] = sd[k] * sd[mask_k]
                del sd[k]
                del sd[mask_k]
    return sd
