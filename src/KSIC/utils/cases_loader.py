from __future__ import annotations
from pathlib import Path
import yaml
from dataclasses import dataclass

@dataclass(frozen=True)
class CaseConfig:
    stamp: str
    epoch: int
    run_status: str
    control_config: str
    geom_losses: bool
    drone_dim: int
    dynamics: str
    dt: float
    open_loop_simulations: dict
    closed_loop_simulations: dict

# Racine du repo = deux niveaux au-dessus de ce fichier
REPO_ROOT = Path(__file__).resolve().parents[3]
CASES_DIR = REPO_ROOT / "configs" / "control" / "cases"

def load_cases(modality: str):
    if modality == "vision":
        path = REPO_ROOT / "configs" / "vision_cases.yaml"
    elif modality == "sensor":
        path = REPO_ROOT / "configs" / "sensor_cases.yaml"
    raw = yaml.safe_load(path.read_text())["cases"]
    return {int(k): CaseConfig(**v) for k, v in raw.items()}
