from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass
class InputsData:
    """Données de contrôle."""
    u_physical: np.ndarray
    u_scaled: np.ndarray

@dataclass
class TrajectoryData:
    """Données pour une trajectoire avec référence et erreur."""
    traj: np.ndarray
    ref_traj: np.ndarray
    error: Optional[np.ndarray]

@dataclass
class SimResults:
    time: np.ndarray
    x_data: Optional[TrajectoryData]
    z_data: TrajectoryData
    im_data: TrajectoryData
    inputs_data: InputsData
