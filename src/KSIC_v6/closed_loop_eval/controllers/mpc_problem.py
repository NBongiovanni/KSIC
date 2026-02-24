# + ajoute un MPCProblem simple
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np


@dataclass
class MPCProblem:
    dt: float
    N: int
    z_dim: int
    u_dim: int
    Q: np.ndarray
    Qf: np.ndarray
    R: np.ndarray
    S: np.ndarray = None
    use_inputs_constraints: bool = False
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None
    f_discrete: Optional[object] = None  # ca.Function(z,u)->z+
    tvp_provider: Optional[Callable[[int], np.ndarray]] = None
    u_guess: Optional[np.ndarray] = None
