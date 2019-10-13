from typing import Optional
import numpy as np


class Sampler:

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def sample(self, n_points: int, n_dims: int) -> np.ndarray:
        raise NotImplementedError
