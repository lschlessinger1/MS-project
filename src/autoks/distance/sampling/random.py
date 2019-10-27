import numpy as np

from src.autoks.distance.sampling.sampler import Sampler


class RandomSampler(Sampler):
    def sample(self, n_points: int, n_dims: int) -> np.ndarray:
        return np.random.random((n_points, n_dims))
