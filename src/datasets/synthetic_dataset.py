from abc import ABC

from src.datasets.dataset import Dataset


class SyntheticDataset(Dataset, ABC):
    n_samples: int
    input_dim: int

    def __init__(self, n_samples, input_dim):
        self.n_samples = n_samples
        self.input_dim = input_dim

    def __repr__(self):
        return f'{self.__class__.__name__}('f' n={self.n_samples!r}, d={self.input_dim!r})'


class Input1DSyntheticDataset(SyntheticDataset, ABC):

    def __init__(self, n_samples, input_dim=1):
        super().__init__(n_samples, input_dim)
        if self.input_dim != 1:
            raise ValueError('Input dimension must be 1')

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n={self.n_samples!r}, d={self.input_dim!r})'
