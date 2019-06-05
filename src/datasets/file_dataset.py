from typing import Tuple

import numpy as np

from src.datasets.dataset import Dataset


class FileDataset(Dataset):
    file_path: str

    def __init__(self, file_path):
        # assume file type of CSV
        self.path = file_path

    def load_or_generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        data = np.genfromtxt(self.path, delimiter=',')
        # assume output dimension is 1
        x, y = data[:, :-1], data[:, -1]
        return x, y

    def __repr__(self):
        return f'{self.__class__.__name__}('f' file_path={self.path!r}) '
