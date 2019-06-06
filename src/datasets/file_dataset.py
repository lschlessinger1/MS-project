import numpy as np

from src.datasets.dataset import Dataset


class FileDataset(Dataset):
    file_path: str

    def __init__(self, file_path):
        # assume file type of CSV
        super().__init__()
        self.path = file_path

    def load_or_generate_data(self) -> None:
        data = np.genfromtxt(self.path, delimiter=',')
        # assume output dimension is 1
        self.x, self.y = data[:, :-1], data[:, -1]

    def __repr__(self):
        return f'{self.__class__.__name__}('f' file_path={self.path!r}) '
