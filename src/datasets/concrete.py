import os
from typing import Optional, Tuple

import numpy as np
import toml

from src.datasets.dataset import Dataset, _download_raw_dataset, _parse_args

RAW_DATA_DIRNAME = Dataset.data_dirname() / 'raw' / 'concrete'
METADATA_FILENAME = RAW_DATA_DIRNAME / 'metadata.toml'

PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / 'processed' / 'concrete'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'processed_data.npz'


class ConcreteDataset(Dataset):
    """Concrete Compressive Strength Dataset

    I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and
    Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998).
    http://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength
    """

    def __init__(self, subsample_fraction: Optional[float] = 0.486):
        super().__init__()
        self.subsample_fraction = subsample_fraction  # by default, take the first 500 samples

    def load_or_generate_data(self) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_concrete()

        with PROCESSED_DATA_FILENAME.open('rb') as f:
            data = np.load(f)

            self.x = data['x']
            self.y = data['y']

        self._subsample()

    def _subsample(self) -> None:
        """Only this fraction of data will be loaded."""
        if self.subsample_fraction is None:
            return

        num_subsample = int(self.x.shape[0] * self.subsample_fraction)
        self.x = self.x[:num_subsample]
        self.y = self.y[:num_subsample]


def _download_concrete() -> None:
    metadata = toml.load(METADATA_FILENAME)
    curdir = os.getcwd()
    os.chdir(RAW_DATA_DIRNAME)
    _download_raw_dataset(metadata)
    _process_raw_dataset(metadata['filename'])
    os.chdir(curdir)


def _process_raw_dataset(filename: str) -> None:
    file_type = filename.split('.')[-1]
    print('Loading training data from .%s file' % file_type)
    x, y = _load_x_y(filename)

    print('Saving to NPZ...')
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)

    np.savez(PROCESSED_DATA_FILENAME, x=x, y=y)


def _load_x_y(data_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load input and outputs"""
    import pandas as pd
    df = pd.read_excel(data_filename)
    x, y = df.values[:, :-1], df.values[:, -1:]
    return x, y


def main():
    """Load dataset and print info."""
    args = _parse_args()
    dataset = ConcreteDataset(subsample_fraction=args.subsample_fraction)
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == '__main__':
    main()
