import os
from typing import Optional, Tuple

import numpy as np
import toml
from scipy.io import loadmat

from src.datasets.dataset import Dataset, _parse_args, _download_raw_dataset

RAW_DATA_DIRNAME = Dataset.data_dirname() / 'raw' / 'mauna'
METADATA_FILENAME = RAW_DATA_DIRNAME / 'metadata.toml'

PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / 'processed' / 'mauna'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'processed_data.npz'


class MaunaDataset(Dataset):
    """Mauna Loa Observatory CO2 dataset.

    1958 - 2003
    https://www.esrl.noaa.gov/gmd/ccgg/trends/data.html

    Dr. Pieter Tans, NOAA/ESRL (www.esrl.noaa.gov/gmd/ccgg/trends/) and Dr. Ralph Keeling, Scripps Institution of
    Oceanography (scrippsco2.ucsd.edu/).
    """

    def __init__(self, subsample_fraction: Optional[float] = None):
        super().__init__()
        self.subsample_fraction = subsample_fraction

    def load_or_generate_data(self) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_mauna()

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


def _download_mauna() -> None:
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
    data = loadmat(data_filename)
    return data['X'], data['y']


def main():
    """Load dataset and print info."""
    args = _parse_args()
    dataset = MaunaDataset(subsample_fraction=args.subsample_fraction)
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == '__main__':
    main()
