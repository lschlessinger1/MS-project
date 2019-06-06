import os
from typing import Optional

import numpy as np
import toml

from src.datasets.dataset import Dataset, _download_raw_dataset, _parse_args

RAW_DATA_DIRNAME = Dataset.data_dirname() / 'raw' / 'solar'
METADATA_FILENAME = RAW_DATA_DIRNAME / 'metadata.toml'

PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / 'processed' / 'solar'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'processed_data.npz'


class SolarDataset(Dataset):
    """Solar irradiance data

    http://lasp.colorado.edu/data/sorce/tsi_data/
    """

    def __init__(self, subsample_fraction: Optional[float] = None):
        super().__init__()

        self.subsample_fraction = subsample_fraction

    def load_or_generate_data(self) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_solar()

        with PROCESSED_DATA_FILENAME.open('rb') as f:
            data = np.load(f)

            self.x = data['x'][:, None]
            self.y = data['y'][:, None]

        self._subsample()

    def _subsample(self):
        """Only this fraction of data will be loaded."""
        if self.subsample_fraction is None:
            return

        num_subsample = int(self.x.shape[0] * self.subsample_fraction)
        self.x = self.x[:num_subsample]
        self.y = self.y[:num_subsample]


def _download_solar():
    metadata = toml.load(METADATA_FILENAME)
    curdir = os.getcwd()
    os.chdir(RAW_DATA_DIRNAME)
    _download_raw_dataset(metadata)
    _process_raw_dataset(metadata['filename'])
    os.chdir(curdir)


def _process_raw_dataset(filename: str):
    print('Loading training data from .txt file')
    x, y = _load_x_y(filename)

    print('Saving to NPZ...')
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)

    np.savez(PROCESSED_DATA_FILENAME, x=x, y=y)


def _load_x_y(data_file: str):
    """Returns a Tx1 matrix X representing the year, and a length-T
    vector y representing the solar irradiance."""
    x_list = []
    y_list = []
    for line in open(data_file):
        if line[0] == ';':
            continue

        parts = line.strip().split()
        year = float(parts[0])
        irrad = float(parts[1])
        x_list.append(year)
        y_list.append(irrad)

    x = np.array(x_list)
    y = np.array(y_list)
    return x, y


def main():
    """Load dataset and print info."""
    args = _parse_args()
    dataset = SolarDataset(subsample_fraction=args.subsample_fraction)
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == '__main__':
    main()
