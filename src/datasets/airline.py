import os
from typing import Optional

import numpy as np
import toml

from src.datasets.dataset import Dataset, _download_raw_dataset, _parse_args

RAW_DATA_DIRNAME = Dataset.data_dirname() / 'raw' / 'airline'
METADATA_FILENAME = RAW_DATA_DIRNAME / 'metadata.toml'

PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / 'processed' / 'airline'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'processed_data.npz'


class AirlineDataset(Dataset):
    """U.S. international airline passengers dataset

    The international airline passenger series describes monthly totals (in thousands) of the international passengers
    for the period between January 1949 and December 1960.

    Time Series: Forecast and Control by Box, Jenkins and Reinsel
    """

    def __init__(self, subsample_fraction: Optional[float] = None):
        super().__init__()
        self.metadata = toml.load(METADATA_FILENAME)

        self.subsample_fraction = subsample_fraction

    def load_or_generate_data(self) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            self._download_airline()

        with PROCESSED_DATA_FILENAME.open('rb') as f:
            data = np.load(f)

            self.x = data['x'][:, None]
            self.y = data['y'][:, None]

        self._subsample()

    def _download_airline(self) -> None:
        curdir = os.getcwd()
        os.chdir(RAW_DATA_DIRNAME)
        _download_raw_dataset(self.metadata)
        _process_raw_dataset(self.metadata['filename'])
        os.chdir(curdir)

    def _subsample(self) -> None:
        """Only this fraction of data will be loaded."""
        if self.subsample_fraction is None:
            return

        num_subsample = int(self.x.shape[0] * self.subsample_fraction)
        self.x = self.x[:num_subsample]
        self.y = self.y[:num_subsample]


def _process_raw_dataset(filename: str):
    print('Loading training data from .txt file')
    x, y = _load_X_y(filename)

    print('Saving to NPZ...')
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)

    np.savez(PROCESSED_DATA_FILENAME, x=x, y=y)


def _load_data(filename: str):
    result = []
    for line in open(filename):
        values = map(float, line.strip().split())
        result += values
    return np.array(result)


def _load_X_y(data_filename: str):
    """X is a vector giving the time step, and y is the total number of international
    airline passengers, in thousands. Each element corresponds to one
    month, and it goes from Jan. 1949 through Dec. 1960."""
    values = _load_data(data_filename)
    return np.arange(values.size).astype(float), values.astype(float)


def main():
    """Load dataset and print info."""
    args = _parse_args()
    dataset = AirlineDataset(subsample_fraction=args.subsample_fraction)
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == '__main__':
    main()
