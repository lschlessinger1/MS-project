import datetime
import os
from typing import Optional, Tuple

import numpy as np
import toml

from src.datasets.dataset import Dataset, _parse_args, _download_raw_dataset

RAW_DATA_DIRNAME = Dataset.data_dirname() / 'raw' / 'methane'
METADATA_FILENAME = RAW_DATA_DIRNAME / 'metadata.toml'

PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / 'processed' / 'methane'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'processed_data.npz'


class MethaneDataset(Dataset):
    """Methane data from the NOAA.

    Atmospheric Methane Dry Air Mole Fractions from the NOAA ESRL GMD Carbon Cycle Cooperative Global Air Sampling
    Network, 1983-2017
    See ftp://aftp.cmdl.noaa.gov/data/trace_gases/ch4/flask/surface/README_surface_flask_ch4.html
    """

    def __init__(self, subsample_fraction: Optional[float] = 0.0817):
        super().__init__()
        self.subsample_fraction = subsample_fraction  # by default, take the first 500 samples

    def load_or_generate_data(self) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_methane()

        with PROCESSED_DATA_FILENAME.open('rb') as f:
            data = np.load(f)

            self.x = data['x'][:, None]
            self.y = data['y'][:, None]

        self._subsample()

    def _subsample(self) -> None:
        """Only this fraction of data will be loaded."""
        if self.subsample_fraction is None:
            return

        num_subsample = int(self.x.shape[0] * self.subsample_fraction)
        self.x = self.x[:num_subsample]
        self.y = self.y[:num_subsample]


def _load_x_y(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a Tx1 array X which gives time (in seconds since the epoch) and a length-T
    vector giving the CH4 measurement at that time."""
    epoch = datetime.datetime.utcfromtimestamp(0)

    x_list = []
    y_list = []
    for line_ in open(filename):
        line = line_.strip()
        if line[0] == '#':
            continue

        parts = line.split()
        year = int(parts[1])
        month = int(parts[2])
        day = int(parts[3])
        hour = int(parts[4])
        minute = int(parts[5])
        second = int(parts[6])
        value = float(parts[11])
        flags = parts[13]

        # ignore anything that's flagged
        if flags != '...':
            continue

        dtime = datetime.datetime(year, month, day, hour, minute, second)
        delta = dtime - epoch
        x_list.append(delta.total_seconds())
        y_list.append(value)

    x = np.array(x_list)
    y = np.array(y_list)
    return x, y


def _download_methane() -> None:
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


def main():
    """Load dataset and print info."""
    args = _parse_args()
    dataset = MethaneDataset(subsample_fraction=args.subsample_fraction)
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == '__main__':
    main()
