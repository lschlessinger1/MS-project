import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from src.datasets import util


class Dataset:
    """Simple abstract class for datasets."""
    x: Optional[np.ndarray]
    y: Optional[np.ndarray]

    def __init__(self):
        self.x = None
        self.y = None
        self.name = self.__class__.__name__.split(Dataset.__name__)[0]

    @classmethod
    def data_dirname(cls) -> Path:
        """Dataset directory name."""
        return Path(__file__).resolve().parents[2] / 'data'

    def load_or_generate_data(self) -> None:
        """Load or generate dataset data."""
        raise NotImplementedError('Must be implemented in a child class')

    def split_train_test(self,
                         test_size: float = 0.2,
                         **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the train-test split of the dataset."""
        self.load_or_generate_data()
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size, **kwargs)
        return x_train, x_test, y_train, y_test

    def __repr__(self):
        return f'{self.__class__.__name__}'


def _download_raw_dataset(metadata: dict) -> None:
    """Download the raw dataset if not already downloaded."""
    if os.path.exists(metadata['filename']):
        return

    print('Downloading raw dataset...')
    util.download_url(metadata['url'], metadata['filename'])

    print('Computing SHA-256...')
    sha256 = util.compute_sha256(metadata['filename'])
    if sha256 != metadata['sha256']:
        raise ValueError('Downloaded data file SHA-256 does not match that listed in metadata document.')


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_fraction",
                        type=float,
                        default=None,
                        help="If given, is used as the fraction of data to expose.")
    return parser.parse_args()
