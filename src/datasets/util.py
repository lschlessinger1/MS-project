import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union, Optional
from urllib.request import urlretrieve

from tqdm import tqdm


def compute_sha256(filename: Union[Path, str]):
    """Return SHA256 checksum of a file."""
    with open(filename, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks: int = 1, b_size: int = 1, t_size: Optional[int] = None):
        """

        :param blocks: Number of blocks transferred so far [default: 1].
        :param b_size: Size of each block (in tqdm units) [default: 1].
        :param t_size: Total size (in tqdm units). If [default: None] remains unchanged.
        :return:
        """
        if t_size is not None:
            self.total = t_size  # pylint: disable=attribute-defined-outside-init
        self.update(blocks * b_size - self.n)  # will also set self.n = b * bsize


def download_url(url, filename):
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec


def download_urls(urls, filenames):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(urlretrieve, url, filename) for url, filename in zip(urls, filenames)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print('Error', e)
