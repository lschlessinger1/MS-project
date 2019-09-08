import logging
import os
from pathlib import Path
from typing import Optional

from src.training.gcp.log import setup_logging

logger = logging.getLogger(__name__)

# Will be set to the run object for the current run, as returned by
# `gcp.init()`.
run = None

config = None  # config object shared with the global run


def log():
    pass


def init(metadata_dir: Optional[str] = None):
    """

    :param metadata_dir: An absolute path to a directory where metadata will be stored
    :return:
    """
    """Create project if it doesn't exist."""
    global run

    gcp_credentials = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    if gcp_credentials:
        logging.info('Found GCP credentials!')
        # Now, resolve path.
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(Path(gcp_credentials).resolve())

    setup_logging()
    logging.info('Logging initialized.')


def save():
    pass


__all__ = ['init', 'config', 'run']
