import os

METADATA_DIR = 'GCP_METADATA_DIR'


def get_metadata_dir(default=None, env=None):
    if env is None:
        env = os.environ
    return env.get(METADATA_DIR, default)
