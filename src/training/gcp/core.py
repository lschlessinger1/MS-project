import os

from src.training.gcp import env

# We use the hidden version if it already exists, otherwise non-hidden.
if os.path.exists(os.path.join(env.get_metadata_dir(os.getcwd()), '.gcp')):
    __stage_dir__ = '.gcp' + os.sep
elif os.path.exists(os.path.join(env.get_metadata_dir(os.getcwd()), 'gcp')):
    __stage_dir__ = "gcp" + os.sep
else:
    __stage_dir__ = None


def gcp_metadata_dir():
    return os.path.join(env.get_metadata_dir(os.getcwd()), __stage_dir__ or ("gcp" + os.sep))
