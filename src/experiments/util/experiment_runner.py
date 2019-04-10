import os
from typing import List, Callable, Optional, Sequence

import numpy as np

from src.experiments.util.data_util import KnownGPDataset, FileDataset, Dataset, cks_known_kernels

EXP_FACTORY = Callable


class ExperimentRunner:

    def __init__(self,
                 exp_factory: EXP_FACTORY,
                 random_seed: Optional[int] = None):
        self.exp_factory = exp_factory
        self.random_seed = random_seed

    def run(self, **kwargs) -> None:
        datasets = self.get_datasets()
        # Set random seed for reproducibility.
        np.random.seed(self.random_seed)
        self.run_experiments(datasets, self.exp_factory, **kwargs)

    def get_datasets(self) -> Sequence[Dataset]:
        raise NotImplementedError('Must be implemented.')

    @staticmethod
    def run_experiments(datasets: Sequence[Dataset],
                        exp_factory: EXP_FACTORY,
                        **kwargs) -> None:
        print(f'Running {len(datasets)} experiment(s)\n')
        for ds in datasets:
            print(f'Performing experiment on \n {ds}')
            experiment = exp_factory(ds, **kwargs)
            experiment.run(title='Random Experiment', create_report=False)


class KnownGPExperimentRunner(ExperimentRunner):

    def get_datasets(self) -> List[KnownGPDataset]:
        # Create synthetic dataset generators
        noise_vars = [10 ** i for i in range(-1, 2)]
        datasets = [KnownGPDataset(kernel, var, 100) for var in noise_vars for kernel in cks_known_kernels()]
        return datasets


class FileExperimentRunner(ExperimentRunner):

    def get_datasets(self) -> List[FileDataset]:
        data_paths = gen_dataset_paths(data_dir='../../data')
        datasets = [FileDataset(path) for path in data_paths]
        return datasets


def gen_dataset_paths(data_dir: str,
                      file_suffix: str = '.csv') -> List[str]:
    """Return a list of dataset file paths.

    Assume that all data files are CSVs
    """
    file_paths = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(file_suffix):
                file_paths.append(os.path.join(root, file))

    return file_paths
