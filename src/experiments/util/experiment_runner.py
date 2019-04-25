import os
from typing import List, Callable, Optional, Sequence

import numpy as np

from src.autoks.core.covariance import Covariance
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
        true_kernels = [Covariance(true_kernel) for true_kernel in cks_known_kernels()]
        datasets = [KnownGPDataset(kernel, var, 100) for var in noise_vars for kernel in true_kernels]
        return datasets


class FileExperimentRunner(ExperimentRunner):

    def get_datasets(self) -> List[FileDataset]:
        data_dir = os.path.join('..', '..', 'data')
        data_paths = gen_dataset_paths(data_dir=data_dir)
        datasets = [FileDataset(path) for path in data_paths]
        return datasets


class BOMSFilesExperimentRunner(ExperimentRunner):

    def get_datasets(self) -> List[Dataset]:
        data_dir = os.path.join('..', '..', 'data')

        data_1d_dir = os.path.join(data_dir, '1d_data')
        airline_path = os.path.join(data_1d_dir, 'airline.csv')
        mauna_path = os.path.join(data_1d_dir, 'mauna.csv')
        methane_500_path = os.path.join(data_1d_dir, 'methane_500.csv')
        solar_path = os.path.join(data_1d_dir, 'solar.csv')
        data_1d_paths = [airline_path, mauna_path, methane_500_path, solar_path]

        data_multi_d_dir = os.path.join(data_dir, 'multi_dimensional')
        concrete_path = os.path.join(data_multi_d_dir, 'concrete_500.csv')
        housing_path = os.path.join(data_multi_d_dir, 'housing.csv')
        data_multi_d_paths = [concrete_path, housing_path]

        data_paths = data_1d_paths + data_multi_d_paths
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
