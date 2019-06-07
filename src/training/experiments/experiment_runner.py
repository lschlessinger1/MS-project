from typing import List, Callable, Optional, Sequence

import numpy as np

from src.autoks.core.covariance import Covariance
from src.datasets.airline import AirlineDataset
from src.datasets.concrete import ConcreteDataset
from src.datasets.dataset import Dataset
from src.datasets.housing import HousingDataset
from src.datasets.mauna import MaunaDataset
from src.datasets.methane import MethaneDataset
from src.datasets.solar import SolarDataset
from src.datasets.synthetic.known_gp_dataset import KnownGPDataset, cks_known_kernels

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
            experiment.run()


class KnownGPExperimentRunner(ExperimentRunner):

    def get_datasets(self) -> List[KnownGPDataset]:
        # Create synthetic dataset generators
        noise_vars = [10 ** i for i in range(-1, 2)]
        true_kernels = [Covariance(true_kernel) for true_kernel in cks_known_kernels()]
        datasets = [KnownGPDataset(kernel, var, 100) for var in noise_vars for kernel in true_kernels]
        return datasets


class BOMSFilesExperimentRunner(ExperimentRunner):

    def get_datasets(self) -> List[Dataset]:
        return [AirlineDataset(), MaunaDataset(), MethaneDataset(), SolarDataset(), ConcreteDataset(), HousingDataset()]
