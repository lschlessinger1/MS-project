import os
from abc import ABC

import numpy as np
from sklearn.model_selection import train_test_split

from src.autoks.experiment import Experiment


def gen_dataset_paths(data_dir: str, file_suffix: str = '.csv'):
    """Return a list of dataset file paths.

    Assume that all data files are CSVs
    """
    file_paths = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(file_suffix):
                file_paths.append(os.path.join(root, file))

    return file_paths


def run_experiments(ds_generators, grammar, objective, base_kernels=None, **kwargs):
    for generator in ds_generators:
        print(f'Performing experiment on {generator.path}')
        X, y = generator.gen_dataset()

        if base_kernels is None:
            if X.shape[1] > 1:
                base_kernels = ['SE', 'RQ']
            else:
                base_kernels = ['SE', 'RQ', 'LIN', 'PER']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        experiment = Experiment(grammar, objective, base_kernels, X_train, y_train, X_test, y_test, **kwargs)
        experiment.run(title='Random Experiment')


class DatasetGenerator:

    def gen_dataset(self):
        raise NotImplementedError('Must be implemented in a child class')


class SyntheticDatasetGenerator(DatasetGenerator, ABC):

    def __init__(self, n_samples, input_dim):
        self.n_samples = n_samples
        self.input_dim = input_dim


class Input1DSynthGenerator(SyntheticDatasetGenerator, ABC):

    def __init__(self, n_samples, input_dim):
        super().__init__(n_samples, input_dim)
        if self.input_dim != 1:
            raise ValueError('Input dimension must be 1')


class FileDatasetGenerator(DatasetGenerator):

    def __init__(self, file_path):
        # assume file type of CSV
        self.path = file_path

    def gen_dataset(self):
        data = np.genfromtxt(self.path, delimiter=',')
        # assume output dimension is 1
        X, y = data[:, :-1], data[:, -1]
        return X, y
