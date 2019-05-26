from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


class BaseExperiment:

    def __init__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_test: Optional[np.ndarray],
                 y_test: Optional[np.ndarray],
                 standardize_x: bool = True,
                 standardize_y: bool = True,
                 hide_warnings: bool = False):

        self.x_train = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train
        if x_test is not None:
            self.x_test = x_test.reshape(-1, 1) if x_test.ndim == 1 else x_test
        else:
            self.x_test = None

        self.y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        if y_test is not None:
            self.y_test = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
        else:
            self.y_test = None

        # only save scaled version of data
        self.standardize_x = standardize_x
        self.standardize_y = standardize_y
        if standardize_x:
            scaler = StandardScaler()
            self.x_train = scaler.fit_transform(self.x_train)
            if self.x_test is not None:
                self.x_test = scaler.transform(self.x_test)

        if standardize_y:
            y_normalizer = StandardScaler()
            self.y_train = y_normalizer.fit_transform(self.y_train)
            if self.x_test is not None:
                self.y_test = y_normalizer.transform(self.y_test)

        self.n_dims = self.x_train.shape[1]

        self.has_test_data = x_test is not None and y_test is not None

        self.hide_warnings = hide_warnings

    def run(self) -> None:
        raise NotImplementedError
