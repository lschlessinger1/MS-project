from typing import Optional

import numpy as np


class BaseExperiment:

    def __init__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_test: Optional[np.ndarray],
                 y_test: Optional[np.ndarray],
                 hide_warnings: bool = False):
        self.x_train = x_train
        self.x_test = x_test

        self.y_train = y_train
        self.y_test = y_test

        self.has_test_data = x_test is not None and y_test is not None

        self.hide_warnings = hide_warnings

    def run(self, verbose: int = 2) -> None:
        raise NotImplementedError
