from typing import Optional

from GPy.core import GP
from GPy.models import GPRegression


class SurrogateModel:

    def update(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def f_max(self):
        raise NotImplementedError


class GPSurrogate(SurrogateModel):
    model: Optional[GP]

    def __init__(self, kernel, model_hyperpriors):
        self.kernel = kernel
        self.model = None
        self.hyperpriors = model_hyperpriors

    def _create_model(self, x, y):
        self.model = GPRegression(x, y, kernel=self.kernel)

    def update(self, x, y):
        if self.model is None:
            self._create_model(x, y)
        else:
            self.model.set_XY(x, y)

        self.model.optimize()

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def f_max(self):
        return self.model.predict(self.model.X)[0].max()
