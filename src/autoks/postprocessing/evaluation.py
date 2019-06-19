import numpy as np
from GPy.core import GP
from GPy.kern import RBF
from GPy.models import GPRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


def compute_skmodel_rmse(model,
                         x_train: np.ndarray,
                         y_train: np.ndarray,
                         x_test: np.ndarray,
                         y_test: np.ndarray) -> float:
    """RMSE of a scikit-learn model.

    :param model:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


def compute_gpy_model_rmse(model: GP,
                           x_test: np.ndarray,
                           y_test: np.ndarray) -> float:
    """RMSE of a GPy model.

    :param model:
    :param x_test:
    :param y_test:
    :return:
    """
    mean, var = model.predict(x_test)
    y_pred = mean
    return np.sqrt(mean_squared_error(y_test, y_pred))


def rmse_rbf(x_train: np.ndarray,
             y_train: np.ndarray,
             x_test: np.ndarray,
             y_test: np.ndarray) -> float:
    """RMSE of a GPy RBF kernel.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    model = GPRegression(x_train, y_train, kernel=RBF(input_dim=x_train.shape[1]))
    model.optimize()
    return compute_gpy_model_rmse(model, x_test, y_test)


def rmse_svr(x_train: np.ndarray,
             y_train: np.ndarray,
             x_test: np.ndarray,
             y_test: np.ndarray) -> float:
    """RMSE of a Support Vector Machine for regression.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    return compute_skmodel_rmse(SVR(kernel='rbf'), x_train, y_train, x_test, y_test)


def rmse_lin_reg(x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_test: np.ndarray,
                 y_test: np.ndarray) -> float:
    """RMSE of a linear regression model.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    return compute_skmodel_rmse(LinearRegression(), x_train, y_train, x_test, y_test)


def rmse_knn(x_train: np.ndarray,
             y_train: np.ndarray,
             x_test: np.ndarray,
             y_test: np.ndarray) -> float:
    """RMSE of a k-nearest neighbors regressor.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    return compute_skmodel_rmse(KNeighborsRegressor(), x_train, y_train, x_test, y_test)


def rmse_to_smse(rmse: float, y_test: np.ndarray) -> float:
    """Computes the standardized mean squared error (SMSE)

    The trivial method of guessing the mean of the training targets will have a SMSE of approximately 1
    """
    mse = rmse ** 2
    target_variance = y_test.var()
    return mse / target_variance
