from typing import Any, Tuple, Optional, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_best_so_far(best_so_far: np.ndarray, title: str = 'Best-So-Far Curve', x_label: str = 'Generation',
                     y_label: str = 'Fitness Best So Far') -> Any:
    """Display the maximum fitness value at each generation

    :param best_so_far:
    :param title:
    :param x_label:
    :param y_label:
    :return:
    """
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return plt.plot(best_so_far)


def setup_plot(x_label: str, y_label: str, title: str) -> Figure:
    """Set up plot.

    :param x_label:
    :param y_label:
    :param title:
    :return:
    """
    fig, ax = plt.subplots(1)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return fig


def setup_values(fig: Figure, values: List[float], value_label: str) -> Tuple[Figure, np.ndarray, np.ndarray]:
    """Set up values.

    :param fig:
    :param values:
    :param value_label:
    :return:
    """
    ax = fig.axes[0]

    x = np.arange(len(values)) + 1
    y = np.array(values)
    ax.plot(x, y, lw=2, label=value_label)

    return fig, x, y


def setup_stds(fig: Figure, stds: List[float], mu: np.ndarray, t: np.ndarray, std_label: str = 'Confidence') -> \
        Tuple[Figure, np.ndarray]:
    """Set up standard deviations.

    :param fig:
    :param stds:
    :param mu:
    :param t:
    :param std_label:
    :return:
    """
    ax = fig.axes[0]

    sigma = np.array(stds)
    ax.fill_between(t, mu + sigma, mu - sigma, alpha=0.5, label=std_label)

    return fig, sigma


def setup_optima(fig: Figure, x: np.ndarray, optima: List[float], optima_label: str) -> Figure:
    """Set up optima.

    :param fig:
    :param x:
    :param optima:
    :param optima_label:
    :return:
    """
    ax = fig.axes[0]
    ax.plot(x, optima, lw=2, label=optima_label)
    return fig


def plot_distribution(values: List[float], stds: Optional[List[float]] = None, optima: Optional[List[float]] = None,
                      x_label: str = 'generation', value_name: str = 'average', metric_name: str = 'fitness',
                      optima_name: str = 'best') -> Figure:
    """Plot distribution of values.

    :param values:
    :param stds:
    :param optima:
    :param x_label:
    :param value_name:
    :param metric_name:
    :param optima_name:
    :return:
    """
    x_name = x_label.capitalize()
    y_name = metric_name.capitalize()
    value_label = ('%s %s' % (value_name, metric_name)).capitalize()
    title = value_label

    fig = setup_plot(x_name, y_name, title)
    fig, x, y = setup_values(fig, values, value_label)

    if stds is not None:
        fig, sigma = setup_stds(fig, stds, y, x)
        title = ''.join((title, r' and $\pm \sigma$ interval'))
        ax = fig.axes[0]
        ax.set_title(title)

    if optima is not None:
        optima_label = ('%s %s' % (optima_name, metric_name)).capitalize()
        fig = setup_optima(fig, x, optima, optima_label)

    ax = fig.axes[0]

    ax.legend()

    return fig
