from typing import Any, Tuple, Optional, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def plot_best_so_far(best_so_far: List[float],
                     ax: Optional[Axes] = None,
                     title: str = 'Best-So-Far Curve',
                     x_label: str = 'Generation',
                     y_label: str = 'Fitness Best So Far') -> Axes:
    """Display the maximum fitness value at each generation

    :param best_so_far:
    :param ax: Plot into this axis, otherwise get the current axis or make a new one if not existing.
    :param title:
    :param x_label:
    :param y_label:
    :return: Axes with the plot containing the best-so-far values.
    """
    if ax is None:
        ax = plt.gca()

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # x-axis will have integer ticks
    ax.plot(best_so_far, marker='o', markerfacecolor='black')

    return ax


def setup_plot(x_label: str,
               y_label: str,
               title: str,
               ax: Optional[Axes] = None) -> Axes:
    """Set up plot.

    :param x_label:
    :param y_label:
    :param title:
    :param ax: Plot into this axis, otherwise get the current axis or make a new one if not existing.
    :return: Axes with the plot containing some boilerplate setup.
    """
    if ax is None:
        ax = plt.gca()

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return ax


def setup_values(values: List[float],
                 value_label: str,
                 ax: Optional = None) -> Tuple[Axes, np.ndarray, np.ndarray]:
    """Set up values.

    :param fig:
    :param values:
    :param value_label:
    :param ax: Plot into this axis, otherwise get the current axis or make a new one if not existing.
    :return: Axes with the plot containing the values.
    """
    if ax is None:
        ax = plt.gca()

    x = np.arange(len(values)) + 1
    y = np.array(values)

    ax.plot(x, y, lw=2, label=value_label, marker='o', markerfacecolor='black')

    return ax, x, y


def setup_stds(stds: List[float],
               mu: np.ndarray,
               t: np.ndarray,
               ax: Optional = None,
               std_label: str = 'Confidence') -> \
        Tuple[Axes, np.ndarray]:
    """Set up standard deviations.

    :param fig:
    :param stds:
    :param mu:
    :param t:
    :param ax: Plot into this axis, otherwise get the current axis or make a new one if not existing.
    :param std_label:
    :return: Axes with the plot containing the standard deviations.
    """
    if ax is None:
        ax = plt.gca()

    sigma = np.array(stds)
    ax.fill_between(t, mu + sigma, mu - sigma, alpha=0.5, label=std_label)

    return ax, sigma


def setup_optima(x: np.ndarray,
                 optima: List[float],
                 optima_label: str,
                 ax: Optional[Axes] = None) -> Axes:
    """Set up optima.

    :param x:
    :param optima:
    :param optima_label:
    :param ax: Plot into this axis, otherwise get the current axis or make a new one if not existing.
    :return: Axes with the plot containing the optima.
    """
    if ax is None:
        ax = plt.gca()

    ax.plot(x, optima, lw=2, label=optima_label, marker='x')

    return ax


def plot_distribution(values: List[float],
                      stds: Optional[List[float]] = None,
                      optima: Optional[List[float]] = None,
                      x_label: str = 'generation',
                      value_name: str = 'average',
                      metric_name: str = 'fitness',
                      optima_name: str = 'best') -> Axes:
    """Plot distribution of values.

    :param values:
    :param stds:
    :param optima:
    :param x_label:
    :param value_name:
    :param metric_name:
    :param optima_name:
    :return: Axes with the distribution plot.
    """
    x_name = x_label.capitalize()
    y_name = metric_name.capitalize()
    value_label = ('%s %s' % (value_name, metric_name)).capitalize()
    title = value_label

    ax = setup_plot(x_name, y_name, title)
    ax, x, y = setup_values(values, value_label, ax=ax)

    if stds is not None:
        ax, sigma = setup_stds(stds, y, x, ax=ax)
        title = ''.join((title, r' and $\pm \sigma$ interval'))
        ax.set_title(title)

    if optima is not None:
        optima_label = ('%s %s' % (optima_name, metric_name)).capitalize()
        ax = setup_optima(x, optima, optima_label, ax=ax)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # x-axis will have integer ticks

    ax.legend()

    return ax
