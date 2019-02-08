import matplotlib.pyplot as plt
import numpy as np


def plot_best_so_far(best_so_far, title='Best-So-Far Curve', x_label='Generation', y_label='Fitness Best So Far'):
    """ Display the maximum fitness value at each generation

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


def setup_plot(x_label, y_label, title):
    fig, ax = plt.subplots(1)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return fig


def setup_values(fig, values, value_label):
    ax = fig.axes[0]

    x = np.arange(len(values)) + 1
    y = np.array(values)
    ax.plot(x, y, lw=2, label=value_label)

    return fig, x, y


def setup_stds(fig, stds, mu, t, std_label='Confidence'):
    ax = fig.axes[0]

    sigma = np.array(stds)
    ax.fill_between(t, mu + sigma, mu - sigma, alpha=0.5, label=std_label)

    return fig, sigma


def setup_optima(fig, x, optima, optima_label):
    ax = fig.axes[0]
    ax.plot(x, optima, lw=2, label=optima_label)
    return fig


def plot_distribution(values, stds=None, optima=None, x_label='generation', value_name='average',
                      metric_name='fitness', optima_name='best'):
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
