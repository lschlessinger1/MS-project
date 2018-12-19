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
    plt.plot(best_so_far)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_score_summary(mean_scores, std_scores, best_scores, x_label='Generation', y_label='Fitness'):
    """ Plot average fitness showing standard deviation area and best fitness

    :param mean_scores:
    :param std_scores:
    :param best_scores:
    :param x_label:
    :param y_label:
    :return:
    """
    mu = np.array(mean_scores)
    sigma = np.array(std_scores)
    t = np.arange(len(mean_scores)) + 1

    fig, ax = plt.subplots(1)
    ax.plot(t, best_scores, lw=2, label='Max. fitness')
    ax.plot(t, mu, lw=2, label='Average fitness')
    ax.fill_between(t, mu + sigma, mu - sigma, alpha=0.5)
    ax.set_title(r'Average Fitness $\mu$ and $\pm \sigma$ interval')
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()
