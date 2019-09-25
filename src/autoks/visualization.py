import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.autoks.core.gp_model import GPModel
from src.autoks.statistics import StatBook
from src.evalg.visualization import plot_distribution, plot_best_so_far


def plot_best_scores(score_name: str,
                     evaluations_name: str,
                     stat_book: StatBook) -> None:
    """Plot the best models scores

    :return:
    """
    if stat_book.name == evaluations_name:
        best_scores = stat_book.running_max(score_name)
        x_label = 'evaluations'
    else:
        best_scores = stat_book.maximum(score_name)
        x_label = 'generation'

    ax = plot_best_so_far(best_scores, x_label=x_label)
    fig = ax.figure
    fig.suptitle(f'{stat_book.label}', y=1)
    fig.subplots_adjust(top=0.88)
    plt.show()


def plot_score_summary(score_name: str,
                       evaluations_name: str,
                       stat_book: StatBook) -> None:
    """Plot a summary of model scores

    :return:
    """
    if stat_book.name == evaluations_name:
        best_scores = stat_book.running_max(score_name)
        mean_scores = stat_book.running_mean(score_name)
        std_scores = stat_book.running_std(score_name)
        x_label = 'evaluations'
    else:
        best_scores = stat_book.maximum(score_name)
        mean_scores = stat_book.mean(score_name)
        std_scores = stat_book.std(score_name)
        x_label = 'generation'

    ax = plot_distribution(mean_scores, std_scores, best_scores, x_label=x_label)
    fig = ax.figure
    fig.suptitle(f'{stat_book.label}', y=1)
    fig.subplots_adjust(top=0.88)
    plt.show()


def plot_n_hyperparams_summary(n_hyperparams_name: str,
                               best_stat_name: str,
                               stat_book: StatBook,
                               x_label: str) -> None:
    """Plot a summary of the number of hyperparameters

    :return:
    """
    if best_stat_name in stat_book.multi_stats[n_hyperparams_name].stats:
        best_n_hyperparameters = stat_book.multi_stats[n_hyperparams_name].stats[best_stat_name].data
    else:
        best_n_hyperparameters = None
    median_n_hyperparameters = stat_book.median(n_hyperparams_name)
    std_n_hyperparameters = stat_book.std(n_hyperparams_name)
    ax = plot_distribution(median_n_hyperparameters, std_n_hyperparameters, best_n_hyperparameters,
                           value_name='median', metric_name='# Hyperparameters', x_label=x_label)
    fig = ax.figure
    fig.suptitle(f'{stat_book.label}', y=1)
    fig.subplots_adjust(top=0.88)
    plt.show()


def plot_n_operands_summary(n_operands_name: str,
                            best_stat_name: str,
                            stat_book: StatBook,
                            x_label: str) -> None:
    """Plot a summary of the number of operands

    :return:
    """
    if best_stat_name in stat_book.multi_stats[n_operands_name].stats:
        best_n_operands = stat_book.multi_stats[n_operands_name].stats[best_stat_name].data
    else:
        best_n_operands = None
    median_n_operands = stat_book.median(n_operands_name)
    std_n_operands = stat_book.std(n_operands_name)
    ax = plot_distribution(median_n_operands, std_n_operands, best_n_operands, value_name='median',
                           metric_name='# Operands', x_label=x_label)
    fig = ax.figure
    fig.suptitle(f'{stat_book.label}', y=1)
    fig.subplots_adjust(top=0.88)
    plt.show()


def plot_base_kernel_freqs(base_kern_freq_names: list,
                           stat_book: StatBook,
                           x_label: str) -> None:
    """Plot base kernel frequency across generations.

    :param base_kern_freq_names:
    :param stat_book:
    :param x_label:

    :return:
    """
    freqs = [(stat_book.sum(key), key) for key in base_kern_freq_names]

    plt.title('Base Kernel Frequency')
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # x-axis will have integer ticks
    for freq, key in freqs:
        plt.plot(freq, label=key, marker='o', markerfacecolor='black')
    plt.legend()
    plt.gcf().suptitle(f'{stat_book.label}', y=1)
    plt.gcf().subplots_adjust(top=0.88)
    plt.show()


def plot_cov_dist_summary(cov_dists_name: str,
                          stat_book: StatBook,
                          x_label: str) -> None:
    """Plot a summary of the homogeneity of models over each generation.

    :return:
    """
    mean_cov_dists = stat_book.mean(cov_dists_name)
    std_cov_dists = stat_book.std(cov_dists_name)
    ax = plot_distribution(mean_cov_dists, std_cov_dists, metric_name='covariance distance', x_label=x_label)
    fig = ax.figure
    fig.suptitle(f'{stat_book.label}', y=1)
    fig.subplots_adjust(top=0.88)
    plt.show()


def plot_kernel_diversity_summary(diversity_scores_name: str,
                                  stat_book: StatBook,
                                  x_label: str) -> None:
    """Plot a summary of the diversity of models over each generation.

    :return:
    """
    mean_diversity_scores = stat_book.running_mean(diversity_scores_name)
    std_diversity_scores = stat_book.running_std(diversity_scores_name)
    ax = plot_distribution(mean_diversity_scores, std_diversity_scores, metric_name='diversity',
                           value_name='population', x_label=x_label)
    fig = ax.figure
    fig.suptitle(f'{stat_book.label}', y=1)
    fig.subplots_adjust(top=0.88)
    plt.show()


def plot_kernel_tree(gp_model: GPModel,
                     graph_name: str = 'best_kernel_tree',
                     directory: str = '../results/figures') -> None:
    """Create a kernel tree file and plot it.

    :param gp_model:
    :param graph_name:
    :param directory:
    :return:
    """
    graph = gp_model.covariance.to_binary_tree().create_graph(name=graph_name)
    graph.format = 'png'  # only tested with PNG
    graph.render(f"{graph_name}.gv", directory, view=False, cleanup=True)
    img = plt.imread(graph.filepath + '.' + graph.format)

    f, ax = plt.subplots(figsize=(5, 5), dpi=100)
    f.subplots_adjust(0, 0, 1, 1)
    ax.imshow(img)
    ax.set_axis_off()

    return ax
