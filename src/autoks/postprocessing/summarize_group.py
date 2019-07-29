import argparse
from pathlib import Path
from typing import List

import numpy as np

from src.autoks.postprocessing.summary import _parse_experiment
from src.evalg.visualization import plot_distribution


def summarize_exp_group(experiment_group_dir_name):
    """Summarize a group of experiments."""
    print(f'Summarizing {experiment_group_dir_name}')
    exp_dicts = _parse_experiment_group(experiment_group_dir_name)
    histories = [d['history'] for d in exp_dicts]

    create_plots(histories)


def _parse_experiment_group(experiment_group_dir_name) -> List[dict]:
    path = Path(experiment_group_dir_name)
    exp_dicts = [_parse_experiment(str(x)) for x in path.iterdir() if x.is_file()]
    print(f' Found {len(exp_dicts)} experiments')
    return exp_dicts


def create_plots(histories):
    best_scores = [history.stat_book_collection.stat_books['evaluations'].running_max('score') for history in histories]

    mean_best = list(np.mean(best_scores, axis=0).tolist())
    mean_std = list(np.std(best_scores, axis=0).tolist())

    import matplotlib.pyplot as plt
    plot_distribution(mean_best, mean_std, value_name='Mean best', x_label='evaluations')
    plt.gcf().suptitle(f'Evaluations', y=1)
    plt.gcf().subplots_adjust(top=0.88)
    plt.show()


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_group_dirname", type=str, help="Directory name of an experiment group to run "
                                                                   "postprocessing on.")
    args = parser.parse_args()
    return args


def main():
    """Summarize experiment."""
    args = _parse_args()
    summarize_exp_group(args.experiment_group_dirname)


if __name__ == '__main__':
    main()
