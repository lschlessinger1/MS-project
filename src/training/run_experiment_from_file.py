import argparse
import json
from multiprocessing.pool import Pool

from src.autoks.postprocessing.summary import summarize
from src.training.run_experiment import run_experiment


def run_experiments(experiments_filename, save: bool):
    """Run experiments from file."""
    with open(experiments_filename) as f:
        experiments_config = json.load(f)

    n_experiments = len(experiments_config['experiments'])
    exp_dir_names = []

    for i in range(n_experiments):
        experiment_config = experiments_config['experiments'][i]
        experiment_config['experiment_group'] = experiments_config['experiment_group']
        exp_dirname = run_experiment(experiment_config, save_models=save)
        exp_dir_names.append(exp_dirname)

    return exp_dir_names


def main():
    """Run experiment."""
    parser = argparse.ArgumentParser(description='Run model search experiment from a file.')

    parser.add_argument(
        "--save",
        default=False,
        dest='save',
        action='store_true',
        help="If true, then final selected models will be saved to canonical, version-controlled location"
    )

    parser.add_argument(
        "--summarize",
        default=False,
        dest='summarize',
        action='store_true',
        help="If true, then the experiment will be summarized"
    )

    parser.add_argument(
        "--n_repeats",
        default=1,
        type=int,
        dest='n_repeats',
        help="The experiment will be repeated `n_repeats` times"
    )

    parser.add_argument(
        "--parallel",
        default=True,
        dest='parallel',
        action='store_true',
        help="If true, then the experiment will use multiprocessing"
    )

    parser.add_argument(
        "--num_processes",
        default=None,
        dest='num_processes',
        type=int,
        help="If using multiprocessing, then the experiment will use `num_processes` processes"
    )

    parser.add_argument("experiments_filename", type=str, help="Filename of JSON file of experiments to run.")
    args = parser.parse_args()

    if args.parallel:
        with Pool(processes=args.num_processes) as p:
            results = p.starmap(run_experiments, [(args.experiments_filename, args.save)] * args.n_repeats)
    else:
        results = [run_experiments(args.experiments_filename, args.save) for _ in range(args.n_repeats)]

    if args.summarize:
        for results_dir_names in results:
            # Convenience option to summarize experiment after running it.
            for exp_dirname in results_dir_names:
                summarize(str(exp_dirname))


if __name__ == '__main__':
    main()
