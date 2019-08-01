"""Simple way to run experiments defined in a file."""
import argparse
import json


def run_experiments(experiments_filename):
    """Run experiments from file."""
    with open(experiments_filename) as f:
        experiments_config = json.load(f)

    n_experiments = len(experiments_config['experiments'])

    for i in range(n_experiments):
        experiment_config = experiments_config['experiments'][i]
        experiment_config['experiment_group'] = experiments_config['experiment_group']
        print(f"pipenv run python src/training/run_experiment.py '{json.dumps(experiment_config)}'")


def main():
    """Parse command-line arguments and run experiments from provided file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments_filename", type=str, help="Filename of JSON file of experiments to run.")
    args = parser.parse_args()
    run_experiments(args.experiments_filename)


if __name__ == '__main__':
    main()
