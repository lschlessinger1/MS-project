"""Script to run an experiment."""
import argparse
import gzip
import importlib
import json
from datetime import datetime
from pathlib import Path

from src.training.util import train_model

DEFAULT_TRAIN_ARGS = {
    'eval_budget': 50,
    'verbose': 1,
}

DIR_NAME = Path(__file__).parents[2].resolve() / 'results'


def run_experiment(experiment_config: dict,
                   save_models: bool,
                   save_experiment: bool = True):
    """
    Run a training experiment.
    Parameters
    ----------
    experiment_config (dict)
        Of the form
        {
            "dataset": "AirlineDataset",
            "dataset_args": {
                "max_overlap": 0.4,
                "subsample_fraction": 0.2
            },
            "gp": "gp_regression",
            "gp_args": {
                "inference_method": "laplace"
            },
            "train_args": {
                "eval_budget": 50,
                "verbose": 1
            }
        }
    save_models (bool)
        If True, will save the final models to a canonical location
    save_experiment (bool)
        If True, will save the experiment to a canonical location
    """
    print(f'Running experiment with config {experiment_config}')

    # Get dataset.
    datasets_module = importlib.import_module('src.datasets')
    dataset_class_ = getattr(datasets_module, experiment_config['dataset'])
    dataset_args = experiment_config.get('dataset_args', {})
    dataset = dataset_class_(**dataset_args)
    dataset.load_or_generate_data()
    print(dataset)

    # Get model selector.
    models_module = importlib.import_module('src.autoks.core.model_selection')
    model_class_ = getattr(models_module, experiment_config['model_selector'])

    # Get GP.
    gp_fn_ = experiment_config['gp']
    gp_args = experiment_config.get('gp_args', {})
    model_selector_args = experiment_config.get('model_selector_args', {})
    model_selector = model_class_(
        gp_fn=gp_fn_,
        gp_args=gp_args,
        **model_selector_args
    )
    print(model_selector)

    experiment_config['train_args'] = {**DEFAULT_TRAIN_ARGS, **experiment_config.get('train_args', {})}
    experiment_config['experiment_group'] = experiment_config.get('experiment_group', None)

    # Starting time of experiment (used if saving experiment)
    timestamp = str("_".join(str(datetime.today()).split(" "))).replace(":", "-")

    model, tracker = train_model(
        model_selector,
        dataset,
        eval_budget=experiment_config['train_args']['eval_budget'],
        verbose=experiment_config['train_args']['verbose']
    )

    # Evaluate model selector.
    has_test_data = hasattr(dataset, 'x_test') and hasattr(dataset, 'y_test')
    if has_test_data:
        score = model_selector.evaluate(dataset.x_test, dataset.y_test)
    else:
        score = model_selector.evaluate(dataset.x, dataset.y)
    print(f'Test evaluation: {score}')

    if save_models:
        model_selector.save_best_model()

    if save_experiment:
        # Create output dictionary.
        output_dict = dict()
        output_dict["tracker"] = tracker.to_dict()
        output_dict["dataset_cls"] = experiment_config['dataset']
        output_dict["dataset_args"] = dataset_args
        output_dict['model_selector'] = model_selector.to_dict()

        # Create results directories.
        DIR_NAME.mkdir(parents=True, exist_ok=True)
        exp_group_dir_name = DIR_NAME / experiment_config['experiment_group'].replace(" ", "_")
        exp_group_dir_name.mkdir(parents=True, exist_ok=True)
        exp_dir_name = exp_group_dir_name / f'{model_selector.name}_{timestamp}_experiment'

        # Save to compressed output file.
        output_filename = str(exp_dir_name) + ".zip"
        with gzip.GzipFile(output_filename, 'w') as outfile:
            json_str = json.dumps(output_dict)
            json_bytes = json_str.encode('utf-8')
            outfile.write(json_bytes)

        return output_filename


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        default=False,
        dest='save',
        action='store_true',
        help="If true, then final selected models will be saved to canonical, version-controlled location"
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help="Experiment JSON ('{\"dataset\": \"AirlineDataset\", \"model_selector\": \"EvolutionaryModelSelector\", "
             "\"gp\": \"gp_regression\"}' "
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()
    experiment_config = json.loads(args.experiment_config)
    run_experiment(experiment_config, args.save)


if __name__ == '__main__':
    main()
