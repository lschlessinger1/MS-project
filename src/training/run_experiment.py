"""Script to run an experiment."""

import argparse
import gzip
import importlib
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

try:
    import comet_ml

    assert comet_ml
    _has_comet_ml = True
except ImportError:
    _has_comet_ml = False

from src.training import gcp
from src.training.gcp.storage import upload_blob
from src.training.util import train_model

DEFAULT_TRAIN_ARGS = {
    'eval_budget': 50,
    'verbose': 1,
}

DIR_NAME = Path(__file__).parents[2].resolve() / 'results'


def run_experiment(experiment_config: dict,
                   save_models: bool,
                   save_experiment: bool = True,
                   use_gcp: bool = True,
                   use_comet: bool = True):
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

    if use_gcp:
        gcp.init()

    experiment = None
    if use_comet:
        if _has_comet_ml:
            comet_config = experiment_config.get('comet_args', {})
            tags = comet_config.pop('tags', [])
            experiment = comet_ml.Experiment(**comet_config)
            experiment.add_tags(tags)
            experiment.log_parameters(experiment_config)
            experiment.log_dataset_hash(dataset.x)
            experiment.log_dataset_info(name=dataset.name)
        else:
            warnings.warn('Please install the `comet_ml` package to use Comet.')

    # Starting time of experiment (used if saving experiment)
    timestamp = str("_".join(str(datetime.today()).split(" "))).replace(":", "-")

    model, history = train_model(
        model_selector,
        dataset,
        eval_budget=experiment_config['train_args']['eval_budget'],
        verbose=experiment_config['train_args']['verbose'],
        use_gcp=use_gcp,
        comet_experiment=experiment
    )

    # Evaluate model selector.
    # TODO: clean this up - don't duplicate evaluation code.
    if experiment:
        with experiment.test():
            x_test, y_test = getattr(dataset, 'x_test', dataset.x), getattr(dataset, 'y_test', dataset.y)
            score = model_selector.evaluate(x_test, y_test)
            print(f'Test evaluation: {score}')
            experiment.log_metric("test_metric", score)
    else:
        x_test, y_test = getattr(dataset, 'x_test', dataset.x), getattr(dataset, 'y_test', dataset.y)
        score = model_selector.evaluate(x_test, y_test)
        print(f'Test evaluation: {score}')

    if use_gcp:
        logging.info({'test_metric': score})

    if save_models:
        model_selector.save_best_model()

    if save_experiment:
        # Create output dictionary.
        output_dict = dict()
        output_dict["history"] = history.to_dict()
        output_dict["dataset_cls"] = experiment_config['dataset']
        output_dict["dataset_args"] = dataset_args
        output_dict['model_selector'] = model_selector.to_dict()

        # Create results directories.
        DIR_NAME.mkdir(parents=True, exist_ok=True)
        exp_group_dir_name = DIR_NAME
        if experiment_config["experiment_group"]:
            exp_group_dir_name /= experiment_config['experiment_group'].replace(" ", "_")
        exp_group_dir_name.mkdir(parents=True, exist_ok=True)
        exp_dir_name = exp_group_dir_name / f'{model_selector.name}_{timestamp}_experiment'

        # Save to compressed output file.
        output_filename = str(exp_dir_name) + ".zip"
        with gzip.GzipFile(output_filename, 'w') as outfile:
            json_str = json.dumps(output_dict)
            json_bytes = json_str.encode('utf-8')
            outfile.write(json_bytes)

        if experiment:
            experiment.log_asset_data(output_dict, file_name=str(exp_dir_name) + ".json")

        if use_gcp:
            # Save output file(s) to bucket
            # TODO: this should be done in the background uploading everything in gcp.run.dir
            bucket_name = "automated-kernel-search"
            upload_blob(bucket_name, json_bytes, outfile.name)
            logging.info(f"Uploaded blob {outfile.name} to bucket {bucket_name}")

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
    parser.add_argument(
        "--nogcp",
        default=False,
        action='store_true',
        help='If true, do not use GCP for this run.'
    )
    parser.add_argument(
        "--nocomet",
        default=False,
        action='store_true',
        help='If true, do not use Comet for this run.'
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()
    experiment_config = json.loads(args.experiment_config)
    run_experiment(experiment_config, args.save, use_gcp=not args.nogcp, use_comet=not args.nocomet)


if __name__ == '__main__':
    main()
