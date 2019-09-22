import argparse
import os
from pathlib import Path

import papermill as pm


def _parse_args():
    parser = argparse.ArgumentParser(description='Download experiments.')

    parser.add_argument(
        "--input_ipynb_path",
        default='../notebooks/evaluation/download-experiments.ipynb',
        dest='input_ipynb_path',
        type=str,
        help="The input path of the generated IPython notebook.")

    parser.add_argument(
        "--output_ipynb_path",
        default='download-experiments-output.ipynb',
        dest='output_ipynb_path',
        type=str,
        help="The output path of the generated IPython notebook."
    )

    parser.add_argument(
        "--result_dir",
        default='results',
        dest='result_dir',
        type=str,
        help="The path of the results directory."
    )

    parser.add_argument("experiment_dir_names",
                        # type=list,
                        help="Experiment directory names.",
                        nargs='+')

    return parser.parse_args()


def main():
    """Download experiment."""
    args = _parse_args()

    input_path = str(Path(args.input_ipynb_path).resolve())
    output_path = str(Path(args.output_ipynb_path).resolve())
    result_dir = str(Path(args.result_dir).resolve())
    gcp_credentials_path = str(Path(os.environ['GOOGLE_APPLICATION_CREDENTIALS']).resolve())

    params = dict(gcp_credentials_path=gcp_credentials_path,
                  experiment_dir_names=args.experiment_dir_names,
                  result_dir=result_dir)

    pm.execute_notebook(input_path, output_path, parameters=params)


if __name__ == '__main__':
    main()
