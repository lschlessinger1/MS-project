"""Function to train a model."""
import warnings
from time import time
from typing import Tuple

from src.autoks.callbacks import ModelSearchLogger
from src.autoks.core.model_selection.base import ModelSelector
from src.datasets.dataset import Dataset

HIDE_WARNINGS = True


def train_model(
        model: ModelSelector,
        dataset: Dataset,
        eval_budget: int,
        verbose: int) -> Tuple[ModelSelector, ModelSearchLogger]:
    """Train model."""
    callbacks = []

    print(model.model_dict)  # summarize GP

    with warnings.catch_warnings():
        if HIDE_WARNINGS:
            warnings.simplefilter("ignore")
        t = time()
        _history = model.train(dataset.x, dataset.y, eval_budget=eval_budget, verbose=verbose, callbacks=callbacks)
        print(f'Training took {time() - t:,.2f}s')

    return model, _history
