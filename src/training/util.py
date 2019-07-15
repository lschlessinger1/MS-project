"""Function to train a model."""
import warnings
from time import time
from typing import Tuple

from src.autoks.core.model_selection.base import ModelSelector
from src.autoks.tracking import ModelSearchTracker
from src.datasets.dataset import Dataset

HIDE_WARNINGS = True


def train_model(
        model: ModelSelector,
        dataset: Dataset,
        eval_budget: int,
        verbose: int) -> Tuple[ModelSelector, ModelSearchTracker]:
    """Train model."""

    print(model.model_dict)  # summarize GP
    tracker = ModelSearchTracker()

    with warnings.catch_warnings():
        if HIDE_WARNINGS:
            warnings.simplefilter("ignore")
        t = time()
        _tracker = model.train(dataset.x, dataset.y, eval_budget=eval_budget, verbose=verbose, tracker=tracker)
        print(f'Training took {time() - t:,.2f}s')

    return model, tracker
