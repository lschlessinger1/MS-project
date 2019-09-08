from collections import defaultdict, deque
from typing import Optional, Iterable, List

from src.autoks.statistics import StatBookCollection, Statistic
from src.autoks.tracking import get_best_n_operands, get_model_scores, get_best_n_hyperparams, get_cov_dists, \
    get_diversity_scores, base_kern_freq, get_n_hyperparams, get_n_operands, update_stat_book
from src.evalg.serialization import Serializable


# Adapted from Keras' callbacks

class Callback:

    def __init__(self):
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_generation_begin(self, gen: int, logs: Optional[dict] = None):
        """Called at the start of a generation."""

    def on_generation_end(self, gen: int, logs: Optional[dict] = None):
        """Called at the end of a generation."""

    def on_evaluate_all_begin(self, logs: Optional[dict] = None):
        """Called at the start of a call to `evaluate_models`."""

    def on_evaluate_all_end(self, logs: Optional[dict] = None):
        """Called at the end of a call to `evaluate_models`."""

    def on_evaluate_begin(self, logs: Optional[dict] = None):
        """Called before evaluating a single model."""

    def on_evaluate_end(self, logs: Optional[dict] = None):
        """Called after evaluating a single model."""

    def on_train_begin(self, logs: Optional[dict] = None):
        """Called at the beginning of training."""

    def on_train_end(self, logs: Optional[dict] = None):
        """Called at the end of training."""

    def on_propose_new_models_begin(self, logs: Optional[dict] = None):
        """Called at the beginning of new model proposals."""

    def on_propose_new_models_end(self, logs: Optional[dict] = None):
        """Called at the end of new model proposals."""


class CallbackList:

    def __init__(self,
                 callbacks: Optional[Iterable[Callback]] = None,
                 queue_length: int = 10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length
        self.params = {}
        self.model = None
        self._reset_model_eval_timing()

    def _reset_model_eval_timing(self):
        self._delta_t_model_eval = 0.
        self._delta_ts = defaultdict(lambda: deque([], maxlen=self.queue_length))

    def append(self, callback: Callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def on_generation_begin(self, generation: int, logs: Optional[dict] = None):
        """Calls the `on_generation_begin` methods of its callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_generation_begin(generation, logs=logs)
        self._reset_model_eval_timing()

    def on_generation_end(self, generation: int, logs: Optional[dict] = None):
        """Calls the `on_generation_end` methods of its callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_generation_end(generation, logs=logs)

    def on_train_begin(self, logs: Optional[dict] = None):
        """Calls the `on_train_begin` methods of its callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs=logs)

    def on_train_end(self, logs: Optional[dict] = None):
        """Calls the `on_train_end` methods of its callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs=logs)

    def on_evaluate_all_begin(self, logs: Optional[dict] = None):
        """Calls the `on_evaluate_all_begin` methods of its callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_evaluate_all_begin(logs=logs)

    def on_evaluate_all_end(self, logs: Optional[dict] = None):
        """Calls the `on_evaluate_all_end` methods of its callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_evaluate_all_end(logs=logs)

    def on_evaluate_begin(self, logs: Optional[dict] = None):
        """Calls the `on_evaluate_begin` methods of its callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_evaluate_begin(logs=logs)

    def on_evaluate_end(self, logs: Optional[dict] = None):
        """Calls the `on_evaluate_end` methods of its callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_evaluate_end(logs=logs)

    def on_propose_new_models_begin(self, logs: Optional[dict] = None):
        """Calls the `on_propose_new_models_begin` methods of its callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_propose_new_models_begin(logs=logs)

    def on_propose_new_models_end(self, logs: Optional[dict] = None):
        """Calls the `on_propose_new_models_end` methods of its callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_propose_new_models_end(logs=logs)

    def __iter__(self):
        return iter(self.callbacks)


class BaseLogger(Callback):
    """Callback that accumulates generational averages of metrics.

    This callback is automatically applied to every model selector.

    Attributes:
        stateful_metrics: An optional iterable of string names of metrics
        that should *not* be averaged over an epoch.
        Metrics in this list will be logged as-is in `on_generation_end`.
        All others will be averaged in `on_generation_end`.
    """

    def __init__(self, stateful_metrics: Optional[Iterable[str]] = None):
        super().__init__()

        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self.seen = 0
        self.totals = {}

    def on_generation_begin(self, gen: int, logs: Optional[dict] = None):
        self.seen = 0
        self.totals = {}

    def on_evaluate_all_end(self, logs: Optional[dict] = None):
        logs = logs or {}
        model_group_size = logs.get('size', 0)
        self.seen += model_group_size

        for k, v in logs.items():
            if k in self.stateful_metrics:
                self.totals[k] = v
            else:
                if k in self.totals:
                    self.totals[k] += v * model_group_size
                else:
                    self.totals[k] = v * model_group_size

    def on_generation_end(self, gen: int, logs: Optional[dict] = None):
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    # Make value available to next callbacks.
                    if k in self.stateful_metrics:
                        logs[k] = self.totals[k]
                    else:
                        logs[k] = self.totals[k] / self.seen


class History(Callback):
    """Callback that records events into a `History` object.
    
    This callback is automatically applied to
    every model selector. The `History` object
    gets returned by the `train` method of model selectors.
    """

    def on_train_begin(self, logs: Optional[dict] = None):
        self.generation = []
        self.history = {}

    def on_generation_end(self, generation: int, logs: Optional[dict] = None):
        logs = logs or {}
        self.generation.append(generation)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class ModelSearchLogger(Callback, Serializable):

    def __init__(self):
        super().__init__()
        # statistics used for plotting
        self.n_hyperparams_name = 'n_hyperparameters'
        self.n_operands_name = 'n_operands'
        self.score_name = 'score'
        self.cov_dists_name = 'cov_dists'
        self.diversity_scores_name = 'diversity_scores'
        self.best_stat_name = 'best'

        # separate these!
        self.evaluations_name = 'evaluations'
        self.active_set_name = 'active_set'
        self.expansion_name = 'expansion'
        self.stat_book_names = [self.evaluations_name, self.expansion_name, self.active_set_name]
        self.base_kern_freq_names = []

        self.stat_book_collection = StatBookCollection()

    def set_stat_book_collection(self, base_kernel_names: List[str]):
        self.base_kern_freq_names = [base_kern_name + '_frequency' for base_kern_name in base_kernel_names]

        # All stat books track these variables
        shared_multi_stat_names = [self.n_hyperparams_name, self.n_operands_name] + self.base_kern_freq_names

        # raw value statistics
        base_kern_stat_funcs = [base_kern_freq(base_kern_name) for base_kern_name in base_kernel_names]
        shared_stats = [get_n_hyperparams, get_n_operands] + base_kern_stat_funcs

        self.stat_book_collection.create_shared_stat_books(self.stat_book_names, shared_multi_stat_names, shared_stats)

        sb_active_set = self.stat_book_collection.stat_books[self.active_set_name]
        sb_active_set.add_raw_value_stat(self.score_name, get_model_scores)
        sb_active_set.add_raw_value_stat(self.cov_dists_name, get_cov_dists)
        sb_active_set.add_raw_value_stat(self.diversity_scores_name, get_diversity_scores)
        sb_active_set.multi_stats[self.n_hyperparams_name].add_statistic(Statistic(self.best_stat_name,
                                                                                   get_best_n_hyperparams))
        sb_active_set.multi_stats[self.n_operands_name].add_statistic(Statistic(self.best_stat_name,
                                                                                get_best_n_operands))

        sb_evals = self.stat_book_collection.stat_books[self.evaluations_name]
        sb_evals.add_raw_value_stat(self.score_name, get_model_scores)

    def on_evaluate_all_end(self, logs: Optional[dict] = None):
        logs = logs or {}
        models = logs.get('gp_models', [])
        x = logs.get('x', None)

        grammar = self.model.grammar
        stat_book = self.stat_book_collection.stat_books[self.active_set_name]
        if models:
            update_stat_book(stat_book, models, x, grammar.base_kernel_names, grammar.n_dims)

    def on_evaluate_end(self, logs: Optional[dict] = None):
        logs = logs or {}
        model = logs.get('gp_model', [])
        model = [model] or []
        x = logs.get('x', None)

        grammar = self.model.grammar
        stat_book = self.stat_book_collection.stat_books[self.evaluations_name]
        if model:
            update_stat_book(stat_book, model, x, grammar.base_kernel_names, grammar.n_dims)

    def on_propose_new_models_end(self, logs: Optional[dict] = None):
        logs = logs or {}
        models = logs.get('gp_models', [])
        x = logs.get('x', None)

        grammar = self.model.grammar
        stat_book = self.stat_book_collection.stat_books[self.expansion_name]
        if models:
            update_stat_book(stat_book, models, x, grammar.base_kernel_names, grammar.n_dims)

    def to_dict(self) -> dict:
        output_dict = super().to_dict()
        output_dict["stat_book_collection"] = self.stat_book_collection.to_dict()
        output_dict["base_kern_freq_names"] = self.base_kern_freq_names
        return output_dict

    @classmethod
    def _build_from_input_dict(cls, input_dict: dict):
        stat_book_collection = StatBookCollection.from_dict(input_dict.pop("stat_book_collection"))
        base_kern_freq_names = input_dict.pop("base_kern_freq_names")
        tracker = super()._build_from_input_dict(input_dict)
        tracker.stat_book_collection = stat_book_collection
        tracker.base_kern_freq_names = base_kern_freq_names
        return tracker

    def save(self, output_file_name: str):
        self.stat_book_collection.save(output_file_name)

    @staticmethod
    def load(output_file_name: str):
        mst = ModelSearchLogger()
        sbc = StatBookCollection.load(output_file_name)
        mst.stat_book_collection = sbc
        return mst


class GCPCallback(Callback, Serializable):
    """Google Cloud Platform (GCP) callback.

    TODO: Implement experiment saving on GCP.
    """

    def __init__(self):
        super().__init__()

    def on_train_end(self, logs: Optional[dict] = None):
        pass
