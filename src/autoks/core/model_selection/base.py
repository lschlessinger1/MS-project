import importlib
from abc import ABC
from collections import namedtuple
from pathlib import Path
from time import time
from typing import List, Tuple, Optional, Callable, Union

import numpy as np
from sklearn.utils import check_X_y, check_array
from tqdm import tqdm

from src.autoks import callbacks as cbks
from src.autoks.backend.model import RawGPModelType
from src.autoks.callbacks import ModelSearchLogger, CallbackList
from src.autoks.core.covariance import Covariance
from src.autoks.core.fitness_functions import negative_bic, log_likelihood_normalized
from src.autoks.core.gp_model import GPModel, pretty_print_gp_models
from src.autoks.core.gp_model_population import GPModelPopulation, ActiveModelPopulation
from src.autoks.core.grammar import BaseGrammar
from src.autoks.core.hyperprior import PriorsMap
from src.autoks.core.prior import PriorDist
from src.autoks.core.query_strategy import QueryStrategy, NaiveQueryStrategy
from src.autoks.preprocessing import standardize
from src.evalg.selection import TruncationSelector
from src.evalg.serialization import Serializable

DIRNAME = Path(__file__).parents[1].resolve() / 'best_model'

FitnessFnInfo = namedtuple("FitnessFnInfo", 'fn aka')

_FITNESS_FUNCTIONS = {
    'nbic': FitnessFnInfo(fn=negative_bic, aka=['nbic', 'negativebic']),
    'loglikn': FitnessFnInfo(fn=log_likelihood_normalized, aka=['loglikn', 'lmlnorm', 'lln'])
}

_FITNESS_FUNCTION_ALIAS = dict((alias, name)
                               for name, info in _FITNESS_FUNCTIONS.items()
                               for alias in info.aka)


class ModelSelector(Serializable):
    """Abstract base class for model selectors."""
    selected_models: List[GPModel]
    n_evals: int
    total_eval_time: int
    total_expansion_time: int
    total_model_search_time: int
    model_dict: dict
    visited: set

    def __init__(self,
                 grammar: BaseGrammar,
                 fitness_fn: Union[str, Callable[[RawGPModelType], float]],
                 n_parents: int = 1,
                 additive_form: bool = False,
                 gp_fn: Union[str, Callable] = 'gp_regression',
                 gp_args: Optional[dict] = None,
                 optimizer: Optional[str] = None,
                 n_restarts_optimizer: int = 10,
                 standardize_x: bool = True,
                 standardize_y: bool = True):
        self.grammar = grammar

        # Set fitness function.
        if callable(fitness_fn):
            f_str = getattr(fitness_fn, '__name__', 'Unknown')
            self.fitness_fn_name = _FITNESS_FUNCTION_ALIAS.get(f_str, None)
            self.fitness_fn = fitness_fn
        elif isinstance(fitness_fn, str):
            f_str = fitness_fn.lower()
            self.fitness_fn_name = _FITNESS_FUNCTION_ALIAS.get(f_str, None)
            if self.fitness_fn_name is not None:
                # Get fitness function by name.
                fitness_fn_info = _FITNESS_FUNCTIONS[self.fitness_fn_name]
                self.fitness_fn = fitness_fn_info.fn
            else:
                raise ValueError(f'Unknown fitness function: {f_str}')
        else:
            raise TypeError(f'Fitness function must be a callable or string. Found {fitness_fn.__class__.__name__}')

        self.n_parents = n_parents

        self.n_evals = 0
        self.additive_form = additive_form
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.standardize_x = standardize_x
        self.standardize_y = standardize_y

        self.total_eval_time = 0
        self.total_expansion_time = 0
        self.total_model_search_time = 0

        # Build default model dict of GP.
        self._gp_args = gp_args or {}
        if grammar.hyperpriors is not None:
            if 'GP' in grammar.hyperpriors:
                self._gp_args = self._gp_args or {}
                # Set likelihood hyperpriors.
                self._gp_args["likelihood_hyperprior"] = grammar.hyperpriors['GP']

        # Set GP function.
        if callable(gp_fn):
            gp_str = getattr(fitness_fn, '__name__', 'Unknown')
            self._gp_fn_name = gp_str
        elif isinstance(gp_fn, str):
            self._gp_fn_name = gp_fn
            gp_models_module = importlib.import_module('src.autoks.core.gp_models')
            gp_fn = getattr(gp_models_module, gp_fn)
        else:
            raise TypeError(f'GP function must be a callable or string. Found {gp_fn}')

        self.model_dict = gp_fn(**self._gp_args)

        self.name = f'{self.__class__.__name__}_{self._gp_fn_name}'

        # Visited set of all evaluated kernel expressions.
        self.visited = set()

        self.built = False

        self.selected_models = []
        self._x_train = None
        self._x_train_mean = None
        self._x_train_std = None
        self._y_train = None
        self._y_train_mean = None
        self._y_train_std = None

    @property
    def best_model_filename(self) -> str:
        """File name of the best model found in the model search."""
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_best_model')

    def train(self,
              x: np.ndarray,
              y: np.ndarray,
              eval_budget: int = 50,
              max_generations: Optional[int] = None,
              verbose: int = 1,
              callbacks: Optional = None,
              callback_metrics: Optional[List[str]] = None):
        """Train the model selector.

        :param x: Input data.
        :param y: Target data.
        :param eval_budget: Number of model evaluations (budget).
        :param max_generations: Maximum number of generations.
        :param verbose: Integer. 0, 1, 2, or 3. Verbosity mode.
        :param callbacks: List of callbacks to be called during training.
        :param callback_metrics: List of the display names of the metrics passed
        to the callbacks.
        :return:
        """
        x, y = check_X_y(x, y, multi_output=True, y_numeric=True)

        x, y = self._prepare_data(x, y)  # create grammar and standardize data
        self._x_train, self._y_train = x, y

        if max_generations is None:
            # By default, the model search is terminated only when the evaluation budget is expended.
            max_iterations = 1000
            max_generations = max(max_iterations, eval_budget * 2)
        else:
            max_generations = max_generations

        self.history = cbks.History()
        ms_logger = ModelSearchLogger()
        ms_logger.set_stat_book_collection(self.grammar.base_kernel_names)
        _callbacks = [cbks.BaseLogger()]
        _callbacks += (callbacks or []) + [self.history] + [ms_logger]

        callbacks = cbks.CallbackList(_callbacks)
        callback_model = self._get_callback_model()
        callbacks.set_model(callback_model)
        callbacks.set_params({
            'eval_budget': eval_budget,
            'max_generations': max_generations,
            'verbose': verbose,
            'metrics': callback_metrics or [],
        })

        # Progress bar
        if verbose:
            self.pbar = tqdm(total=eval_budget, unit='ev', desc='Model Evaluations')
            self.pbar.set_postfix(best_so_far=float('-inf'))

        # Perform model search.
        callbacks.on_train_begin()
        t_init = time()
        population = self._train(eval_budget=eval_budget, max_generations=max_generations, verbose=verbose,
                                 callbacks=callbacks)
        self.total_model_search_time += time() - t_init
        callbacks.on_train_end()

        if verbose:
            self.pbar.close()

        self.selected_models = [model for model in population.models if model.evaluated]
        return ms_logger  # self.history

    def _prepare_data(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input and output data for model search."""
        if not self.built:
            self.grammar.build(x.shape[1])
            self.built = True

        if self.standardize_x:
            self._x_train_mean = np.mean(x, axis=0)
            self._x_train_std = np.std(x, ddof=0, axis=0)
            x = standardize(x, self._x_train_mean, self._x_train_std)
        else:
            self._x_train_mean = np.zeros(x.shape[1])
            self._x_train_std = np.ones(x.shape[1])

        if self.standardize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = np.std(y, ddof=0, axis=0)
            y = standardize(y, self._y_train_mean, self._y_train_std)
        else:
            self._y_train_mean = np.zeros(y.shape[1])
            self._y_train_std = np.ones(y.shape[1])

        return x, y

    def _train(self,
               eval_budget: int,
               max_generations: int,
               callbacks: CallbackList,
               verbose: int = 1) -> GPModelPopulation:
        raise NotImplementedError

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Predict on test inputs."""
        # For now, just use best model to predict. In the future, model averaging should be used.
        x = check_array(x)
        best_gp_model = self.best_model()
        gp = best_gp_model.build_model(self._x_train, self._y_train)
        return gp.predict(x, **kwargs)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the model selector on test data."""
        x, y = check_X_y(x, y, multi_output=True, y_numeric=True)

        if self.standardize_x:
            x = standardize(x, self._x_train_mean, self._x_train_std)

        if self.standardize_y:
            y = standardize(y, self._y_train_mean, self._y_train_std)

        best_gp_model = self.best_model()

        return self.fitness_fn(best_gp_model.build_model(x, y))

    def _get_initial_candidate_covariances(self) -> List[Covariance]:
        """Get the initial set of candidate covariances."""
        raise NotImplementedError

    def _get_initial_candidates(self) -> List[GPModel]:
        """Get initial set of candidate GP models."""
        initial_covariances = self._get_initial_candidate_covariances()
        return self._covariances_to_gp_models(initial_covariances)

    def _initialize(self,
                    eval_budget: int,
                    callbacks: CallbackList,
                    verbose: int = 0) -> ActiveModelPopulation:
        """Initialize models."""
        population = ActiveModelPopulation()

        initial_candidates = self._get_initial_candidates()

        population.update(initial_candidates)

        if verbose == 3:
            pretty_print_gp_models(population.models, 'Initial candidate')

        self._evaluate_models(population.candidates(), eval_budget, callbacks=callbacks, verbose=verbose)

        return population

    def _select_parents(self,
                        population: ActiveModelPopulation) -> List[GPModel]:
        """Select parents to expand.

        By default, choose top k models.
        """
        parent_selector = TruncationSelector(self.n_parents)
        return list(parent_selector.select(np.array(population.models), np.array(population.fitness_scores())).tolist())

    def _propose_new_models(self,
                            population: ActiveModelPopulation,
                            callbacks: CallbackList,
                            verbose: int = 0) -> List[GPModel]:
        """Propose new models using the grammar.

        :param population:
        :param verbose:
        :return:
        """
        callbacks.on_propose_new_models_begin()
        parents = self._select_parents(population)
        t0_exp = time()
        new_covariances = self.grammar.get_candidates(parents, verbose=verbose)
        self.total_expansion_time += time() - t0_exp
        new_models = self._covariances_to_gp_models(new_covariances)

        new_model_logs = {
            'gp_models': new_models,
            'x': self._x_train
        }
        callbacks.on_propose_new_models_end(new_model_logs)

        return new_models

    def _evaluate_models(self,
                         models: List[GPModel],
                         eval_budget: int,
                         callbacks: CallbackList,
                         verbose: int = 0) -> List[GPModel]:
        """Evaluate a set models on some training data.

        :param models:
        :param eval_budget:
        :param verbose:
        :return:
        """
        callbacks.on_evaluate_all_begin(logs={'gp_models': models})
        evaluated_models = []

        for gp_model in models:
            callbacks.on_evaluate_begin(logs={'gp_model': gp_model})

            if self.n_evals >= eval_budget:
                if verbose == 3:
                    print('Stopping optimization and evaluation. Evaluation budget reached.\n')
                break
            elif gp_model.covariance.symbolic_expr_expanded in self.visited:
                if verbose == 3:
                    print('Skipping model because it was previously evaluated')
                    gp_model.covariance.pretty_print()
                    print()
                continue

            if not gp_model.evaluated:
                gp_model = self._evaluate_model(gp_model, verbose=verbose)

            evaluated_models.append(gp_model)

            eval_logs = {
                'gp_model': gp_model,
                'x': self._x_train
            }
            callbacks.on_evaluate_end(eval_logs)

        eval_all_logs = {
            'gp_models': evaluated_models,
            'x': self._x_train
        }
        callbacks.on_evaluate_all_end(eval_all_logs)

        if verbose == 3:
            print('Printing all results')
            # Sort models by scores with un-evaluated models last
            for gp_model in sorted(evaluated_models, key=lambda m: (m.score is not None, m.score), reverse=True):
                gp_model.covariance.pretty_print()
                print('\tfitness =', gp_model.score)
            print('')

        return evaluated_models

    def _evaluate_model(self,
                        model: GPModel,
                        verbose: int = 0) -> GPModel:
        """Evaluate a single model on some training data.

        :param model:
        :param verbose:
        :return:
        """
        t0 = time()
        model.score_model(self._x_train, self._y_train, self.fitness_fn, optimizer=self.optimizer,
                          n_restarts=self.n_restarts_optimizer)
        self.total_eval_time += time() - t0
        self.n_evals += 1
        self.visited.add(model.covariance.symbolic_expr_expanded)
        if verbose:
            self.pbar.update()

        return model

    def best_model(self) -> GPModel:
        """Get the best scoring model."""
        evaluated_gp_models = [model for model in self.selected_models if model.evaluated]
        sorted_gp_models = sorted(evaluated_gp_models, key=lambda m: m.score, reverse=True)

        # Set model dict if needed
        for gp_model in sorted_gp_models:
            if gp_model.model_input_dict != self.model_dict:
                gp_model.model_input_dict = self.model_dict

        return sorted_gp_models[0]

    def load_best_model(self) -> GPModel:
        return GPModel.load(self.best_model_filename)

    def save_best_model(self):
        return self.best_model().save(self.best_model_filename)

    def to_dict(self) -> dict:
        input_dict = super().to_dict()

        input_dict['name'] = self.name
        input_dict["grammar"] = self.grammar.to_dict()
        input_dict["fitness_fn"] = self.fitness_fn_name  # save fitness function name instead of function itself.
        input_dict['n_parents'] = self.n_parents
        input_dict['n_evals'] = self.n_evals
        input_dict['additive_form'] = self.additive_form
        input_dict['optimizer'] = self.optimizer
        input_dict['n_restarts_optimizer'] = self.n_restarts_optimizer
        input_dict['standardize_x'] = self.standardize_x
        input_dict['standardize_y'] = self.standardize_y

        input_dict['total_eval_time'] = self.total_eval_time
        input_dict['total_expansion_time'] = self.total_expansion_time
        input_dict['total_model_search_time'] = self.total_model_search_time

        input_dict['gp_fn_name'] = self._gp_fn_name
        input_dict['gp_args'] = self._gp_args.copy()
        if 'likelihood_hyperprior' in self._gp_args:
            likelihood_hyperprior_map = {}
            for param_name, prior in input_dict['gp_args']['likelihood_hyperprior'].items():
                likelihood_hyperprior_map[param_name] = prior.to_dict()
            input_dict['gp_args']['likelihood_hyperprior'] = likelihood_hyperprior_map

        input_dict['built'] = self.built
        input_dict['selected_models'] = [m.to_dict() for m in self.selected_models]

        input_dict['_x_train_mean'] = None if self._x_train_mean is None else self._x_train_mean.tolist()
        input_dict['_x_train_std'] = None if self._x_train_std is None else self._x_train_std.tolist()
        input_dict['_y_train_mean'] = None if self._y_train_mean is None else self._y_train_mean.tolist()
        input_dict['_y_train_std'] = None if self._y_train_std is None else self._y_train_std.tolist()

        return input_dict

    @classmethod
    def _build_from_input_dict(cls, input_dict: dict):
        n_evals = input_dict.pop('n_evals')
        total_eval_time = input_dict.pop('total_eval_time')
        total_expansion_time = input_dict.pop('total_expansion_time')
        total_model_search_time = input_dict.pop('total_model_search_time')
        name = input_dict.pop('name')
        built = input_dict.pop('built')
        selected_models = input_dict.pop('selected_models')
        x_train_mean = input_dict.pop('_x_train_mean')
        x_train_std = input_dict.pop('_x_train_std')
        y_train_mean = input_dict.pop('_y_train_mean')
        y_train_std = input_dict.pop('_y_train_std')

        model_selector = super()._build_from_input_dict(input_dict)

        model_selector.n_evals = n_evals
        model_selector.total_eval_time = total_eval_time
        model_selector.total_expansion_time = total_expansion_time
        model_selector.total_model_search_time = total_model_search_time
        model_selector.name = name
        model_selector.built = built
        model_selector.selected_models = [GPModel.from_dict(m) for m in selected_models]
        model_selector._x_train_mean = None if x_train_mean is None else np.array(x_train_mean)
        model_selector._x_train_std = None if x_train_std is None else np.array(x_train_std)
        model_selector._y_train_mean = None if y_train_mean is None else np.array(y_train_mean)
        model_selector._y_train_std = None if y_train_std is None else np.array(y_train_std)

        return model_selector

    @classmethod
    def _format_input_dict(cls, input_dict: dict) -> dict:
        input_dict = super()._format_input_dict(input_dict)
        input_dict['grammar'] = BaseGrammar.from_dict(input_dict['grammar'])
        gp_fn_name = input_dict.pop('gp_fn_name')
        input_dict['gp_fn'] = gp_fn_name
        if 'gp_args' in input_dict and 'likelihood_hyperprior' in input_dict['gp_args']:
            for param_name, prior in input_dict['gp_args']['likelihood_hyperprior'].items():
                input_dict['gp_args']['likelihood_hyperprior'][param_name] = PriorDist.from_dict(prior)

        return input_dict

    def _get_callback_model(self):
        """Returns the Callback Model for this Model."""
        if hasattr(self, 'callback_model') and self.callback_model:
            return self.callback_model
        return self

    def get_timing_report(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Get the runtime report of the model search.

        :return:
        """
        eval_time = self.total_eval_time
        expansion_time = self.total_expansion_time
        total_time = self.total_model_search_time
        other_time = total_time - eval_time - expansion_time

        labels = ['Evaluation', 'Expansion', 'Other']
        x = np.array([eval_time, expansion_time, other_time])
        x_pct = 100 * (x / total_time)

        return labels, x, x_pct

    def _print_search_summary(self,
                              depth: int,
                              population: ActiveModelPopulation,
                              eval_budget: int,
                              max_generations: int,
                              verbose: int = 0) -> None:
        """Print a summary of the model population at a given generation."""
        best_objective = population.best_fitness()
        if verbose >= 2:
            print()
            best_kernel = population.best_model()
            sizes = population.sizes()
            print(f'Iteration {depth}/{max_generations}')
            print(f'Evaluated {self.n_evals}/{eval_budget}')
            print(f'Avg. fitness = %0.6f' % population.mean_fitness())
            print(f'Best fitness = %.6f' % best_objective)
            print(f'Avg. size = %.2f' % np.mean(sizes))
            print('Best kernel:')
            best_kernel.covariance.pretty_print()
            print('')
        elif verbose == 1:
            self.pbar.set_postfix(best_so_far=best_objective)

    def _covariances_to_gp_models(self, covariances: List[Covariance]) -> List[GPModel]:
        """Convert covariances to GP models."""
        return [self._covariance_to_gp_model(cov) for cov in covariances]

    def _covariance_to_gp_model(self, cov: Covariance) -> GPModel:
        """Convert a covariance to a GP model."""
        gp_model = GPModel(cov)

        # Set model dict
        gp_model.model_input_dict = self.model_dict
        gp_model.likelihood = self.model_dict["likelihood"].copy()

        # Convert to additive form if necessary
        if self.additive_form:
            gp_model.covariance = gp_model.covariance.to_additive_form()

        return gp_model


class SurrogateBasedModelSelector(ModelSelector, ABC):

    def __init__(self,
                 grammar: BaseGrammar,
                 fitness_fn: Union[str, Callable[[RawGPModelType], float]] = 'loglikn',
                 query_strategy: Optional[QueryStrategy] = None,
                 n_parents: int = 1,
                 additive_form: bool = False,
                 gp_fn: Union[str, Callable] = 'gp_regression',
                 gp_args: Optional[dict] = None,
                 optimizer: Optional[str] = None,
                 n_restarts_optimizer: int = 10):
        super().__init__(grammar, fitness_fn, n_parents, additive_form, gp_fn, gp_args, optimizer, n_restarts_optimizer)

        if query_strategy is not None:
            self.query_strategy = query_strategy
        else:
            self.query_strategy = NaiveQueryStrategy()

        self.total_query_time = 0

    def query_models(self,
                     population: GPModelPopulation,
                     query_strategy: QueryStrategy,
                     x_train,
                     y_train,
                     hyperpriors: Optional[PriorsMap] = None,
                     verbose: int = 0) \
            -> Tuple[List[GPModel], List[int], List[float]]:
        """Select gp_models using the acquisition function of the query strategy.

        :param population:
        :param query_strategy:
        :param hyperpriors:
        :param verbose: Verbosity mode, 0 or 1.
        :return:
        """
        t0 = time()
        kernels = population.models
        unevaluated_kernels_ind = [i for (i, kernel) in enumerate(kernels) if not kernel.evaluated]
        unevaluated_kernels = [kernels[i] for i in unevaluated_kernels_ind]
        ind, acq_scores = query_strategy.query(unevaluated_kernels_ind, kernels, x_train, y_train,
                                               hyperpriors, None)
        selected_kernels = query_strategy.select(np.array(unevaluated_kernels), np.array(acq_scores))
        self.total_query_time += time() - t0

        if verbose:
            n_selected = len(ind)
            plural_suffix = '' if n_selected == 1 else 's'
            print(f'Query strategy selected {n_selected} kernel{plural_suffix}:')

            for kern in selected_kernels:
                kern.covariance.pretty_print()
            print('')

        return selected_kernels, ind, acq_scores

    def get_timing_report(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Get the runtime report of the kernel search.

        :return:
        """
        eval_time = self.total_eval_time
        expansion_time = self.total_expansion_time
        query_time = self.total_query_time
        total_time = self.total_model_search_time
        other_time = total_time - eval_time - expansion_time - query_time

        labels = ['Evaluation', 'Expansion', 'Query', 'Other']
        x = np.array([eval_time, expansion_time, query_time, other_time])
        x_pct = 100 * (x / total_time)

        return labels, x, x_pct

    def __str__(self):
        return f'{self.name}'f'(grammar={self.grammar.__class__.__name__}, ' \
               f'fitness_fn={self.fitness_fn_name})'

    def __repr__(self):
        return f'{self.__class__.__name__}'f'(grammar={self.grammar.__class__.__name__}, fitness_fn={self.fitness_fn_name})'
