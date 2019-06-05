from abc import ABC
from time import time
from typing import List, Tuple, Optional, Callable

import numpy as np
from GPy import likelihoods
from GPy.inference.latent_function_inference import Laplace
from GPy.likelihoods import Likelihood
from tqdm import tqdm

from src.autoks.backend.kernel import set_priors
from src.autoks.backend.model import RawGPModelType
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model import GPModel, pretty_print_gp_models
from src.autoks.core.gp_model_population import GPModelPopulation, ActiveModelPopulation
from src.autoks.core.grammar import BaseGrammar
from src.autoks.core.hyperprior import Hyperpriors
from src.autoks.core.query_strategy import QueryStrategy, NaiveQueryStrategy
from src.evalg.selection import TruncationSelector


class ModelSelector:
    """Abstract base class for model selectors"""
    selected_models: List[GPModel]
    n_evals: int
    total_eval_time: int
    total_expansion_time: int
    total_model_search_time: int
    default_likelihood: Likelihood
    model_dict: dict
    visited: set

    def __init__(self,
                 grammar: BaseGrammar,
                 objective: Callable[[RawGPModelType], float],
                 eval_budget: int = 50,
                 max_generations: Optional[int] = None,
                 n_parents: int = 1,
                 additive_form: bool = False,
                 debug: bool = False,
                 verbose: bool = False,
                 optimizer: Optional[str] = None,
                 n_restarts_optimizer: int = 10,
                 use_laplace: bool = True,
                 active_set_callback: Optional[Callable] = None,
                 eval_callback: Optional[Callable] = None,
                 expansion_callback: Optional[Callable] = None):
        self.grammar = grammar
        self.objective = objective

        self.eval_budget = eval_budget  # number of model evaluations (budget)

        if max_generations is None:
            # By default, the model search is terminated only when the evaluation budget is expended.
            max_iterations = 1000
            self.max_generations = max(max_iterations, eval_budget * 2)
        else:
            self.max_generations = max_generations

        self.n_parents = n_parents

        self.n_evals = 0
        self.additive_form = additive_form
        self.debug = debug
        self.verbose = verbose
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer

        self.total_eval_time = 0
        self.total_expansion_time = 0
        self.total_model_search_time = 0

        # build default model dict of GP (assuming GP Regression)
        default_likelihood = likelihoods.Gaussian()
        if self.grammar.hyperpriors is not None:
            # set likelihood hyperpriors
            likelihood_priors = self.grammar.hyperpriors['GP']
            default_likelihood = set_priors(default_likelihood, likelihood_priors)

        default_model_dict = dict()
        self.default_likelihood = default_likelihood

        if use_laplace:
            default_model_dict['inference_method'] = Laplace()

        self.model_dict = default_model_dict

        # visited set of all expanded kernel expressions previously evaluated
        self.visited = set()

        # Callbacks
        def do_nothing(*args, **kwargs):
            pass

        if active_set_callback is None:
            active_set_callback = do_nothing
        self.active_set_callback = active_set_callback

        if eval_callback is None:
            eval_callback = do_nothing
        self.eval_callback = eval_callback

        if expansion_callback is None:
            expansion_callback = do_nothing
        self.expansion_callback = expansion_callback

        # Progress bar
        if not self.debug:
            self.pbar = tqdm(total=self.eval_budget, unit='ev', desc='Model Evaluations')

    def train(self,
              x: np.ndarray,
              y: np.ndarray):
        """Train the model selector."""
        t_init = time()
        population = self._train(x, y)
        if not self.debug:
            self.pbar.close()
        self.total_model_search_time += time() - t_init
        self.selected_models = population.models
        return self

    def _train(self, x: np.ndarray, y: np.ndarray) -> GPModelPopulation:
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict on test inputs."""
        # for now, just use best model to predict. Should be using model averaging...
        # sort selected models by scores
        pass

    def score(self,
              x: np.ndarray,
              y: np.ndarray) -> float:
        """Score the model selector on test data."""
        pass

    def get_initial_candidate_covariances(self) -> List[Covariance]:
        """Get the initial set of candidate covariances."""
        raise NotImplementedError

    def get_initial_candidates(self) -> List[GPModel]:
        """Get initial set of candidate GP models."""
        initial_covariances = self.get_initial_candidate_covariances()
        return self._covariances_to_gp_models(initial_covariances)

    def initialize(self,
                   x: np.ndarray,
                   y: np.ndarray) -> ActiveModelPopulation:
        """Initialize models."""
        population = ActiveModelPopulation()

        initial_candidates = self.get_initial_candidates()

        population.update(initial_candidates)

        if self.debug:
            pretty_print_gp_models(population.models, 'Initial candidate')

        self.evaluate_models(population.candidates(), x, y)

        return population

    def select_parents(self, population: ActiveModelPopulation) -> List[GPModel]:
        """Select parents to expand.

        By default, choose top k models.
        """
        parent_selector = TruncationSelector(self.n_parents)
        return list(parent_selector.select(np.array(population.models), np.array(population.objectives())).tolist())

    def propose_new_models(self, population: ActiveModelPopulation) -> List[GPModel]:
        """Propose new models using the grammar.

        :param population:
        :return:
        """
        parents = self.select_parents(population)
        t0_exp = time()
        new_covariances = self.grammar.get_candidates(parents, verbose=self.debug)
        self.total_expansion_time += time() - t0_exp

        return self._covariances_to_gp_models(new_covariances)

    def evaluate_models(self,
                        models: List[GPModel],
                        x: np.ndarray,
                        y: np.ndarray) -> List[GPModel]:
        """Evaluate a set models on some training data.

        :param models:
        :param x:
        :param y:
        :return:
        """
        evaluated_models = []

        for gp_model in models:
            if self.n_evals >= self.eval_budget:
                if self.debug:
                    print('Stopping optimization and evaluation. Evaluation budget reached.\n')
                break
            elif gp_model.covariance.symbolic_expr_expanded in self.visited:
                if self.debug:
                    print('Skipping model because it was previously evaluated')
                    gp_model.covariance.pretty_print()
                    print()
                continue

            if not gp_model.evaluated:
                t0 = time()
                gp_model.score_model(x, y, self.objective, optimizer=self.optimizer,
                                     n_restarts=self.n_restarts_optimizer)
                self.total_eval_time += time() - t0
                self.n_evals += 1
                self.visited.add(gp_model.covariance.symbolic_expr_expanded)
                if not self.debug:
                    self.pbar.update()

            evaluated_models.append(gp_model)
            self.eval_callback([gp_model], self, x, y)

        if self.debug:
            print('Printing all results')
            # Sort models by scores with un-evaluated models last
            for gp_model in sorted(evaluated_models, key=lambda m: (m.score is not None, m.score), reverse=True):
                gp_model.covariance.pretty_print()
                print('\tobjective =', gp_model.score)
            print('')
        return evaluated_models

    def best_model(self) -> GPModel:
        """Get the best scoring model."""
        evaluated_gp_models = [model for model in self.selected_models if model.evaluated]
        sorted_gp_models = sorted(evaluated_gp_models, key=lambda m: m.score, reverse=True)
        return sorted_gp_models[0]

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
                              population: ActiveModelPopulation) -> None:
        """Print a summary of the model population at a given generation."""
        best_objective = population.best_objective()
        if self.verbose:
            print()
            best_kernel = population.best_model()
            sizes = population.sizes()
            print(f'Iteration {depth}/{self.max_generations}')
            print(f'Evaluated {self.n_evals}/{self.eval_budget}')
            print(f'Avg. objective = %0.6f' % population.avg_objective())
            print(f'Best objective = %.6f' % best_objective)
            print(f'Avg. size = %.2f' % np.mean(sizes))
            print('Best kernel:')
            best_kernel.covariance.pretty_print()
            print('')
        else:
            print()
            print('Evaluated %d: best-so-far = %.5f' % (self.n_evals, best_objective))

    def _covariances_to_gp_models(self, covariances: List[Covariance]) -> List[GPModel]:
        """Convert covariances to GP models."""
        return [self._covariance_to_gp_model(cov) for cov in covariances]

    def _covariance_to_gp_model(self, cov: Covariance) -> GPModel:
        """Convert a covariance to a GP model."""
        gp_model = GPModel(cov)

        # Set model dict
        gp_model.model_input_dict = self.model_dict
        gp_model.likelihood = self.default_likelihood

        # Convert to additive form if necessary
        if self.additive_form:
            gp_model.covariance = gp_model.covariance.to_additive_form()

        return gp_model


class SurrogateBasedModelSelector(ModelSelector, ABC):

    def __init__(self, grammar, objective, query_strategy=None, eval_budget=50, max_generations=None, n_parents=1,
                 additive_form=False, debug=False, verbose=False, optimizer=None, n_restarts_optimizer=10,
                 use_laplace=True, active_set_callback=None, eval_callback=None, expansion_callback=None):
        super().__init__(grammar, objective, eval_budget, max_generations, n_parents, additive_form, debug,
                         verbose, optimizer, n_restarts_optimizer, use_laplace, active_set_callback, eval_callback,
                         expansion_callback)

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
                     hyperpriors: Optional[Hyperpriors] = None) \
            -> Tuple[List[GPModel], List[int], List[float]]:
        """Select gp_models using the acquisition function of the query strategy.

        :param population:
        :param query_strategy:
        :param hyperpriors:
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

        if self.debug:
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
