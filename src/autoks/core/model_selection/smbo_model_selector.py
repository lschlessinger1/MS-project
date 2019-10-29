from typing import List, Union, Callable, Optional

from src.autoks.acquisition import expected_improvement
from src.autoks.backend.model import RawGPModelType
from src.autoks.callbacks import CallbackList
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model import GPModel, pretty_print_gp_models
from src.autoks.core.gp_model_population import GPModelPopulation, ActiveModelPopulation
from src.autoks.core.grammar import BaseGrammar
from src.autoks.core.hyperprior import boms_hyperpriors
from src.autoks.core.model_selection.base import ModelSelector
from src.autoks.core.prior import PriorDist
from src.autoks.core.strategies.bayes_opt_gp_strategy import BayesOptGPStrategy
from src.autoks.distance.distance import ActiveModels, CorrelationDistanceBuilder


class SMBOModelSelector(ModelSelector):
    """Sequential model-based optimization (SMBO)."""

    def __init__(self,
                 grammar: BaseGrammar = None,
                 fitness_fn: Union[str, Callable[[RawGPModelType], float]] = 'loglikn',
                 gp_fn: Union[str, Callable] = 'gp_regression',
                 gp_args: Optional[dict] = None,
                 **ms_args):
        self.eval_budget_low_level = ms_args.pop("eval_budget_low_level", 100)
        base_kernel_names = ms_args.get('base_kernel_names', None)
        grammar = BaseGrammar(base_kernel_names=base_kernel_names, hyperpriors=boms_hyperpriors())
        super().__init__(grammar, fitness_fn, gp_fn=gp_fn, gp_args=gp_args)
        self.gp_fn = gp_fn
        self.gp_args = gp_args
        self.model_selector_args = ms_args

        self.active_models = None
        self.acquisition_fn = expected_improvement
        self.max_n_models = 1000
        self.num_samples = 20
        self.max_n_hyperparameters = 100
        noise_prior_args = self.grammar.hyperpriors['GP']['variance']._raw_prior_args
        self.noise_prior = PriorDist.from_prior_str("GAUSSIAN", noise_prior_args).raw_prior
        # TODO: include strategy and distance builder string here

    def _train(self, eval_budget: int, max_generations: int, callbacks: CallbackList,
               verbose: int = 1) -> GPModelPopulation:
        population = self._initialize(eval_budget, verbose=verbose, callbacks=callbacks)

        depth = 0
        while self.n_evals < eval_budget:
            callbacks.on_generation_begin(generation=depth, logs={'gp_models': population.models})

            if depth > max_generations:
                break

            self._print_search_summary(depth, population, eval_budget, max_generations, verbose=verbose)

            selected_models = [model for model in population.models if model.evaluated]
            fitness_scores = population.fitness_scores()

            next_model = self.strategy.query(selected_models, fitness_scores, self._x_train, self._y_train,
                                             self.eval_budget_low_level, self.gp_fn, self.gp_args,
                                             **self.model_selector_args)

            population.update([next_model])

            assert len(population.candidates()) > 0
            self._evaluate_models(population.candidates(), eval_budget, callbacks=callbacks, verbose=verbose)

            callbacks.on_generation_end(generation=depth, logs={'gp_models': population.models})
            depth += 1

        return population

    def _initialize(self,
                    eval_budget: int,
                    callbacks: CallbackList,
                    verbose: int = 0) -> ActiveModelPopulation:
        """Initialize models."""
        population = ActiveModelPopulation()
        self.active_models = ActiveModels(self.max_n_models)

        initial_candidates = self._get_initial_candidates()

        population.update(initial_candidates)

        candidates_copies = [GPModel(candidate.covariance, candidate.likelihood) for candidate in initial_candidates]
        initial_candidate_indices = self.active_models.update(candidates_copies)
        no_duplicates = len(initial_candidate_indices) == len(candidates_copies)
        assert no_duplicates

        # init kernel builder
        self.kernel_builder = CorrelationDistanceBuilder(self.noise_prior,
                                                         self.num_samples,
                                                         self.max_n_hyperparameters,
                                                         self.max_n_models,
                                                         self.active_models,
                                                         initial_candidate_indices,
                                                         self._x_train)
        # init strategy
        kernel_kernel_hyperpriors = self.grammar.hyperpriors
        self.strategy = BayesOptGPStrategy(self.active_models, self.acquisition_fn, self.kernel_builder,
                                           kernel_kernel_hyperpriors)

        if verbose == 3:
            pretty_print_gp_models(population.models, 'Initial candidate')

        self._evaluate_models(population.candidates(), eval_budget, callbacks=callbacks, verbose=verbose)
        self.active_models.selected_indices = [0]

        return population

    def _get_initial_candidate_covariances(self) -> List[Covariance]:
        covariance = self.grammar.base_kernels[0]
        return [covariance]
