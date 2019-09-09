import warnings
from typing import List, Union, Callable, Optional

import numpy as np

from src.autoks.backend.model import RawGPModelType
from src.autoks.callbacks import CallbackList
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model import GPModel, pretty_print_gp_models
from src.autoks.core.gp_model_population import GPModelPopulation, ActiveModelPopulation
from src.autoks.core.grammar import BaseGrammar
from src.autoks.core.model_selection import SurrogateEvolutionaryModelSelector
from src.autoks.core.model_selection.base import ModelSelector
from src.autoks.distance.distance import ActiveModels


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
        grammar = BaseGrammar(base_kernel_names=base_kernel_names)
        super().__init__(grammar, fitness_fn, gp_fn=gp_fn, gp_args=gp_args)
        self.gp_fn = gp_fn
        self.gp_args = gp_args
        self.model_selector_args = ms_args

        self.active_models = None
        self.covariance_to_info_map = dict()

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
            model_selector = SurrogateEvolutionaryModelSelector(selected_models, fitness_scores, self._x_train,
                                                                self.covariance_to_info_map,
                                                                gp_fn=self.gp_fn, gp_args=self.gp_args,
                                                                **self.model_selector_args)

            model_selector.train(self._x_train, self._y_train, eval_budget=self.eval_budget_low_level, verbose=0)
            new_models = model_selector.selected_models

            all_fitness_scores_zero = all(m.score == 0 for m in new_models)
            if all_fitness_scores_zero:
                warnings.warn('All acquisition scores are 0. Be sure this is correct behavior. Reverting to random '
                              'selection of candidate models.')
                pool = [m for m in new_models if m not in selected_models]
                assert len(pool) > 0
                next_model = np.random.choice(pool)
            else:
                next_model_index = int(np.argmax([model.score for model in new_models]))
                next_model = new_models[next_model_index]
            next_model.covariance.raw_kernel.unset_priors()
            # Reinitialize to get rid of acquisition score.
            next_model = GPModel(next_model.covariance, next_model.likelihood)

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
        self.active_models = ActiveModels(max_n_models=1000)

        initial_candidates = self._get_initial_candidates()

        population.update(initial_candidates)
        initial_candidate_indices = self.active_models.update(initial_candidates)

        no_duplicates = len(initial_candidate_indices) == len(initial_candidates)
        assert no_duplicates

        if verbose == 3:
            pretty_print_gp_models(population.models, 'Initial candidate')

        self._evaluate_models(population.candidates(), eval_budget, callbacks=callbacks, verbose=verbose)
        self.active_models.selected_indices = [0]

        return population

    def _get_initial_candidate_covariances(self) -> List[Covariance]:
        covariance = self.grammar.base_kernels[0]
        covariance.raw_kernel.unset_priors()
        return [covariance]
