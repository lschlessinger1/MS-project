from typing import List, Callable, Optional, Union

from src.autoks.backend.model import RawGPModelType
from src.autoks.callbacks import CallbackList
from src.autoks.core.covariance import Covariance
from src.autoks.core.gp_model_population import GPModelPopulation
from src.autoks.core.grammar import GeometricRandomGrammar, UniformRandomGrammar, BaseGrammar
from src.autoks.core.hyperprior import boms_hyperpriors
from src.autoks.core.model_selection.base import ModelSelector


class RandomBaseSelector(ModelSelector):

    def __init__(self,
                 grammar: BaseGrammar,
                 fitness_fn: Union[str, Callable[[RawGPModelType], float]] = 'loglikn',
                 n_parents: int = 1,
                 additive_form: bool = False,
                 optimizer: Optional[str] = None,
                 n_restarts_optimizer: int = 3,
                 gp_fn: Union[str, Callable] = 'gp_regression',
                 gp_args: Optional[dict] = None):

        super().__init__(grammar, fitness_fn, n_parents, additive_form, gp_fn, gp_args, optimizer, n_restarts_optimizer)

    def _get_initial_candidate_covariances(self) -> List[Covariance]:
        return []

    def _train(self,
               eval_budget: int,
               max_generations: int,
               callbacks: CallbackList,
               verbose: int = 0) -> GPModelPopulation:
        population = self._initialize(eval_budget, callbacks=callbacks, verbose=verbose)

        # hacky fix to set number of models
        if isinstance(self.grammar, GeometricRandomGrammar):
            self.grammar._number_of_random_walks = eval_budget
        elif isinstance(self.grammar, UniformRandomGrammar):
            self.grammar.n_models = eval_budget

        depth = 0
        while self.n_evals < eval_budget:
            callbacks.on_generation_begin(generation=depth, logs={'gp_models': population.models})

            if depth > max_generations:
                break

            self._print_search_summary(depth, population, eval_budget, max_generations, verbose=verbose)

            new_models = self._propose_new_models(population, callbacks=callbacks, verbose=verbose)
            population.models = new_models
            assert len(population.candidates()) == eval_budget

            self._evaluate_models(population.candidates(), eval_budget, callbacks=callbacks, verbose=verbose,
                                  skip_evaluated=False)

            callbacks.on_generation_end(generation=depth, logs={'gp_models': population.models})
            depth += 1

        return population

    @classmethod
    def _build_from_input_dict(cls, input_dict: dict):
        standardize_x = input_dict.pop('standardize_x')
        standardize_y = input_dict.pop('standardize_y')

        model_selector = super()._build_from_input_dict(input_dict)

        model_selector.standardize_x = standardize_x
        model_selector.standardize_y = standardize_y

        return model_selector

    def __str__(self):
        return self.name


class GeomRandomModelSelector(RandomBaseSelector):

    def __init__(self,
                 grammar: Optional[GeometricRandomGrammar] = None,
                 base_kernel_names: Optional[List[str]] = None,
                 fitness_fn: Union[str, Callable[[RawGPModelType], float]] = 'loglikn',
                 n_parents: int = 1,
                 additive_form: bool = False,
                 optimizer: Optional[str] = None,
                 n_restarts_optimizer: int = 3,
                 gp_fn: Union[str, Callable] = 'gp_regression',
                 gp_args: Optional[dict] = None):
        if grammar is None:
            grammar = GeometricRandomGrammar(base_kernel_names=base_kernel_names)

        super().__init__(grammar, fitness_fn, n_parents, additive_form, optimizer, n_restarts_optimizer, gp_fn, gp_args)


class UniformRandomModelSelector(RandomBaseSelector):

    def __init__(self,
                 grammar: Optional[UniformRandomGrammar] = None,
                 hyperprior: Optional = None,
                 base_kernel_names: Optional[List[str]] = None,
                 fitness_fn: Union[str, Callable[[RawGPModelType], float]] = 'loglikn',
                 n_parents: int = 1,
                 additive_form: bool = False,
                 optimizer: Optional[str] = None,
                 n_restarts_optimizer: int = 3,
                 gp_fn: Union[str, Callable] = 'gp_regression',
                 gp_args: Optional[dict] = None):
        if grammar is None:
            hyperprior = hyperprior or boms_hyperpriors()
            grammar = UniformRandomGrammar(base_kernel_names=base_kernel_names, hyperpriors=hyperprior)

        super().__init__(grammar, fitness_fn, n_parents, additive_form, optimizer, n_restarts_optimizer, gp_fn, gp_args)
