from typing import List, Optional, Tuple

from src.autoks.core.gp_model import GPModel

# We define a model to be this for the active set
Model = GPModel


class ActiveSet:
    _remove_priority: List[int]
    _models: List[Optional[Model]]

    def __init__(self, max_n_models: int):
        self.max_n_models = max_n_models
        self._models = [None] * self.max_n_models
        self._remove_priority = []  # list of indices representing order in which to remove models
        self.selected_indices = []
        self._model_dict_cache = dict()

    @property
    def remove_priority(self) -> List[int]:
        return self._remove_priority

    @remove_priority.setter
    def remove_priority(self, remove_priority) -> None:
        if len(remove_priority) == 0:
            self._remove_priority = []
            return

        if len(remove_priority) > self.max_n_models:
            raise ValueError(f'Size of removal priority {len(remove_priority)} cannot be greater than maximum number '
                             f'of models {self.max_n_models}.')

        # also check for improper indices
        if min(remove_priority) < 0 or self.max_n_models - 1 < max(remove_priority):
            raise ValueError(f'All indices must be between 0 <= i <= {self.max_n_models - 1}.')

        self._remove_priority = remove_priority.copy()

    @property
    def models(self) -> List[Optional[Model]]:
        return self._models

    @models.setter
    def models(self, models) -> None:
        if len(models) > self.max_n_models:
            raise ValueError(f'Size of models {len(models)} cannot be greater than maximum number of models '
                             f'{self.max_n_models}.')
        self._models = models

    def get_index_to_insert(self) -> int:
        n_models = len(self)
        if n_models < self.max_n_models:
            return n_models

        if len(self.remove_priority) == 0:
            raise ValueError(f'Must set removal priority when {self.__class__.__name__} is full.')

        return self.remove_priority.pop(0)

    def update(self, candidates: List[Model]) -> List[int]:
        """Returns new candidate indices"""
        newly_inserted_indices = []

        # i.e. candidates that haven't already been inserted.
        unique_candidates = [c for c in candidates if c.covariance.symbolic_expr_expanded not in self._model_dict_cache]

        for candidate in unique_candidates:
            inserted_index, success = self.add_model(candidate)
            if success:
                newly_inserted_indices.append(inserted_index)

        # Assume that any model that exceeded the size limit was replaced.
        return newly_inserted_indices[-self.max_n_models:]

    def add_model(self, model: Model) -> Tuple[int, bool]:
        new_model_key = model.covariance.symbolic_expr_expanded
        if model in self.models or new_model_key in self._model_dict_cache:
            return -1, False

        idx = self.get_index_to_insert()

        self._models[idx] = model

        if idx in self._model_dict_cache:
            old_model_key = self._model_dict_cache[idx]
            del self._model_dict_cache[old_model_key]
        self._model_dict_cache[idx] = new_model_key
        self._model_dict_cache[new_model_key] = idx

        return idx, True

    def get_selected_models(self) -> List[Model]:
        return [self.models[i] for i in self.selected_indices]

    def get_candidate_indices(self) -> List[int]:
        return [i for (i, model) in enumerate(self.models) if i not in self.selected_indices and model is not None]

    def get_candidates(self) -> List[Model]:
        return [self.models[i] for i in self.get_candidate_indices()]

    def index(self, model: Model) -> int:
        return self._model_dict_cache[model.covariance.symbolic_expr_expanded]

    def get(self, model: Model, default: Optional = None) -> int:
        return self._model_dict_cache.get(model.covariance.symbolic_expr_expanded, default)

    def __len__(self) -> int:
        return sum(1 for m in self.models if m is not None)

    def __repr__(self):
        return f'{self.__class__.__name__}('f'max_n_models={self.max_n_models!r}, ' \
            f'selected_indices={self.selected_indices!r})'
