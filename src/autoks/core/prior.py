import importlib
from typing import Type

from src.autoks.backend.prior import RawPriorType, PRIOR_DICT
from src.evalg.serialization import Serializable


class PriorDist(Serializable):
    """Wrapper for backend prior."""

    def __init__(self, raw_prior_cls: Type[RawPriorType], raw_prior_args: dict):
        self._raw_prior_cls = raw_prior_cls
        self._raw_prior_args = raw_prior_args
        self.raw_prior = self._raw_prior_cls(**self._raw_prior_args)

    @classmethod
    def from_prior_str(cls, prior_name: str, raw_prior_args: dict):
        return cls(PRIOR_DICT[prior_name], raw_prior_args)

    def to_dict(self) -> dict:
        """Get a dictionary representation of the object.

        This dict representation includes metadata such as the object's module and class name.

        :return:
        """
        input_dict = super().to_dict()
        input_dict["raw_prior_cls"] = self.raw_prior.__class__.__name__
        input_dict["raw_prior_module"] = self.raw_prior.__module__
        input_dict["raw_prior_args"] = self._raw_prior_args

        return input_dict

    @classmethod
    def _format_input_dict(cls, input_dict: dict) -> dict:
        input_dict = super()._format_input_dict(input_dict)
        class_name = input_dict.pop("raw_prior_cls")
        module_name = input_dict.pop("raw_prior_module")
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        input_dict["raw_prior_cls"] = class_
        return input_dict

    def __eq__(self, other):
        if isinstance(other, PriorDist):
            return self._raw_prior_cls == other._raw_prior_cls and self._raw_prior_args == other._raw_prior_args
        return False
