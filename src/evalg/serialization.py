import importlib


class Serializable:

    def to_dict(self) -> dict:
        return {
            "__class__": self.__class__.__name__,
            "__module__": self.__module__
        }

    @classmethod
    def from_dict(cls, input_dict: dict):
        class_name = input_dict.pop("__class__")
        module_name = input_dict.pop("__module__")
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        return class_._build_from_input_dict(input_dict)

    @classmethod
    def _build_from_input_dict(cls, input_dict: dict):
        """Build input dict and set any attributes from the input dictionary"""
        input_dict = cls._format_input_dict(input_dict)
        return cls(**input_dict)

    @classmethod
    def _format_input_dict(cls, input_dict: dict) -> dict:
        """Format input dictionary before instantiation"""
        return input_dict
