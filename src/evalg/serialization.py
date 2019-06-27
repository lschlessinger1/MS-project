import importlib


class Serializable:

    def to_dict(self) -> dict:
        obj_dict = {
            "__class__": self.__class__.__name__,
            "__module__": self.__module__
        }
        return obj_dict

    @staticmethod
    def from_dict(input_dict: dict):
        class_name = input_dict.pop("__class__")
        module_name = input_dict.pop("__module__")
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        obj = class_(**input_dict)
        return obj
