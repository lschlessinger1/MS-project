import gzip
import importlib
import json


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

    @classmethod
    def load(cls, output_file_name: str):
        compress = output_file_name.split(".")[-1] == "zip"

        if compress:
            with gzip.GzipFile(output_file_name, 'r') as json_data:
                json_bytes = json_data.read()
                json_str = json_bytes.decode('utf-8')
                output_dict = json.loads(json_str)
        else:
            with open(output_file_name) as json_data:
                output_dict = json.load(json_data)

        return cls.from_dict(output_dict)

    def save(self,
             output_filename: str,
             compress: bool = True):
        output_dict = self.to_dict()
        if compress:
            with gzip.GzipFile(output_filename + ".zip", 'w') as outfile:
                json_str = json.dumps(output_dict)
                json_bytes = json_str.encode('utf-8')
                outfile.write(json_bytes)
                return outfile.name
        else:
            with open(output_filename + ".json", 'w') as outfile:
                json.dump(output_dict, outfile)
                return outfile.name
