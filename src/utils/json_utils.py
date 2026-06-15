from typing import TypeVar, Any
from pydantic import BaseModel
import json


T = TypeVar('T', bound=BaseModel)


class JSONUtils:
    '''
    A utility class for saving and loading JSON data.
    '''

    @staticmethod
    def save_json(data: Any, file_path: str, default: Any = False) -> None:
        '''
        Saves the given data to a JSON file at the specified file path.

        Args:
            data (Any): The data to be saved in JSON format.
            file_path (str): The path to the file where the JSON data will be
            saved.
            default (Any, optional): A function that converts non-serializable
            objects to a serializable format. Defaults to False.
        '''
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(
                    data, f,
                    ensure_ascii=False,
                    indent=4,
                    default=default
                )
        except Exception as e:
            raise ValueError(
                f'Error occurred while saving JSON to {file_path}: {e}'
            )

    @staticmethod
    def load_json(file_path: str) -> Any:
        '''
        Loads JSON data from the specified file path.

        Args:
            file_path (str): The path to the JSON file to be loaded.
        '''
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'File {file_path!r} not found!')
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON from {file_path!r}: {e}')
        except Exception as e:
            raise ValueError(
                f'Error occurred while loading JSON from {file_path!r}: {e}'
            )

    @staticmethod
    def load_json_to_model(file: str, model_class: type[T]) -> T:
        '''
        Loads JSON data from the specified file and validates it against the
        provided Pydantic model class.

        Args:
            file (str): The path to the JSON file to be loaded.
            model_class (type[T]): The Pydantic model class to validate the
            loaded data against.
        '''
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'File {file!r} not found!')
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON from {file!r}: {e}')
        except Exception as e:
            raise ValueError(
                f'Error occurred while loading JSON to model from {file!r}: '
                f'{e}'
            )
        return model_class.model_validate(data)
