import json
from typing import Any


class JSONUtils:
    @staticmethod
    def save_json(data: Any, file_path: str) -> None:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            raise ValueError(
                f'Error occurred while saving JSON to {file_path}: {e}'
            )
