from ..utils import Logger, Color, JSONUtils
from ..models import RagDataset
from ..config import Config

from pathlib import Path


class DatasetInterface:
    logger: Logger
    config: Config

    _cache: dict[str, RagDataset]

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = Logger(
            'DatasetInterface',
            Color.BRIGHT_GREEN,
            config.verbose
        )

        self._cache = {}

    def load_dataset(
                self,
                file_path: str,
            ) -> RagDataset:
        if not file_path:
            raise ValueError(
                f'No file name found for index type {file_path}.'
            )

        path: Path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f'File not found: {path.as_posix()}')

        if path.as_posix() in self._cache:
            return self._cache[path.as_posix()]
        data = JSONUtils.load_json(path.as_posix())
        dataset = RagDataset(
            rag_questions=data.get('rag_questions', [])
        )
        self._cache.update({
            path.as_posix(): dataset
        })

        return dataset
