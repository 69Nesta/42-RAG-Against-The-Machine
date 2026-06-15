from ..utils import Logger, Color, JSONUtils
from ..models import RagDataset
from ..config import Config

from pathlib import Path


class DatasetInterface:
    '''
    Interface for managing datasets, including loading and caching datasets
    from JSON files.
    '''

    logger: Logger
    config: Config

    _cache: dict[str, RagDataset]

    def __init__(self, config: Config) -> None:
        '''
        Initializes the DatasetInterface with the provided configuration.

        Args:
            config: The application configuration containing paths and
            settings.
        '''
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
        '''
        Loads a dataset from the specified JSON file path, utilizing caching
        to avoid redundant file reads.

        Args:
            file_path: The path to the JSON file containing the dataset.
        Returns:
            An instance of RagDataset loaded from the specified file.
        Raises:
            ValueError: If the file path is not provided.
            FileNotFoundError: If the specified file does not exist.
        '''
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
