from ..utils import Logger, Color, JSONUtils
from ..models import StudentSearchResults
from ..config import Config

from pathlib import Path


class SearchResultsInterface:
    '''
    Interface for managing search results, including loading search results
    from JSON files and caching them for efficient retrieval.
    '''
    logger: Logger
    app_config: Config

    _cache: dict[str, StudentSearchResults] = {}

    def __init__(
                self,
                config: Config,
            ) -> None:
        '''
        Initializes the SearchResultsInterface with the provided configuration.

        Args:
            config: The application configuration containing paths and
            settings.
        '''
        self.app_config = config
        self.logger = Logger(
            'SearchResultsInterface',
            Color.CYAN,
            config.verbose
        )

    def get_search_results(self, file: str) -> StudentSearchResults:
        '''
        Retrieves search results from the specified JSON file, utilizing
        caching to avoid redundant file reads.

        Args:
            file: The path to the JSON file containing the search results.
        Returns:
            An instance of StudentSearchResults loaded from the specified file.
        Raises:
            ValueError: If the file path is not provided.
            FileNotFoundError: If the specified file does not exist.
        '''
        path: Path = Path(file)
        file_path: str = path.as_posix()

        if file_path in self._cache:
            self.logger.log_tqdm(f'Cache hit for {path!r}')
            return self._cache[file_path]

        self.logger.log_tqdm(
            f'Cache miss for {file_path!r}. Loading from disk...'
        )

        search_results: StudentSearchResults = JSONUtils.load_json_to_model(
            file_path,
            StudentSearchResults
        )

        self._cache.update({
            file_path: search_results
        })
        return search_results
