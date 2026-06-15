from ..utils import Logger, Color, JSONUtils
from ..models import StudentSearchResults
from ..config import Config

from pathlib import Path


class SearchResultsInterface:
    logger: Logger
    app_config: Config

    _cache: dict[str, StudentSearchResults] = {}

    def __init__(
                self,
                config: Config,
            ) -> None:
        self.app_config = config
        self.logger = Logger(
            'SearchResultsInterface',
            Color.CYAN,
            config.verbose
        )

    def get_search_results(self, file: str) -> StudentSearchResults:
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
