from .utils import Logger, Color
from .enums import IndexType
from .config import Config


class RAG:
    logger: Logger
    verbose: bool

    def __init__(self, verbose: bool = False) -> None:
        self.logger = Logger('Main', verbose, Color.MAGENTA)
        self.logger.log('Initializing RAG...')
        self.config = Config()
        self.verbose = verbose

    def index(
                self,
                lib_path: str = 'data/raw/vllm-0.10.1',
                maximum_chunk_size: int = 2000,
                index_type: IndexType = IndexType.ALL,
            ) -> None:
        from .modules.indexer import Indexer
        self.logger.log('Starting Indexer...')
        try:
            Indexer(lib_path, maximum_chunk_size, index_type, self.verbose)
        except Exception as e:
            self.logger.error(f'Error while indexing: {e}')

    def search(self) -> None:
        self.logger.info('Not implemented')

    def search_dataset(self) -> None:
        self.logger.info('Not implemented')

    def answer(self) -> None:
        self.logger.info('Not implemented')

    def answer_dataset(self) -> None:
        self.logger.info('Not implemented')

    def evaluate(self) -> None:
        self.logger.info('Not implemented')
