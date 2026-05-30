from .utils import Logger, Color
from .enums import IndexType
from .config import Config

from pydantic import ValidationError


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
                processed_bm25_index_path: str = 'data/processed/bm25_index',
                processed_chunks_path: str = 'data/processed/chunks',
                processed_chunks_metadata_path: str =
                'data/processed/chunks_metadata.json',
            ) -> None:
        from .modules.indexer import Indexer
        self.logger.log('Starting Indexer...')

        try:
            Indexer(
                lib_path,
                maximum_chunk_size,
                index_type,
                processed_bm25_index_path,
                processed_chunks_path,
                processed_chunks_metadata_path,
                self.verbose,
            )
        except ValidationError as e:
            self.logger.pydantic_error(e, 'Error while validating parameters:')
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
