from .interfaces import ChromaDBInterface
from .utils import Logger, Color
from .enums import IndexType
from .config import Config

from pydantic import ValidationError


class RAG:
    logger: Logger
    config: Config

    chromadb_interface: ChromaDBInterface

    def __init__(
                self,
                verbose: bool = False,
                model_name: str = 'openai/qwen3:0.6b',
                use_chroma: bool = True,
                processed_chromadb_path: str = 'data/processed/chunks/chromadb',
                chromadb_collection_name: str = 'chunks',
            ) -> None:
        self.logger = Logger('Main', Color.MAGENTA, verbose)
        self.logger.log('Initializing RAG...')
        self.config = Config(
            verbose=verbose,
            model_name=model_name,
            use_chroma=use_chroma,
            chromadb_path=processed_chromadb_path,
            chromadb_collection_name=chromadb_collection_name,
        )

        self.chromadb_interface = ChromaDBInterface(
            config=self.config,
        )

    def index(
                self,
                lib_path: str = 'data/raw/vllm-0.10.1',
                maximum_chunk_size: int = 2000,
                index_type: IndexType = IndexType.ALL,
                processed_bm25_index_path: str = 'data/processed/bm25_index',
                processed_chunks_path: str = 'data/processed/chunks',
                processed_chunks_metadata_path: str =
                'data/processed/chunks/chunks_metadata.json',
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
                self.chromadb_interface,
                self.config,
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
