from .interfaces import ChromaDBInterface, DatasetInterface, MetadataInterface
from .models import UnansweredQuestion
from .utils import Logger, Color
from .enums import IndexType
from .config import Config

from pydantic import ValidationError


class RAG:
    logger: Logger
    config: Config

    chromadb_interface: ChromaDBInterface
    dataset_interface: DatasetInterface
    metadata_interface: MetadataInterface

    def __init__(
                self,
                verbose: bool = False,
                model_name: str = 'openai/qwen3:0.6b',
                use_chroma: bool = True,
                chromadb_collection_name: str = 'chunks',
                processed_chromadb_path: str = 'data/processed/chunks/chromadb'
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

        self.chromadb_interface = ChromaDBInterface(config=self.config)
        self.dataset_interface = DatasetInterface(config=self.config)
        self.metadata_interface = MetadataInterface(config=self.config)

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

    def search(
                self,
                question: str,
                k: int = 5,
                save_directory: str = 'data/processed/output',
                file_name: str = 'search_results.json'
            ) -> None:
        from .modules.search import Search

        try:
            Search(
                k,
                save_directory,
                self.chromadb_interface,
                self.metadata_interface,
                self.dataset_interface,
                self.config
            ).search(
                [UnansweredQuestion(question=question)],
                file_name
            )
        except ValidationError as e:
            self.logger.pydantic_error(e, 'Error while validating parameters:')
        except Exception as e:
            self.logger.error(f'Error while searching: {e}')

    def search_dataset(
                self,
                dataset_path: str =
                'data/datasets/UnansweredQuestions/dataset_docs_public.json',
                k: int = 5,
                save_directory: str = 'data/processed/output',
            ) -> None:
        from .modules.search import Search

        try:
            Search(
                k,
                save_directory,
                self.chromadb_interface,
                self.metadata_interface,
                self.dataset_interface,
                self.config
            ).search_dataset(
                dataset_path
            )
        except ValidationError as e:
            self.logger.pydantic_error(e, 'Error while validating parameters:')
        except Exception as e:
            self.logger.error(f'Error while searching: {e}')

    def answer(self) -> None:
        self.logger.warning('Not implemented')

    def answer_dataset(self) -> None:
        self.logger.warning('Not implemented')

    def evaluate(self) -> None:
        self.logger.warning('Not implemented')
