from .interfaces import ChromaDBInterface, DatasetInterface, ChunksInterface
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
    chunks_interface: ChunksInterface

    def __init__(
        self,
        verbose: bool = False,
        model_name: str = 'openai/qwen3:0.6b',
        # model_name: str = 'openai/Qwen/Qwen3-0.6B',
        temperature: float = 0.1,
        api_base: str = 'http://localhost:11434/v1',
        api_key: str = 'EMPTY',
        max_tokens: int = 102400,
        dspy_cache: bool = True,

        use_chroma: bool = True,
        chromadb_collection_name: str = 'chunks',
        chromadb_path: str = 'data/processed/chunks/chromadb',

        processed_bm25_index_path: str = 'data/processed/bm25_index',
        processed_chunks_path: str = 'data/processed/chunks/contents.json',

        bm25_weights_rrf: float = 1.15,
        chroma_weights_rrf: float = .85,
    ) -> None:
        self.logger = Logger('RAG', Color.MAGENTA, verbose)
        self.logger.log('Initializing RAG...')
        self.config = Config(
            verbose=verbose,

            model_name=model_name,
            temperature=temperature,
            api_base=api_base,
            api_key=api_key,
            max_tokens=max_tokens,
            dspy_cache=dspy_cache,

            use_chroma=use_chroma,
            chromadb_path=chromadb_path,
            chromadb_collection_name=chromadb_collection_name,

            processed_bm25_index_path=processed_bm25_index_path,
            processed_chunks_path=processed_chunks_path,

            bm25_weights_rrf=bm25_weights_rrf,
            chroma_weights_rrf=chroma_weights_rrf,
        )

        self.chromadb_interface = ChromaDBInterface(config=self.config)
        self.dataset_interface = DatasetInterface(config=self.config)
        self.chunks_interface = ChunksInterface(config=self.config)

    def index(
                self,
                lib_path: str = 'data/raw/vllm-0.10.1',
                maximum_chunk_size: int = 2000,
                index_type: IndexType = IndexType.ALL
            ) -> None:
        from .modules.indexer import Indexer

        try:
            Indexer(
                lib_path,
                maximum_chunk_size,
                index_type,
                self.chromadb_interface,
                self.chunks_interface,
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
                save_directory: str = 'data/output/search_results',
                file_name: str = 'search_results.json'
            ) -> None:
        from .modules.search import Search

        try:
            Search(
                k,
                save_directory,
                self.chromadb_interface,
                self.dataset_interface,
                self.chunks_interface,
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
                save_directory: str = 'data/output/search_results',
            ) -> None:
        from .modules.search import Search

        try:
            Search(
                k,
                save_directory,
                self.chromadb_interface,
                self.dataset_interface,
                self.chunks_interface,
                self.config
            ).search_dataset(
                dataset_path
            )
        except ValidationError as e:
            self.logger.pydantic_error(e, 'Error while validating parameters:')
        except Exception as e:
            self.logger.error(f'Error while searching: {e}')

    def answer(
                self,
                question: str,
                k: int = 5,
                save_directory: str = 'data/output/search_results_and_answer',
            ) -> None:
        from .modules.answer import Answer

        try:
            Answer(
                save_directory,
                self.chromadb_interface,
                self.dataset_interface,
                self.chunks_interface,
                self.config
            ).answer(question, k)
        except ValidationError as e:
            self.logger.pydantic_error(e, 'Error while validating parameters:')
        except Exception as e:
            self.logger.error(f'Error while searching: {e}')
        self.logger.warning('Not implemented')

    def answer_dataset(
                self,
                student_search_results_path: str =
                'data/output/search_results/dataset_code_public.json',
                save_directory: str = 'data/output/search_results_and_answer',
            ) -> None:
        from .modules.answer import Answer

        try:
            Answer(
                save_directory,
                self.chromadb_interface,
                self.dataset_interface,
                self.chunks_interface,
                self.config
            ).answer_dataset(student_search_results_path)
        except ValidationError as e:
            self.logger.pydantic_error(e, 'Error while validating parameters:')
        except Exception as e:
            self.logger.error(f'Error while searching: {e}')

    def evaluate(
                self,
                student_answer_path: str =
                'data/output/search_results/dataset_docs_public.json',
                dataset_path: str =
                'data/datasets/AnsweredQuestions/dataset_docs_public.json',
            ) -> None:
        from .modules.evaluate import Evaluate

        try:
            Evaluate(
                student_answer_path,
                dataset_path,
                self.dataset_interface,
                self.chunks_interface,
                self.config
            )
        except ValidationError as e:
            self.logger.pydantic_error(e, 'Error while validating parameters:')
        except Exception as e:
            self.logger.error(f'Error while searching: {e}')
