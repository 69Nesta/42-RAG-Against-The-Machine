from .interfaces import (
    ChromaDBInterface, DatasetInterface,
    ChunksInterface, Bm25sInterface,
    DspyInterface
)
from .models import UnansweredQuestion
from .utils import Logger, Color
from .enums import FileType
from .config import Config

from pydantic import ValidationError


class RAG:
    logger: Logger
    config: Config

    chromadb_interface: ChromaDBInterface
    dataset_interface: DatasetInterface
    chunks_interface: ChunksInterface
    bm25s_interface: Bm25sInterface
    dspy_interface: DspyInterface

    def __init__(
        self,
        verbose: bool = False,
        model_name: str = 'openai/qwen3:0.6b',
        # model_name: str = 'openai/Qwen/Qwen3-0.6B',
        temperature: float = 0.3,
        api_base: str = 'http://localhost:11434/v1',
        api_key: str = 'EMPTY',
        max_tokens: int = 102400,
        dspy_cache: bool = True,

        use_query_expansion: bool = False,
        use_hyde: bool = False,

        use_chroma: bool = False,
        chromadb_collection_name: str = 'chunks',
        chromadb_path: str = 'data/processed/chunks/chromadb',

        bm25_k1: float = 2.0,
        bm25_b: float = 0.75,

        processed_bm25_index_path: str = 'data/processed/bm25_index',
        processed_chunks_path: str = 'data/processed/chunks/contents.json',

        rrf_weights_bm25: float = 1.2,
        rrf_weights_chroma: float = 1.0,
        rrf_weights_bm25_expanded: float = .3,
        rrf_weights_chroma_expanded: float = .5,
        rrf_weights_HyDE: float = .7

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

            use_query_expansion=use_query_expansion,
            use_hyde=use_hyde,

            use_chroma=use_chroma,
            chromadb_path=chromadb_path,
            chromadb_collection_name=chromadb_collection_name,

            bm25_k1=bm25_k1,
            bm25_b=bm25_b,

            processed_bm25_index_path=processed_bm25_index_path,
            processed_chunks_path=processed_chunks_path,

            rrf_weights_bm25=rrf_weights_bm25,
            rrf_weights_chroma=rrf_weights_chroma,
            rrf_weights_bm25_expanded=rrf_weights_bm25_expanded,
            rrf_weights_chroma_expanded=rrf_weights_chroma_expanded,
            rrf_weights_HyDE=rrf_weights_HyDE
        )

        self.chromadb_interface = ChromaDBInterface(config=self.config)
        self.dataset_interface = DatasetInterface(config=self.config)
        self.chunks_interface = ChunksInterface(config=self.config)
        self.bm25s_interface = Bm25sInterface(config=self.config)
        self.dspy_interface = DspyInterface(config=self.config)

    def index(
                self,
                lib_path: str = 'data/raw/vllm-0.10.1',
                maximum_chunk_size: int = 2000,
                index_type: FileType = FileType.ALL,
                overlap: int = 5
            ) -> None:
        from .modules.indexer_module import IndexerModule

        try:
            IndexerModule(
                lib_path,
                maximum_chunk_size,
                index_type,
                overlap,
                self.chromadb_interface,
                self.chunks_interface,
                self.bm25s_interface,
                self.config,
            ).index()
        except ValidationError as e:
            self.logger.pydantic_error(e, 'Error while validating parameters:')
        except Exception as e:
            self.logger.error(f'Error while indexing: {e}')

    def search(
                self,
                question: str,
                k: int = 5,
                save_directory: str = 'data/output/search_results',
                file_name: str = 'search_results.json',
                search_type: FileType = FileType.ALL,
            ) -> None:
        from .modules.search_module import SearchModule

        try:
            SearchModule(
                k,
                save_directory,
                search_type,
                self.chromadb_interface,
                self.dataset_interface,
                self.chunks_interface,
                self.bm25s_interface,
                self.dspy_interface,
                self.config
            ).search(
                UnansweredQuestion(question=question),
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
                search_type: FileType = FileType.ALL,
            ) -> None:
        from .modules.search_module import SearchModule

        try:
            SearchModule(
                k,
                save_directory,
                search_type,
                self.chromadb_interface,
                self.dataset_interface,
                self.chunks_interface,
                self.bm25s_interface,
                self.dspy_interface,
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
                search_type: FileType = FileType.ALL,
            ) -> None:
        from .modules.answer_module import AnswerModule

        try:
            AnswerModule(
                save_directory,
                self.chromadb_interface,
                self.dataset_interface,
                self.chunks_interface,
                self.dspy_interface,
                self.config
            ).answer(
                question,
                k,
                self.bm25s_interface,
                search_type,
            )
        except ValidationError as e:
            self.logger.pydantic_error(e, 'Error while validating parameters:')
        except Exception as e:
            self.logger.error(f'Error while searching: {e}')

    def answer_dataset(
                self,
                student_search_results_path: str =
                'data/output/search_results/dataset_docs_public.json',
                save_directory: str = 'data/output/search_results_and_answer',
            ) -> None:
        from .modules.answer_module import AnswerModule

        try:
            AnswerModule(
                save_directory,
                self.chromadb_interface,
                self.dataset_interface,
                self.chunks_interface,
                self.dspy_interface,
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
        from .modules.evaluate_module import EvaluateModule

        try:
            EvaluateModule(
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
