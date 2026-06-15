from .interfaces import (
    ChromaDBInterface, DatasetInterface,
    ChunksInterface, Bm25sInterface,
    DspyInterface
)
from .utils import Logger, Color, Loader
from .models import UnansweredQuestion
from .enums import FileType
from .config import Config

from pydantic import ValidationError


class RAG:
    '''
    RAG (Retrieval-Augmented Generation) is a framework that combines
    retrieval-based methods with generative models to enhance the quality of
    generated responses. It retrieves relevant information from a knowledge
    base and uses it to generate more accurate and contextually relevant
    answers.
    '''

    loader: Loader
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
        max_tokens: int = 2048,
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
        '''
        Initializes the RAG framework with the provided configuration
        parameters.

        Args:
            verbose (bool): Whether to enable verbose logging.
            model_name (str): The name of the language model to use.
            temperature (float): The temperature for the language model.
            api_base (str): The base URL for the language model API.
            api_key (str): The API key for the language model.
            max_tokens (int): The maximum number of tokens for the language
            model.
            dspy_cache (bool): Whether to enable caching for the Dspy
            interface.

            use_query_expansion (bool): Whether to use query expansion.
            use_hyde (bool): Whether to use HyDE for query expansion.

            use_chroma (bool): Whether to use ChromaDB for retrieval.
            chromadb_collection_name (str): The name of the ChromaDB
            collection.
            chromadb_path (str): The path to the ChromaDB database.

            bm25_k1 (float): The k1 parameter for BM25.
            bm25_b (float): The b parameter for BM25.

            processed_bm25_index_path (str): The path to the processed BM25
            index.
            processed_chunks_path (str): The path to the processed chunks.

            rrf_weights_bm25 (float): The RRF weight for BM25 results.
            rrf_weights_chroma (float): The RRF weight for ChromaDB results.
            rrf_weights_bm25_expanded (float): The RRF weight for BM25 results
            from the expanded query.
            rrf_weights_chroma_expanded (float): The RRF weight for ChromaDB
            results from the expanded query.
            rrf_weights_HyDE (float): The RRF weight for HyDE results.
        '''
        self.logger = Logger('RAG', Color.MAGENTA, verbose)
        self.logger.log('Initializing RAG...')
        self.loader = Loader(self.logger)

        self.loader.print_logo()
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
        '''
        Indexes the documents in the specified library path.

        Args:
            lib_path (str): The path to the library containing the documents
            to index.
            maximum_chunk_size (int): The maximum size of each chunk when
            splitting the documents.
            index_type (FileType): The type of files to index (e.g., docs,
            code, all).
            overlap (int): The number of overlapping tokens between chunks
            when splitting the documents.
        '''
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
        '''
        Searches for relevant information in the indexed documents based on the
        provided question.
        Args:
            question (str): The question to search for.
            k (int): The number of top relevant results to retrieve.
            save_directory (str): The directory to save the search results.
            file_name (str): The name of the file to save the search results.
            search_type (FileType): The type of files to search in (e.g.,
            docs, code, all).
        '''
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
        '''
        Searches for relevant information in the indexed documents based on the
        questions in the provided dataset.
        Args:
            dataset_path (str): The path to the dataset containing the
            questions to search for.
            k (int): The number of top relevant results to retrieve for each
            question.
            save_directory (str): The directory to save the search results.
            search_type (FileType): The type of files to search in (e.g.,
            docs, code, all).
        '''
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
        '''
        Answers the provided question using the retrieved information from the
        indexed documents.

        Args:
            question (str): The question to answer.
            k (int): The number of top relevant results to retrieve for
            answering the question.
            save_directory (str): The directory to save the search results and
            the generated answer.
            search_type (FileType): The type of files to search in (e.g.,
            docs, code, all).
        '''
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
        '''
        Answers the questions in the provided dataset using the retrieved
        information from the indexed documents.

        Args:
            student_search_results_path (str): The path to the dataset
            containing the search results for the questions to answer.
            save_directory (str): The directory to save the search results and
            the generated answers.
        '''
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
        '''
        Evaluates the answers generated for the questions in the provided
        dataset against the ground truth answers.

        Args:
            student_answer_path (str): The path to the dataset containing the
            generated answers for the questions to evaluate.
            dataset_path (str): The path to the dataset containing the ground
            truth answers for the questions to evaluate against.
        '''
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
