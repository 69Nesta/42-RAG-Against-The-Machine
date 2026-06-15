from ..models import (
    MinimalSearchResults,
    StudentSearchResults,
    UnansweredQuestion,
    MinimalSource,
)
from ..interfaces import (
    ChromaDBInterface,
    DatasetInterface,
    ChunksInterface,
    Bm25sInterface,
    DspyInterface
)
from ..utils import Logger, Color, JSONUtils, TimeUtils
from ..enums import FileType, FilesExt
from ..config import Config

from pydantic import BaseModel, Field, model_validator
from pathlib import Path
from tqdm import tqdm


class SearchConfig(BaseModel):
    '''
    Configuration class for the SearchModule.
    '''

    k: int = Field(..., gt=0, le=10)
    save_directory: str = Field(..., min_length=1)
    search_type: FileType

    @model_validator(mode='after')
    def check_paths_differ(self) -> 'SearchConfig':
        '''
        Validates that the save_directory is a directory and not a file.
        '''
        checks = [
            (self.save_directory, True),
        ]

        for path, should_be_dir in checks:
            p = Path(path)
            if p.exists() and p.is_dir() != should_be_dir:
                kind = 'directory' if should_be_dir else 'file'
                raise ValueError(f'{path} must be a {kind}!')

        return self


class SearchModule:
    logger: Logger
    app_config: Config
    chromadb_interface: ChromaDBInterface
    dataset_interface: DatasetInterface
    chunks_interface: ChunksInterface
    bm25s_interface: Bm25sInterface
    dspy_interface: DspyInterface

    config: SearchConfig

    def __init__(
                self,
                k: int,
                save_directory: str,
                search_type: FileType,
                chromadb_interface: ChromaDBInterface,
                dataset_interface: DatasetInterface,
                chunks_interface: ChunksInterface,
                bm25s_interface: Bm25sInterface,
                dspy_interface: DspyInterface,
                config: Config,
            ) -> None:
        '''
        Initializes the SearchModule with the provided parameters and
        interfaces.

        Args:
            k (int): The number of top results to retrieve.
            save_directory (str): The directory where search results will be
            saved.
            search_type (FileType): The type of files to search.
            chromadb_interface (ChromaDBInterface): Interface for ChromaDB
            operations.
            dataset_interface (DatasetInterface): Interface for dataset
            operations.
            chunks_interface (ChunksInterface): Interface for chunk operations.
            bm25s_interface (Bm25sInterface): Interface for BM25 operations.
            dspy_interface (DspyInterface): Interface for Dspy operations.
            config (Config): Application configuration settings.
        '''
        self.app_config = config
        self.chromadb_interface = chromadb_interface
        self.dataset_interface = dataset_interface
        self.chunks_interface = chunks_interface
        self.bm25s_interface = bm25s_interface
        self.dspy_interface = dspy_interface

        self.logger = Logger('SearchModule', Color.BRIGHT_BLUE, config.verbose)
        self.logger.log('Initializing Search Module...')

        self.config = SearchConfig(
            save_directory=Path(save_directory).as_posix(),
            k=k,
            search_type=search_type,
        )

        self.bm25s_interface.load()

    def print_retrieved_sources(
                self,
                sources: list[MinimalSource]
            ) -> None:
        '''
        Prints the retrieved sources in a formatted table.

        Args:
            sources (list[MinimalSource]): A list of MinimalSource objects to
            be printed.
        '''
        self.logger.info('')
        self.logger.table_info(
            headers=['#', 'File Path', 'Character Range'],
            rows=[
                [
                    f'{Color.YELLOW}{idx:<4}{Color.RESET}',
                    (str(source.file_path) if len(str(source.file_path)) <= 63
                     else '…' + str(source.file_path)[-62:]),
                    f'{Color.WHITE}{source.first_character_index:<4} – '
                    f'{source.last_character_index:<4}{Color.RESET}'
                ]
                for idx, source in enumerate(sources, start=1)
            ]
        )
        self.logger.info('')

    def search_sources(
                self,
                query: UnansweredQuestion,
                print_expanded: bool = False,
                print_hyde: bool = False
            ) -> list[MinimalSource]:
        '''
        Searches for sources based on the provided query and returns a list of
        MinimalSource objects.

        Args:
            query (UnansweredQuestion): The query object containing the
            question.
            print_expanded (bool, optional): Whether to print expanded query
            information. Defaults to False.
            print_hyde (bool, optional): Whether to print HyDE information.
            Defaults to False.
        Returns:
            list[MinimalSource]: A list of MinimalSource objects retrieved
            based on the query.
        '''
        if not query.question.strip():
            return []

        fake_k: int = max(self.config.k * 20, 150)
        documents_weights: list[tuple[list[MinimalSource], float]] = []

        documents_weights.append((
            self._transform_to_sources(
                self.bm25s_interface.retrieve([query.question], fake_k)
            ),
            self.app_config.rrf_weights_bm25
        ))

        if self.app_config.use_chroma:
            documents_weights.append((
                self._transform_to_sources(
                    self.chromadb_interface.search([query.question], fake_k)
                ),
                self.app_config.rrf_weights_chroma
            ))

        if self.app_config.use_query_expansion:
            self._add_expanded_sources(
                fake_k,
                query.question,
                documents_weights,
                print_expanded
            )

        if self.app_config.use_hyde:
            self._add_HyDE(
                fake_k,
                query.question,
                documents_weights,
                print_hyde
            )

        return self._apply_rrf(documents_weights, self.config.k)

    def search(self, question: UnansweredQuestion, file: str) -> None:
        '''
        Searches for sources based on the provided question and saves the
        results to a specified file.

        Args:
            question (UnansweredQuestion): The question object containing the
            question text and ID.
            file (str): The file path where the search results will be saved.
        '''
        self.logger.log('Starting search...')

        if not question.question.strip():
            self.logger.warning(
                'Empty question provided, skipping search and saving empty '
                'results.'
            )
            self._save(
                StudentSearchResults(
                    search_results=[MinimalSearchResults(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=[]
                    )],
                    k=self.config.k
                ),
                file
            )
            return

        minimal_search_results: MinimalSearchResults = MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=self.search_sources(
                query=question,
                print_expanded=True,
                print_hyde=True
            )
        )

        self.logger.log('')
        self.logger.info(f'{Color.BOLD}Retrieved Sources:{Color.RESET}')
        self.print_retrieved_sources(minimal_search_results.retrieved_sources)
        self.logger.log('')

        self._save(
            StudentSearchResults(
                search_results=[minimal_search_results],
                k=self.config.k
            ),
            file
        )

    def search_dataset(self, dataset_path: str) -> None:
        '''
        Searches for sources for each question in the dataset and saves the
        results to a specified file.

        Args:
            dataset_path (str): The path to the dataset file containing
            questions to be searched.
        '''
        start_time: TimeUtils = TimeUtils()
        path: Path = Path(dataset_path)
        if not path.exists() or not path.is_file():
            self.logger.error(f'Dataset file {dataset_path!r} does not exist!')
            return

        dataset = self.dataset_interface.load_dataset(dataset_path)

        minimal_search_results: list[MinimalSearchResults] = []
        for question in tqdm(
            dataset.rag_questions,
            desc='Searching questions',
            unit='question'
        ):
            minimal_search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=self.search_sources(question)
                )
            )
            if self.app_config.verbose:
                self.logger.table_log(
                    headers=['#', 'File Path', 'Character Range'],
                    rows=[
                        [
                            f'{Color.YELLOW}{idx:<4}{Color.RESET}',
                            (str(source.file_path)
                             if len(str(source.file_path)) <= 63
                             else '…' + str(source.file_path)[-62:]),
                            f'{Color.WHITE}{source.first_character_index:<4} –'
                            f' {source.last_character_index:<4}{Color.RESET}'
                        ]
                        for idx, source in enumerate(
                            minimal_search_results[-1].retrieved_sources,
                            start=1
                        )
                    ]
                )

        self._save(
            StudentSearchResults(
                search_results=minimal_search_results,
                k=self.config.k
            ),
            path.name
        )

        self.logger.info(
            f' Processed {len(dataset.rag_questions)} '
            f'questions in {start_time.get_elapsed_time_formated()} !'
        )

    def _transform_to_sources(
                self,
                documents: list[str]
            ) -> list[MinimalSource]:
        '''
        Transforms a list of document IDs into a list of MinimalSource objects.

        Args:
            documents (list[str]): A list of document IDs to be transformed.
        Returns:
            list[MinimalSource]: A list of MinimalSource objects corresponding
            to the provided document IDs.
        '''
        return [
            self.chunks_interface.get_metadata_by_id(doc_id)
            for doc_id in documents
        ]

    def _filter_sources(
                self,
                sources: list[MinimalSource]
            ) -> list[MinimalSource]:
        '''
        Filters the provided sources based on the configured search type.

        Args:
            sources (list[MinimalSource]): A list of MinimalSource objects to
            be filtered.
        Returns:
            list[MinimalSource]: A list of MinimalSource objects that match the
            configured search type.
        '''
        return [
            source
            for source in sources
            if Path(source.file_path).suffix in FilesExt[
                self.config.search_type
            ]
        ]

    def _add_expanded_sources(
                self,
                k: int,
                query: str,
                documents_weights: list[tuple[list[MinimalSource], float]],
                print_expanded: bool = False
            ) -> None:
        '''
        Expands the query using the Dspy interface and adds the expanded
        sources to the documents_weights list.

        Args:
            k (int): The number of top results to retrieve.
            query (str): The original query string to be expanded.
            documents_weights (list[tuple[list[MinimalSource], float]]): A
            list of tuples containing lists of MinimalSource objects and their
            corresponding weights.
            print_expanded (bool, optional): Whether to print expanded query
            information. Defaults to False.
        '''

        expended_query = self.dspy_interface.expand_query_predict(query=query)

        if not print_expanded:
            self.logger.log_tqdm(
                'Expanded query BM25 keywords: '
                f'{expended_query.bm25_keywords!r}'
            )

            self.logger.log_tqdm(
                'Expanded query semantic queries: '
                f'{expended_query.semantic_queries!r}'
            )

        if print_expanded:
            self.logger.info('')
            self.logger.box_info(
                [f'BM25 keywords: {expended_query.bm25_keywords!r}'],
                'Expanded query BM25 keywords'
            )
            if self.app_config.use_chroma:
                self.logger.box_info(
                    [f'Semantic queries: {expended_query.semantic_queries!r}'],
                    'Expanded query semantic queries'
                )

        documents_weights.append((
            self._transform_to_sources(self.bm25s_interface.retrieve(
                expended_query.bm25_keywords,
                k
            )),
            self.app_config.rrf_weights_bm25_expanded
        ))
        if self.app_config.use_chroma:
            documents_weights.append((
                self._transform_to_sources(self.chromadb_interface.search(
                    expended_query.semantic_queries,
                    k
                )),
                self.app_config.rrf_weights_chroma_expanded
            ))

    def _add_HyDE(
                self,
                k: int,
                query: str,
                documents_weights: list[tuple[list[MinimalSource], float]],
                print_hyde: bool = False
            ) -> None:
        '''
        Generates a hypothetical passage using the Dspy interface and adds the
        corresponding sources to the documents_weights list.

        Args:
            k (int): The number of top results to retrieve.
            query (str): The original query string for which the hypothetical
            passage will be generated.
            documents_weights (list[tuple[list[MinimalSource], float]]): A
            list of tuples containing lists of MinimalSource objects and their
            corresponding weights.
            print_hyde (bool, optional): Whether to print HyDE information.
            Defaults to False.
        '''
        hyde = self.dspy_interface.hyde_predict(question=query)

        if not print_hyde:
            self.logger.log_tqdm(
                f'HyDE hypothetical passage: {hyde.hypothetical_passage!r}'
            )
        else:
            self.logger.info('')
            self.logger.box_info(
                [hyde.hypothetical_passage],
                'HyDE hypothetical passage'
            )
            self.logger.info('')

        documents_weights.append((
            self._transform_to_sources(self.chromadb_interface.search(
                hyde.hypothetical_passage,
                k
            )),
            self.app_config.rrf_weights_HyDE
        ))

    def _apply_rrf(
                self,
                documents: list[tuple[list[MinimalSource], float]],
                k: int
            ) -> list[MinimalSource]:
        '''
        Applies the Reciprocal Rank Fusion (RRF) algorithm to combine the
        results from different sources and returns the top k MinimalSource
        objects.

        Args:
            documents (list[tuple[list[MinimalSource], float]]): A list of
            tuples containing lists of MinimalSource objects and their
            corresponding weights.
            k (int): The number of top results to retrieve.
        Returns:
            list[MinimalSource]: A list of the top k MinimalSource objects
            after applying the RRF algorithm.
        '''
        scores: dict[MinimalSource, float] = {}
        for docs, weight in documents:
            for rank, doc_id in enumerate(self._filter_sources(docs)):
                score: float = weight * (1.0 / (k + rank + 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused_ids = [doc_id for doc_id, _ in ranked_docs]
        return fused_ids[:k]

    def _save(self, search_results: StudentSearchResults, file: str) -> None:
        '''
        Saves the search results to a JSON file in the specified save
        directory.

        Args:
            search_results (StudentSearchResults): The search results to be
            saved.
            file (str): The file name where the search results will be saved.
        '''
        save_path: Path = Path(self.config.save_directory) / file
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.log(
            f'Saving search results to {save_path.as_posix()!r}...'
        )

        JSONUtils.save_json(search_results.model_dump(), save_path.as_posix())

        self.logger.info(
            f' Saved → \'{Color.ITALIC}{save_path.as_posix()}{Color.RESET}\''
        )
