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
    k: int = Field(..., gt=0, le=10)
    save_directory: str = Field(..., min_length=1)
    search_type: FileType

    @model_validator(mode='after')
    def check_paths_differ(self) -> 'SearchConfig':
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
                query: UnansweredQuestion
            ) -> list[MinimalSource]:

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
                documents_weights
            )

        if self.app_config.use_hyde:
            self._add_HyDE(fake_k, query.question, documents_weights)

        return self._apply_rrf(documents_weights, self.config.k)

    def search(self, question: UnansweredQuestion, file: str) -> None:
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
            retrieved_sources=self.search_sources(question)
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
        return [
            self.chunks_interface.get_metadata_by_id(doc_id)
            for doc_id in documents
        ]

    def _filter_sources(
                self,
                sources: list[MinimalSource]
            ) -> list[MinimalSource]:
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
            ) -> None:
        expended_query = self.dspy_interface.expand_query_predict(query=query)

        self.logger.log_tqdm(
            'Expanded query BM25 keywords: '
            f'{expended_query.bm25_keywords!r}'
        )

        self.logger.log_tqdm(
            'Expanded query semantic queries: '
            f'{expended_query.semantic_queries!r}'
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
            ) -> None:
        hyde = self.dspy_interface.hyde_predict(question=query)

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
        scores: dict[MinimalSource, float] = {}
        for docs, weight in documents:
            for rank, doc_id in enumerate(self._filter_sources(docs)):
                score: float = weight * (1.0 / (k + rank + 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused_ids = [doc_id for doc_id, _ in ranked_docs]
        return fused_ids[:k]

    def _save(self, search_results: StudentSearchResults, file: str) -> None:
        save_path: Path = Path(self.config.save_directory) / file
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.log(
            f'Saving search results to {save_path.as_posix()!r}...'
        )

        JSONUtils.save_json(search_results.model_dump(), save_path.as_posix())

        self.logger.info(
            f' Saved → \'{Color.ITALIC}{save_path.as_posix()}{Color.RESET}\''
        )
