from ..models import (
    MinimalSearchResults,
    StudentSearchResults,
    UnansweredQuestion,
    MinimalSource,
)
from ..interfaces import ChromaDBInterface, MetadataInterface, DatasetInterface
from ..utils import Logger, Color, JSONUtils
from ..config import Config

from pydantic import BaseModel, Field, model_validator
from pathlib import Path
import bm25s


class SearchConfig(BaseModel):
    k: int = Field(..., gt=0, le=100)
    save_directory: str = Field(..., min_length=1)

    @model_validator(mode='after')
    def check_paths_differ(self) -> 'SearchConfig':
        checks = [
            (self.save_directory, True),
        ]

        for path, should_be_dir in checks:
            p = Path(path)
            if p.exists() and p.is_dir() != should_be_dir:
                kind = "directory" if should_be_dir else "file"
                raise ValueError(f"{path} must be a {kind}!")

        return self


t_retrive_document = tuple[str, str, float]


class Search:
    logger: Logger
    app_config: Config
    chromadb_interface: ChromaDBInterface
    metadata_interface: MetadataInterface
    dataset_interface: DatasetInterface

    config: SearchConfig
    retriver: bm25s.BM25

    def __init__(
                self,
                k: int,
                save_directory: str,
                chromadb_interface: ChromaDBInterface,
                metadata_interface: MetadataInterface,
                dataset_interface: DatasetInterface,
                config: Config,
            ) -> None:
        self.app_config = config
        self.chromadb_interface = chromadb_interface
        self.metadata_interface = metadata_interface
        self.dataset_interface = dataset_interface

        self.logger = Logger('Search', Color.CYAN, config.verbose)
        self.logger.log('Initializing Search...')

        self.config = SearchConfig(
            save_directory=Path(save_directory).as_posix(),
            k=k,
        )

        self.retriver = bm25s.BM25.load(
            self.app_config.processed_bm25_index_path,
            load_corpus=True
        )

    def search(self, questions: list[UnansweredQuestion], file: str) -> None:
        self.logger.log('Starting search...')

        minimal_search_results: list[MinimalSearchResults] = []
        for question in questions:
            self.logger.log(f"Searching for question: {question.question!r}")

            fused_ids: list[str] = self._reciprocal_rank_fusion(
                [
                    (
                        self._get_bm25_results(question, self.config.k),
                        self.app_config.bm25_weights_rrf
                    ),
                    (
                        self._get_chromadb_results(question, self.config.k),
                        self.app_config.chroma_weights_rrf
                    )
                ],
                self.config.k
            )

            documents: list[MinimalSource] = []
            for doc_id in fused_ids:
                metadata = self.metadata_interface.get_by_id(doc_id)
                if metadata is not None:
                    documents.append(metadata)

            minimal_search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=documents
                )
            )

        self._save(
            StudentSearchResults(
                search_results=minimal_search_results,
                k=self.config.k
            ),
            file
        )

    def search_dataset(self, dataset_path: str) -> None:
        path: Path = Path(dataset_path)
        if not path.exists() or not path.is_file():
            self.logger.error(f"Dataset file {dataset_path!r} does not exist!")
            return

        dataset = self.dataset_interface.load_dataset(dataset_path)
        self.search(
            questions=dataset.rag_questions,
            file=path.name
        )

    def _get_bm25_results(
                self,
                question: UnansweredQuestion,
                k: int
            ) -> list[t_retrive_document]:
        query_tokens = bm25s.tokenize(question.question)
        docs, scores = self.retriver.retrieve(
            query_tokens=query_tokens,
            k=k
        )

        result: list[tuple[str, str, float]] = []
        for doc, score in zip(docs[0], scores[0]):
            ids: str = str(doc.get('id', ''))
            text: str = str(doc.get('text', ''))

            result.append((ids, text, score))

        return result

    def _get_chromadb_results(
                self,
                question: UnansweredQuestion,
                k: int
            ) -> list[t_retrive_document]:
        return self.chromadb_interface.search(
            query=question.question,
            k=k
        )

    def _reciprocal_rank_fusion(
                self,
                documents: list[tuple[list[t_retrive_document], float]],
                k: int
            ) -> list[str]:
        scores: dict[str, float] = {}
        for docs, weight in documents:
            sorted_docs: list[t_retrive_document] = sorted(
                docs, key=lambda x: x[2], reverse=True
            )
            for rank, (doc_id, _, _) in enumerate(sorted_docs):
                score: float = weight * (1.0 / (k + rank + 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused_ids = [doc_id for doc_id, _ in ranked_docs]
        return fused_ids[:k]

    def _save(self, search_results: StudentSearchResults, file: str) -> None:
        save_path: Path = Path(self.config.save_directory) / file
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f'Saving search results to {save_path.as_posix()!r}...'
        )

        JSONUtils.save_json(
            search_results.model_dump(),
            save_path.as_posix()
        )
