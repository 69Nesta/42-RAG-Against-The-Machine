from ..models import (
    MinimalSource, MinimalSearchResults, StudentSearchResults,
    RagDataset, AnsweredQuestion
)

from ..interfaces import (
    SearchResultsInterface,
    DatasetInterface,
    ChunksInterface
)
from ..utils import Logger, Color
from ..config import Config

from pydantic import BaseModel, Field, model_validator
from pathlib import Path


class EvaluateConfig(BaseModel):
    student_answer_path: str = Field(..., min_length=1)
    dataset_path: str = Field(..., min_length=1)

    @model_validator(mode='after')
    def check_paths_differ(self) -> 'EvaluateConfig':
        checks = [
            (self.student_answer_path, False),
            (self.dataset_path, False),
        ]

        for path, should_be_dir in checks:
            p = Path(path)
            if p.exists() and p.is_dir() != should_be_dir:
                kind = 'directory' if should_be_dir else 'file'
                raise ValueError(f'{path} must be a {kind}!')

        return self


class EvaluateModule:
    logger: Logger
    app_config: Config

    dataset_interface: DatasetInterface
    chunks_interface: ChunksInterface
    search_results_interface: SearchResultsInterface

    config: EvaluateConfig

    def __init__(
                self,
                student_answer_path: str,
                dataset_path: str,
                dataset_interface: DatasetInterface,
                chunks_interface: ChunksInterface,
                config: Config,
            ) -> None:
        self.app_config = config
        self.dataset_interface = dataset_interface
        self.chunks_interface = chunks_interface

        self.logger = Logger('EvaluateModule', Color.BLUE, config.verbose)
        self.logger.log('Initializing Evaluate Module...')

        self.config = EvaluateConfig(
            student_answer_path=Path(student_answer_path).as_posix(),
            dataset_path=Path(dataset_path).as_posix()
        )
        self.search_results_interface = SearchResultsInterface(config)

        self._evaluate()

    @staticmethod
    def overlap_ratio(doc: MinimalSource, origin: MinimalSource) -> float:
        overlap_start: int = max(
            doc.first_character_index,
            origin.first_character_index
        )
        overlap_end: int = min(
            doc.last_character_index,
            origin.last_character_index
        )

        overlap: int = max(0, overlap_end - overlap_start)

        origin_length: int = (
            origin.last_character_index - origin.first_character_index
        )

        return overlap / origin_length if origin_length > 0 else 0.0

    def doc_is_valid(self, doc: MinimalSource, origin: MinimalSource) -> bool:
        if Path(doc.file_path) != Path(origin.file_path):
            return False

        return self.overlap_ratio(doc, origin) >= 0.05

    def evaluate_search_recall_k(
                self,
                origins: list[MinimalSource],
                docs: list[MinimalSource],
                k: int
            ) -> float:
        founds: int = 0

        for origin in origins:
            if any([
                self.doc_is_valid(doc, origin)
                for doc in docs[:k]
            ]):
                founds += 1

        return founds / len(origins)

    def evaluate_search_mrr(
                self,
                origins: list[MinimalSource],
                docs: list[MinimalSource]
            ) -> float:

        for origin in origins:
            for idx, doc in enumerate(docs, start=1):
                if self.doc_is_valid(doc, origin):
                    return 1 / idx

        return 0.0

    def _evaluate(self) -> None:
        dataset: RagDataset = self.dataset_interface.load_dataset(
            self.config.dataset_path
        )
        student_search: StudentSearchResults = \
            self.search_results_interface.get_search_results(
                self.config.student_answer_path
            )
        student_search_map: dict[str, MinimalSearchResults] = {
            element.question_id: element
            for element in student_search.search_results
        }

        total_questions: int = len(dataset.rag_questions)

        invalid_questions: int = 0
        unfound_questions: int = 0
        processed_questions: int = 0
        valid_questions: dict[int, float] = {}
        total_mrr: float = .0

        recall_values = [1, 3, 5, 10]

        for question in dataset.rag_questions:
            if (not isinstance(question, AnsweredQuestion) or
               not len(question.sources)):
                invalid_questions += 1
                continue

            if not student_search_map.get(question.question_id):
                unfound_questions += 1
                continue

            processed_questions += 1
            for k in recall_values:
                recall = self.evaluate_search_recall_k(
                    question.sources,
                    student_search_map[question.question_id].retrieved_sources,
                    k
                )

                valid_questions.update({
                    k: valid_questions.get(k, 0) + recall
                })
            total_mrr += self.evaluate_search_mrr(
                question.sources,
                student_search_map[question.question_id].retrieved_sources
            )

        if invalid_questions > 0:
            self.logger.warning(
                f'Found {invalid_questions} invalid questions in the dataset!'
            )
        if unfound_questions > 0:
            self.logger.warning(
                f'Found {unfound_questions} questions not found in student '
                'answers!'
            )

        self.logger.info('')
        self.logger.box_info(
            [
                f'Total questions: {total_questions}',
                f'Processed questions: {processed_questions}',
                f'Invalid questions: {invalid_questions}',
                f'Unfound questions: {unfound_questions}',
            ],
            'Evaluation Summary'
        )
        self.logger.log('')

        metrics_rows: list[list[str]] = []
        for k in recall_values:
            recall_score: float = valid_questions.get(k, 0)
            divisor = processed_questions if processed_questions > 0 else 1
            recall_score /= divisor

            metrics_rows.append([
                f'Recall@{k} ',
                f'{recall_score:<6.3f}',
                f'{recall_score * 100:.1f}%',
            ])
        self.logger.info('')
        self.logger.table_info(
            headers=['Metric', 'Value', 'Percentage'],
            rows=metrics_rows
        )
        mrr: float = (
            total_mrr / processed_questions if processed_questions > 0 else .0
        )
        self.logger.info('')
        self.logger.box_info(
            [f'MRR: {mrr:.3f}'],
            'Mean Reciprocal Rank (MRR)',
            30
        )

        self.logger.log('')
