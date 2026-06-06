from ..models import (
    MinimalSource, MinimalSearchResults, StudentSearchResults,
    RagDataset, AnsweredQuestion
)

from ..interfaces import (
    SearchResultsInterface,
    DatasetInterface,
    ChunksInterface
)
from ..utils import Logger, Color, MathUtils
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
    def _doc_is_in_range(doc: MinimalSource, origin: MinimalSource) -> bool:
        is_in_range: bool = all([
            MathUtils.is_in_range(
                doc.first_character_index,
                origin.first_character_index,
                5.0
            ),
            MathUtils.is_in_range(
                doc.last_character_index,
                origin.last_character_index,
                5.0
            )
        ])
        is_inside: bool = (
            doc.first_character_index <= origin.first_character_index and
            doc.last_character_index >= origin.last_character_index
        )
        return any([is_in_range, is_inside])

    def _evaluate_sources(
                self,
                original_doc: MinimalSource,
                student_docs: list[MinimalSource],
                k: int
            ) -> bool:
        for doc in student_docs[:k]:
            if Path(doc.file_path) != Path(original_doc.file_path):
                continue
            if self._doc_is_in_range(doc, original_doc):
                return True

        return False

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
        valid_questions: dict[int, int] = {}
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
                is_valid = self._evaluate_sources(
                    question.sources[0],
                    student_search_map[question.question_id].retrieved_sources,
                    k
                )

                valid_questions.update({
                    k: valid_questions.get(k, 0) + is_valid
                })

        # Log any data quality issues
        if invalid_questions > 0:
            self.logger.warning(
                f'Found {invalid_questions} invalid questions in the dataset!'
            )
        if unfound_questions > 0:
            self.logger.warning(
                f'Found {unfound_questions} questions not found in student '
                'answers!'
            )

        # Display evaluation results
        self.logger.log('')
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

        self.logger.table_info(
            headers=['Metric', 'Value', 'Percentage'],
            rows=metrics_rows
        )

        self.logger.log('')
