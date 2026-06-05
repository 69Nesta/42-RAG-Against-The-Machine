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
                doc.first_character_index,
                origin.first_character_index,
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

        numbers_of_invalid_question: int = 0
        numbers_of_unfound_question: int = 0
        numbers_of_processed_question: int = 0
        numbers_of_valid_question: dict[int, int] = {}
        recall = [1, 3, 5, 10]

        for question in dataset.rag_questions:
            if (not isinstance(question, AnsweredQuestion) or
               not len(question.sources)):
                numbers_of_invalid_question += 1
                continue

            if not student_search_map.get(question.question_id):
                numbers_of_unfound_question += 1
                continue

            numbers_of_processed_question += 1
            for k in recall:
                is_valid = self._evaluate_sources(
                    question.sources[0],
                    student_search_map[question.question_id].retrieved_sources,
                    k
                )

                numbers_of_valid_question.update({
                    k: numbers_of_valid_question.get(k, 0) + is_valid
                })

        if numbers_of_invalid_question > 0:
            self.logger.warning(
                f'Found {numbers_of_invalid_question} invalid questions in the'
                ' dataset!'
            )
        if numbers_of_unfound_question > 0:
            self.logger.warning(
                f'Found {numbers_of_unfound_question} questions in the dataset'
                ' that were not found in the student answers!'
            )

        self.logger.log('')
        self.logger.info('Evaluation Results')
        self.logger.info('========================================')
        self.logger.info(
            f'Questions evaluated: {numbers_of_processed_question}'
        )

        for k in recall:
            recall_value: float = numbers_of_valid_question.get(k, 0)
            recall_value /= numbers_of_processed_question

            self.logger.info(
                f'Recall@{k}: {recall_value:.3f} ({recall_value * 100:.1f}%)'
            )
