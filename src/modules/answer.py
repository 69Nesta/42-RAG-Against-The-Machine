from ..models import (
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSource
)
from ..interfaces import (
    SearchResultsInterface,
    ChromaDBInterface,
    DatasetInterface,
    ChunksInterface,
    DspyInterface,
)
from ..utils import Logger, Color, JSONUtils
from ..config import Config

from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm
from pathlib import Path


class AnswerConfig(BaseModel):
    save_directory: str = Field(..., min_length=1)

    @model_validator(mode='after')
    def check_paths_differ(self) -> 'AnswerConfig':
        checks = [
            (self.save_directory, True),
        ]

        for path, should_be_dir in checks:
            p = Path(path)
            if p.exists() and p.is_dir() != should_be_dir:
                kind = 'directory' if should_be_dir else 'file'
                raise ValueError(f'{path} must be a {kind}!')

        return self


class AnswerModule:
    logger: Logger
    app_config: Config

    chromadb_interface: ChromaDBInterface
    dataset_interface: DatasetInterface
    chunks_interface: ChunksInterface
    dspy_interface: DspyInterface

    search_results_interface: SearchResultsInterface

    config: AnswerConfig

    def __init__(
                self,
                save_directory: str,
                chromadb_interface: ChromaDBInterface,
                dataset_interface: DatasetInterface,
                chunks_interface: ChunksInterface,
                config: Config,
            ) -> None:
        self.app_config = config
        self.chromadb_interface = chromadb_interface
        self.dataset_interface = dataset_interface
        self.chunks_interface = chunks_interface

        self.logger = Logger('AnwerModule', Color.CYAN, config.verbose)
        self.logger.log('Initializing Answer Module...')

        self.config = AnswerConfig(
            save_directory=Path(save_directory).as_posix(),
        )

        self.dspy_interface = DspyInterface(config)

    def _answer(
                self,
                question: UnansweredQuestion,
                sources: list[MinimalSource]
            ) -> AnsweredQuestion:
        self.logger.log_tqdm(f'Answering question: {question.question!r}...')

        documents: list[str] = [
            self.chunks_interface.get_chunk_by_metadata(source).content
            for source in sources
        ]

        dspy_answer = self.dspy_interface.predict(
            documents=documents,
            question=question.question,
        )

        answer = AnsweredQuestion(
            question_id=question.question_id,
            question=question.question,
            sources=sources,
            answer=dspy_answer.answer,
        )

        return answer

    def answer(self, question_str: str, k: int = 5) -> None:
        from ..modules.search import SearchModule

        question: UnansweredQuestion = UnansweredQuestion(
            question=question_str,
        )

        search_module: SearchModule = SearchModule(
            k,
            save_directory=self.config.save_directory,
            chromadb_interface=self.chromadb_interface,
            dataset_interface=self.dataset_interface,
            chunks_interface=self.chunks_interface,
            config=self.app_config
        )

        sources: list[MinimalSource] = search_module.search_sources(question)

        answer: AnsweredQuestion = self._answer(
            question=question,
            sources=sources
        )

        self.logger.info(f'Answer: {answer.answer!r}')

        path = Path(self.config.save_directory, 'answer.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        JSONUtils.save_json(
            answer.model_dump(),
            path.as_posix()
        )

    def answer_dataset(self, student_search_results_path: str) -> None:
        self.search_results_interface = SearchResultsInterface(self.app_config)
        loaded_results = self.search_results_interface.get_search_results(
            student_search_results_path
        )

        self.logger.info(
            f'Loaded {len(loaded_results.search_results)} '
            f'questions from {student_search_results_path!r}'
        )

        answers: StudentSearchResultsAndAnswer = StudentSearchResultsAndAnswer(
            search_results=[],
            k=loaded_results.k
        )

        self.logger.info('Answering questions...')
        for search_result in tqdm(
            loaded_results.search_results,
            desc='Answering questions',
            unit='question',
        ):
            question = UnansweredQuestion(
                question_id=search_result.question_id,
                question=search_result.question,
            )

            answer: AnsweredQuestion = self._answer(
                question=question,
                sources=search_result.retrieved_sources
            )
            self.logger.log_tqdm(f'Answer: {answer.answer!r}')
            answers.search_results.append(
                MinimalAnswer(
                    question_id=search_result.question_id,
                    question=search_result.question,
                    retrieved_sources=search_result.retrieved_sources,
                    answer=answer.answer
                )
            )

        self._save(answers, Path(student_search_results_path).name)

    def _save(self, answers: StudentSearchResultsAndAnswer, file: str) -> None:
        save_path: Path = Path(self.config.save_directory) / file
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f'Saving results to {save_path.as_posix()!r}...'
        )

        JSONUtils.save_json(
            answers.model_dump(),
            save_path.as_posix()
        )
