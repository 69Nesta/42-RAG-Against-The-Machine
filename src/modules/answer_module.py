
from ..models import (
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
    ChunkContentModel,
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSource
)
from ..interfaces import (
    SearchResultsInterface,
    ChromaDBInterface,
    DatasetInterface,
    ChunksInterface,
    Bm25sInterface,
    DspyInterface,
)
from ..utils import Logger, Color, JSONUtils, TimeUtils
from ..enums import FileType
from ..config import Config

from pydantic import BaseModel, Field, model_validator
from pathlib import Path
from tqdm import tqdm
import os


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
                dspy_interface: DspyInterface,
                config: Config,
            ) -> None:
        self.app_config = config
        self.chromadb_interface = chromadb_interface
        self.dataset_interface = dataset_interface
        self.chunks_interface = chunks_interface
        self.dspy_interface = dspy_interface

        self.logger = Logger('AnwerModule', Color.CYAN, config.verbose)
        self.logger.log('Initializing Answer Module...')

        self.config = AnswerConfig(
            save_directory=Path(save_directory).as_posix(),
        )

    def _answer_pipeline(self, answer: str, sources_ids: list[str]) -> str:
        sources: list[ChunkContentModel] = []

        for source_id in sources_ids:
            try:
                sources.append(
                    self.chunks_interface.get_chunk_by_id(source_id)
                )
            except Exception:
                pass

        sources_paths: set[str] = set()

        for source in sources:
            sources_paths.add(source.metadata.file_path)

        answer = (
            answer
            .replace('[[ ## completed ## ]]', '')
            .strip()
        )

        if sources_paths:
            if answer and answer[-1] not in '.!?':
                answer += '.'

            answer += '\n\nSources:\n'
            answer += '\n'.join(f'- {path!r}' for path in sources_paths)

        return answer

    def _answer(
                self,
                question: UnansweredQuestion,
                sources: list[MinimalSource]
            ) -> AnsweredQuestion:
        self.logger.log_tqdm(f'Answering question: {question.question!r}...')

        documents: list[ChunkContentModel] = [
            self.chunks_interface.get_chunk_by_metadata(source)
            for source in sources
        ]

        dspy_answer = self.dspy_interface.answer_predict(
            documents=documents,
            question=question.question,
        )

        answer_str: str = self._answer_pipeline(
            dspy_answer.answer,
            dspy_answer.used_documents
        )

        answer = AnsweredQuestion(
            question_id=question.question_id,
            question=question.question,
            sources=sources,
            answer=answer_str,
        )

        return answer

    def answer(
                self,
                question_str: str,
                k: int,
                bm25s_interface: Bm25sInterface,
                search_type: FileType,
            ) -> None:
        from .search_module import SearchModule

        question: UnansweredQuestion = UnansweredQuestion(
            question=question_str
        )

        self.logger.info('')
        self.logger.box_info(
            [question.question],
            'Question'
        )
        self.logger.log('')

        search_module: SearchModule = SearchModule(
            k,
            save_directory=self.config.save_directory,
            search_type=search_type,
            chromadb_interface=self.chromadb_interface,
            dataset_interface=self.dataset_interface,
            chunks_interface=self.chunks_interface,
            bm25s_interface=bm25s_interface,
            dspy_interface=self.dspy_interface,
            config=self.app_config
        )
        sources: list[MinimalSource] = search_module.search_sources(
            question,
            print_expanded=True,
            print_hyde=True
        )

        self.logger.info('')
        self.logger.info(f'{Color.BOLD}Retrieved Sources:{Color.RESET}')
        self.logger.info('')
        self.logger.table_info(
            headers=['#', 'File', 'Char range'],
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
        self.logger.log('')

        answer: AnsweredQuestion = self._answer(
            question=question,
            sources=sources
        )

        self.logger.info('')
        self.logger.box_info(
            [answer.answer],
            f'Answer ({len(sources)} source{"s" if len(sources) != 1 else ""})'
        )

        path = Path(self.config.save_directory, 'answer.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        JSONUtils.save_json(answer.model_dump(), path.as_posix())
        self.logger.info('')
        self.logger.info(
            f'  Saved → \'{Color.ITALIC}{path}{Color.RESET}\''
        )

    def answer_dataset(self, student_search_results_path: str) -> None:
        start_time: TimeUtils = TimeUtils()
        self.search_results_interface = SearchResultsInterface(self.app_config)
        try:
            loaded_results = self.search_results_interface.get_search_results(
                student_search_results_path
            )
        except Exception:
            self.logger.error(
                f'Error while loading search results from '
                f'{student_search_results_path!r} !'
            )
            return

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

        self.logger.info(
            f'Answered {len(answers.search_results)} questions in '
            f'{start_time.get_elapsed_time_formated()} !'
        )

    def _save(self, answers: StudentSearchResultsAndAnswer, file: str) -> None:
        save_path: Path = Path(self.config.save_directory) / file
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.log(
            f'Saving results to {save_path.as_posix()!r}...'
        )

        JSONUtils.save_json(
            answers.model_dump(),
            save_path.as_posix()
        )
        self.logger.info(
            f'  Saved → \'{Color.ITALIC}{save_path.as_posix()}{Color.RESET}\''
        )
