from ..models import (
    StudentSearchResults,
    UnansweredQuestion,
    AnsweredQuestion,
    MinimalSource
)
from ..interfaces import ChunksInterface, DatasetInterface, DspyInterface
from ..utils import Logger, Color, JSONUtils
from ..config import Config

from pydantic import BaseModel, Field, model_validator
from pathlib import Path


class AnswerConfig(BaseModel):
    k: int = Field(..., gt=0, le=100)
    save_directory: str = Field(..., min_length=1)

    @model_validator(mode='after')
    def check_paths_differ(self) -> 'AnswerConfig':
        checks = [
            (self.save_directory, True),
        ]

        for path, should_be_dir in checks:
            p = Path(path)
            if p.exists() and p.is_dir() != should_be_dir:
                kind = "directory" if should_be_dir else "file"
                raise ValueError(f"{path} must be a {kind}!")

        return self


class Answer:
    logger: Logger
    app_config: Config

    dataset_interface: DatasetInterface
    chunks_interface: ChunksInterface
    dspy_interface: DspyInterface

    config: AnswerConfig

    def __init__(
                self,
                k: int,
                save_directory: str,
                dataset_interface: DatasetInterface,
                chunks_interface: ChunksInterface,
                config: Config,
            ) -> None:
        self.app_config = config
        self.dataset_interface = dataset_interface
        self.chunks_interface = chunks_interface

        self.logger = Logger('Anwer', Color.CYAN, config.verbose)
        self.logger.log('Initializing Answer...')

        self.config = AnswerConfig(
            save_directory=Path(save_directory).as_posix(),
            k=k,
        )

        self.dspy_interface = DspyInterface(config)

    def _answer(
                self,
                question: UnansweredQuestion,
                sources: list[MinimalSource]
            ) -> AnsweredQuestion:
        self.logger.log(f"Answering question: {question.question!r}...")

        raise NotImplementedError('Not implemented yet!')

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
