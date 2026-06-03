from ..models import ChunkContentModel, MinimalSource
from ..utils import Logger, Color, JSONUtils
from ..config import Config


class ChunksInterface:
    logger: Logger
    app_config: Config

    _loaded: bool
    _chunks: dict[str, ChunkContentModel]

    def __init__(self, config: Config):
        self.app_config = config
        self.logger = Logger('ChunksInterface', Color.BLUE, config.verbose)

        self._loaded = False
        self._chunks = {}

    def save_chunks(self, chunks: dict[str, ChunkContentModel]) -> None:
        self.logger.info(
            f'Saving chunks to {self.app_config.processed_chunks_path}'
        )
        JSONUtils.save_json(
            chunks,
            self.app_config.processed_chunks_path,
        )

        self._loaded = True
        self._chunks = chunks

    def _load(self) -> None:
        if self._loaded:
            return

        self.logger.info(
            f'Loading chunks from {self.app_config.processed_chunks_path}'
        )
        chunks_dict = JSONUtils.load_json(
            self.app_config.processed_chunks_path
        )
        if not isinstance(chunks_dict, dict):
            raise ValueError(
                f'Expected a dict in {self.app_config.processed_chunks_path},'
                f' got {type(chunks_dict)}'
            )
        self._chunks = {
            k: ChunkContentModel(**v) for k, v in chunks_dict.items()
        }
        self._loaded = True

    def get_chunk(self, content_id: str) -> ChunkContentModel:
        self._load()

        if content_id not in self._chunks:
            raise ValueError(f'Content ID {content_id} not found in chunks.')

        return self._chunks[content_id]

    def get_metadata(self, content_id: str) -> MinimalSource:
        return self.get_chunk(content_id).metadata
