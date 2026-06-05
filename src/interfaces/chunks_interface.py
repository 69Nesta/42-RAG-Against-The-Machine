from ..models import ChunkContentModel, MinimalSource
from ..utils import Logger, Color, JSONUtils
from ..config import Config


class ChunksInterface:
    logger: Logger
    app_config: Config

    _loaded: bool
    _chunks: dict[str, ChunkContentModel]
    _chunks_by_metadata: dict[MinimalSource, ChunkContentModel]

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
            default=lambda o: o.model_dump()
        )

        self._loaded = True
        self._chunks = chunks
        self._generate_chunks_by_metadata()

    def _load(self) -> None:
        if self._loaded:
            return

        self.logger.log_tqdm(
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
        self._generate_chunks_by_metadata()

    def _generate_chunks_by_metadata(self) -> None:
        self._chunks_by_metadata = {
            chunk.metadata: chunk for chunk in self._chunks.values()
        }

    def get_chunk_by_id(self, content_id: str) -> ChunkContentModel:
        self._load()

        if content_id not in self._chunks:
            raise ValueError(f'Content ID {content_id} not found in chunks.')

        return self._chunks[content_id]

    def get_metadata_by_id(self, content_id: str) -> MinimalSource:
        return self.get_chunk_by_id(content_id).metadata

    def get_chunk_by_metadata(
                self,
                metadata: MinimalSource
            ) -> ChunkContentModel:
        self._load()
        chunks: ChunkContentModel | None = self._chunks_by_metadata.get(
            metadata
        )

        if chunks is None:
            raise ValueError(f'Metadata {metadata} not found in chunks.')
        return chunks
