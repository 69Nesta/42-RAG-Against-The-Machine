from ..models import ChunkContentModel, MinimalSource
from ..utils import Logger, Color, JSONUtils
from ..config import Config


class ChunksInterface:
    '''
    Interface for managing chunks of content, including loading, saving,
    and retrieving chunks by ID or metadata.
    '''
    logger: Logger
    app_config: Config

    _loaded: bool
    _chunks: dict[str, ChunkContentModel]
    _chunks_by_metadata: dict[MinimalSource, ChunkContentModel]

    def __init__(self, config: Config):
        '''
        Initializes the ChunksInterface with the provided configuration.

        Args:
            config: The application configuration containing paths and
            settings.
        '''
        self.app_config = config
        self.logger = Logger(
            'ChunksInterface',
            Color.BRIGHT_CYAN,
            config.verbose
        )

        self._loaded = False
        self._chunks = {}

    def save_chunks(self, chunks: dict[str, ChunkContentModel]) -> None:
        '''
        Saves the provided chunks to the file system in JSON format.

        Args:
            chunks: A dictionary mapping chunk IDs to ChunkContentModel
            instances.
        '''
        self.logger.info(
            f'Saving chunks to {self.app_config.processed_chunks_path!r}'
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
        '''
        Loads chunks from the file system if they have not already been loaded.
        '''
        if self._loaded:
            return

        self.logger.log_tqdm(
            f'Loading chunks from {self.app_config.processed_chunks_path!r}'
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
        '''
        Generates a mapping of chunk metadata to chunk content for quick
        retrieval based on metadata.
        '''
        self._chunks_by_metadata = {
            chunk.metadata: chunk for chunk in self._chunks.values()
        }

    def get_chunk_by_id(self, content_id: str) -> ChunkContentModel:
        '''
        Retrieves a chunk of content by its unique identifier.

        Args:
            content_id: The unique identifier of the chunk to retrieve.
        Returns:
            The ChunkContentModel instance corresponding to the provided ID.
        Raises:
            ValueError: If the chunk with the specified ID is not found.
        '''
        self._load()
        chunk_id: str = str(content_id)

        if chunk_id not in self._chunks:
            raise ValueError(f'Content ID {chunk_id!r} not found in chunks.')

        return self._chunks[chunk_id]

    def get_metadata_by_id(self, chunk_id: str) -> MinimalSource:
        '''
        Retrieves the metadata associated with a chunk of content by its
        unique identifier.

        Args:
            chunk_id: The unique identifier of the chunk whose metadata is to
            be retrieved.
        Returns:
            The MinimalSource metadata associated with the specified chunk ID.
        Raises:
            ValueError: If the chunk with the specified ID is not found.
        '''

        return self.get_chunk_by_id(chunk_id).metadata

    def get_chunk_by_metadata(
                self,
                metadata: MinimalSource
            ) -> ChunkContentModel:
        '''
        Retrieves a chunk of content based on its associated metadata.

        Args:
            metadata: The MinimalSource metadata associated with the chunk to
            retrieve.
        Returns:
            The ChunkContentModel instance corresponding to the provided
            metadata.
        Raises:
            ValueError: If no chunk with the specified metadata is found.
        '''
        self._load()
        chunks: ChunkContentModel | None = self._chunks_by_metadata.get(
            metadata
        )

        if chunks is None:
            raise ValueError(f'Metadata {metadata} not found in chunks.')
        return chunks
