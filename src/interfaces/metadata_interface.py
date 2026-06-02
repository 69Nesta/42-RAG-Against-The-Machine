from ..models import MinimalSource
from ..utils import Logger, Color
from ..utils import JSONUtils
from ..config import Config


class MetadataInterface:
    logger: Logger
    config: Config

    loaded: bool
    _metadatas: dict[str, MinimalSource]

    def __init__(self, config: Config) -> None:
        self.logger = Logger('MetadataInterface', Color.BLUE, config.verbose)
        self.config = config

        self.loaded = False
        self._metadatas = {}

    def _load(self) -> None:
        self.logger.log('Loading metadata from disk...')
        metadatas_list = JSONUtils.load_json(
            self.config.processed_chunks_metadata_path
        )

        try:
            self._metadatas = {
                m_id: MinimalSource(**m) for m_id, m in metadatas_list.items()
            }
        except Exception as e:
            raise ValueError(
                f"Error loading metadata: {e}. Check the format of "
                f"{self.config.processed_chunks_metadata_path}."
            )

        self.loaded = True
        self.logger.log(f"Loaded {len(self._metadatas)} metadata entries.")

    def get_by_id(self, document_id: str | int) -> MinimalSource | None:
        if not self.loaded:
            self._load()

        return self._metadatas.get(str(document_id))
