from ..utils import Logger, Color
from ..config import Config

from chromadb import PersistentClient, ClientAPI, Collection
from pathlib import Path


class ChromaDBInterface:
    logger: Logger
    config: Config

    enabled: bool
    client_path: str
    collection_name: str

    client: ClientAPI
    collection: Collection

    def __init__(
                self,
                config: Config,
            ) -> None:
        self.config = config

        self.enabled = config.use_chroma
        self.client_path = config.chromadb_path
        self.collection_name = config.chromadb_collection_name

        self.logger = Logger(
            'ChromaDBInterface',
            Color.BRIGHT_YELLOW,
            self.config.verbose
        )
        if not self.enabled:
            return
        Path(config.chromadb_path).mkdir(parents=True, exist_ok=True)

        self.logger.info(
            'Initializing Chroma interface with collection '
            f'name: {self.collection_name}'
        )
        self.client = PersistentClient(
            path=self.client_path
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def get_collection(self) -> Collection:
        if not self.enabled:
            raise ValueError("Chroma interface is not enabled.")
        return self.collection

    def get_client(self) -> ClientAPI:
        if not self.enabled:
            raise ValueError("Chroma interface is not enabled.")
        return self.client
