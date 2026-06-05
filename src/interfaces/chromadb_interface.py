from ..utils import Logger, Color
from ..config import Config

from chromadb import PersistentClient, Collection
from chromadb.api import ClientAPI
from typing import Any, Callable
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

        self.logger.log(
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
            raise ValueError('Chroma interface is not enabled.')
        return self.collection

    def get_client(self) -> ClientAPI:
        if not self.enabled:
            raise ValueError('Chroma interface is not enabled.')
        return self.client

    def batch_add(
                self,
                collection: Collection,
                ids: list[str],
                documents: list[str],
                metadatas: list[dict[str, Any]] | None = None,
                embeddings: list[list[float]] | None = None,
                batch_size: int = 128,
                progress_bar_func: Callable[[int, int], None] | None = None,
            ) -> None:
        total = len(ids)

        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)

            kwargs: dict[str, Any] = {
                'ids': ids[i:end],
                'documents': documents[i:end],
            }

            if embeddings is not None:
                kwargs['embeddings'] = embeddings[i:end]

            if metadatas is not None:
                kwargs['metadatas'] = metadatas[i:end]

            collection.add(**kwargs)

            if progress_bar_func is not None:
                progress_bar_func(end, total)

    def search(self, query: str, k: int = 5) -> list[tuple[str, str, float]]:
        if not self.enabled:
            return []

        results = self.get_collection().query(
            query_texts=[query],
            n_results=k,
        )

        ids = results.get('ids')
        documents = results.get('documents')
        distances = results.get('distances')
        if not ids or not distances or not documents:
            return []

        return list(zip(
            ids[0],
            documents[0],
            distances[0]
        ))
