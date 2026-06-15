from ..utils import Logger, Color
from ..config import Config

from chromadb import PersistentClient, Collection
from chromadb.api import ClientAPI
from typing import Any, Callable
from pathlib import Path


class ChromaDBInterface:
    '''
    Interface for ChromaDB operations, including collection management,
    batch addition of documents, and searching for relevant documents.
    '''

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
        '''
        Initializes the ChromaDB interface with the provided configuration.
        '''
        self.config = config

        self.enabled = config.use_chroma or config.use_hyde
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
        self._initialize_collection()

    def _initialize_collection(self) -> None:
        '''
        Initializes the ChromaDB collection, creating it if it does not exist.
        '''
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def get_collection(self) -> Collection:
        '''
        Returns the ChromaDB collection, ensuring it is initialized.

        Returns:
            The ChromaDB collection instance.
        '''
        if not self.enabled:
            raise ValueError('Chroma interface is not enabled.')
        return self.collection

    def get_client(self) -> ClientAPI:
        '''
        Returns the ChromaDB client, ensuring it is initialized.

        Returns:
            The ChromaDB client instance.
        '''
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
        '''
        Adds documents to the ChromaDB collection in batches, with optional
        progress tracking.

        Args:
            collection: The ChromaDB collection to add documents to.
            ids: A list of unique identifiers for the documents.
            documents: A list of document texts to be added.
            metadatas: Optional list of metadata dictionaries corresponding to
            each document.
            embeddings: Optional list of embedding vectors corresponding to
            each document.
            batch_size: The number of documents to add in each batch.
            progress_bar_func: Optional function to track progress, called
            with the current count and total.
        '''
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

    def search(self, queries: list[str], k: int = 5) -> list[str]:
        '''
        Searches the ChromaDB collection for relevant documents based on the
        provided queries.

        Args:
            queries: A list of query strings to search for.
            k: The number of top results to return for each query.
        Returns:
            A list of document IDs that match the queries.
        '''
        if not self.enabled:
            return []

        results = self.get_collection().query(
            query_texts=queries,
            n_results=k,
        )
        ids = results.get('ids')
        if not ids:
            return []

        return ids[0]

    def clear(self) -> None:
        '''
        Clears the ChromaDB collection by deleting and reinitializing it.
        '''

        if not self.config.use_chroma:
            return

        self.client.delete_collection(self.config.chromadb_collection_name)
        self._initialize_collection()
