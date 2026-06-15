from ..utils import Logger, Color
from ..config import Config

from typing import cast
import bm25s
import os


class Bm25sInterface:
    '''
    A class that provides an interface for indexing, saving, loading, and
    retrieving documents using the BM25 algorithm.
    '''
    logger: Logger
    app_config: Config
    _retriever: bm25s.BM25 | None
    _is_indexed: bool

    def __init__(self, config: Config):
        '''
        Initializes the Bm25sInterface with the given configuration.

        Args:
            config (Config): The application configuration containing BM25
            parameters and file paths.
        '''
        self.app_config = config
        self.logger = Logger(
            'Bm25sInterface',
            Color.BRIGHT_GREEN,
            config.verbose
        )
        self._retriever = None
        self._is_indexed = False

    @property
    def retriever(self) -> bm25s.BM25:
        '''
        Returns the BM25 retriever instance, initializing it if it hasn't been
        done yet.
        '''
        if self._retriever is None:
            self._retriever = bm25s.BM25(
                k1=self.app_config.bm25_k1,
                b=self.app_config.bm25_b
            )
        return self._retriever

    def index(self, corpus: list[str]) -> None:
        '''
        Indexes the given corpus of documents using the BM25 algorithm.

        Args:
            corpus (list[str]): A list of documents to be indexed.
        '''
        tokens = bm25s.tokenize(corpus)
        self.retriever.index(tokens)
        self._is_indexed = True
        self.logger.log(f'BM25 index built ({len(corpus)} docs)')

    def save(self) -> None:
        '''
        Saves the BM25 index to a file.

        '''
        self._assert_indexed()
        path: str = self.app_config.processed_bm25_index_path
        self.logger.log(f'Saving BM25 index to {path!r}...')
        self.retriever.save(path)

    def load(self) -> None:
        '''
        Loads the BM25 index from a file.
        '''
        path = self.app_config.processed_bm25_index_path
        self._retriever = bm25s.BM25.load(
            path,
            load_corpus=False
        )
        self._is_indexed = True
        self.logger.log(f'BM25 index loaded from {path!r}')

    def retrieve(self, queries: list[str], k: int) -> list[str]:
        '''
        Retrieves the top K documents for each query in the given list of
        queries using the BM25 algorithm.

        Args:
            queries (list[str]): A list of queries for which to retrieve
            documents.
            k (int): The number of top documents to retrieve for each query.
        '''
        self._assert_indexed()
        tokens = bm25s.tokenize(queries)
        if not queries:
            return []
        results, _ = self.retriever.retrieve(tokens, k=k)
        return cast(list[str], results[0].tolist())

    def _assert_indexed(self) -> None:
        '''
        Asserts that the BM25 retriever has been indexed before performing
        retrieval or saving operations. Raises a RuntimeError if the retriever
        is not indexed.
        '''

        if not self._is_indexed:
            raise RuntimeError('BM25 retriever is not indexed yet')

    def clear(self) -> None:
        '''
        Clears the BM25 index and resets the retriever state. This method also
        removes the saved BM25 index file if it exists.
        '''

        try:
            os.remove(self.app_config.processed_bm25_index_path)
        except Exception:
            pass

        self._retriever = None
        self._is_indexed = False
