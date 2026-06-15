from ..utils import Logger, Color
from ..config import Config

from typing import cast
import bm25s
import os


class Bm25sInterface:
    logger: Logger
    app_config: Config
    _retriever: bm25s.BM25 | None
    _is_indexed: bool

    def __init__(self, config: Config):
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
        if self._retriever is None:
            self._retriever = bm25s.BM25(
                k1=self.app_config.bm25_k1,
                b=self.app_config.bm25_b
            )
        return self._retriever

    def index(self, corpus: list[str]) -> None:
        tokens = bm25s.tokenize(corpus)
        self.retriever.index(tokens)
        self._is_indexed = True
        self.logger.log(f'BM25 index built ({len(corpus)} docs)')

    def save(self) -> None:
        self._assert_indexed()
        path: str = self.app_config.processed_bm25_index_path
        self.logger.log(f'Saving BM25 index to {path!r}...')
        self.retriever.save(path)

    def load(self) -> None:
        path = self.app_config.processed_bm25_index_path
        self._retriever = bm25s.BM25.load(
            path,
            load_corpus=False
        )
        self._is_indexed = True
        self.logger.log(f'BM25 index loaded from {path!r}')

    def retrieve(self, queries: list[str], k: int) -> list[str]:
        self._assert_indexed()
        tokens = bm25s.tokenize(queries)
        results, _ = self.retriever.retrieve(tokens, k=k)
        return cast(list[str], results[0].tolist())

    def _assert_indexed(self) -> None:
        if not self._is_indexed:
            raise RuntimeError('BM25 retriever is not indexed yet')

    def clear(self) -> None:
        try:
            os.remove(self.app_config.processed_bm25_index_path)
        except Exception:
            pass

        self._retriever = None
        self._is_indexed = False
