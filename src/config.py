from pydantic import BaseModel, model_validator
from pathlib import Path


class Config(BaseModel):
    '''
    Configuration class for the RAG system.
    This class holds various configuration parameters that control the behavior
    of the system, including model settings, API keys, and paths for data
    storage.
    '''

    verbose: bool

    model_name: str
    temperature: float
    api_base: str
    api_key: str
    max_tokens: int
    dspy_cache: bool

    use_query_expansion: bool
    use_hyde: bool

    use_chroma: bool
    chromadb_path: str
    chromadb_collection_name: str

    bm25_k1: float
    bm25_b: float

    processed_bm25_index_path: str
    processed_chunks_path: str

    rrf_weights_bm25: float
    rrf_weights_chroma: float

    rrf_weights_bm25_expanded: float
    rrf_weights_chroma_expanded: float
    rrf_weights_HyDE: float

    @model_validator(mode='after')
    def check_paths_differ(self) -> 'Config':
        '''
        Validates that the paths for the processed BM25 index and processed
        chunks are different and that they are of the correct type (file or
        directory).
        Raises:
            ValueError: If the paths are the same or if they are not of the
            expected type (file or directory).
        '''

        if self.processed_bm25_index_path == self.processed_chunks_path:
            raise ValueError(
                'processed_bm25_index_path and processed_chunks_path must '
                'be different!'
            )

        checks = [
            (self.processed_bm25_index_path, True),
            (self.processed_chunks_path, False),
        ]

        for path, should_be_dir in checks:
            p = Path(path)
            if p.exists() and p.is_dir() != should_be_dir:
                kind = 'directory' if should_be_dir else 'file'
                raise ValueError(f'{path} must be a {kind}!')

        return self
