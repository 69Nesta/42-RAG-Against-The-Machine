from pydantic import BaseModel, model_validator
from pathlib import Path


class Config(BaseModel):
    verbose: bool = False

    model_name: str = 'openai/qwen3:0.6b'
    use_chroma: bool = False

    chromadb_path: str = 'data/processed/chroma'
    chromadb_collection_name: str = 'chunks'

    processed_bm25_index_path: str = 'data/processed/bm25_index'
    processed_chunks_path: str = 'data/processed/chunks'
    processed_chunks_metadata_path: str = \
        'data/processed/chunks/chunks_metadata.json'

    bm25_weights_rrf: float = 1.15
    chroma_weights_rrf: float = .85

    @model_validator(mode='after')
    def check_paths_differ(self) -> 'Config':
        if self.processed_bm25_index_path == self.processed_chunks_path:
            raise ValueError(
                'processed_bm25_index_path and processed_chunks_path must '
                'be different!'
            )

        checks = [
            (self.processed_bm25_index_path, True),
            (self.processed_chunks_path, True),
            (self.processed_chunks_metadata_path, False),
        ]

        for path, should_be_dir in checks:
            p = Path(path)
            if p.exists() and p.is_dir() != should_be_dir:
                kind = "directory" if should_be_dir else "file"
                raise ValueError(f"{path} must be a {kind}!")

        return self
