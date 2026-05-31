from pydantic import BaseModel


class Config(BaseModel):
    verbose: bool = False

    model_name: str = 'openai/qwen3:0.6b'
    use_chroma: bool = False

    chromadb_path: str = 'data/processed/chroma'
    chromadb_collection_name: str = 'chunks'
