from .minimal_source_model import MinimalSource
from pydantic import BaseModel


class ChunkContentModel(BaseModel):
    id: str
    content: str
    metadata: MinimalSource
