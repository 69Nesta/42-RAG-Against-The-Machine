from pydantic import BaseModel, ConfigDict


class MinimalSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_path: str
    first_character_index: int
    last_character_index: int
