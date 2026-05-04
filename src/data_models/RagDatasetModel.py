from . import AnsweredQuestion, UnansweredQuestion
from pydantic import BaseModel


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]
