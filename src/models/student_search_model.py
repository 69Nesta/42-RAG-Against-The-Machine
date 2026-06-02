from .minimal_answer_model import MinimalSearchResults
from .minimal_answer_model import MinimalAnswer
from pydantic import BaseModel


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    k: int
    search_results: list[MinimalAnswer]
