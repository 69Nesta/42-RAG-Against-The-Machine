from pydantic import BaseModel
from . import MinimalSearchResults, MinimalAnswer


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: list[MinimalAnswer]
