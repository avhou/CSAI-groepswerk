from pydantic import BaseModel

class YearQueryResponse(BaseModel):
    YEAR: int
