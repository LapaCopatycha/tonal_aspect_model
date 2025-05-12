from pydantic import BaseModel

class SReview(BaseModel):
    phrase: str

class SMarkedReview(BaseModel):
    marks: dict