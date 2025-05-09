from fastapi import FastAPI

from src.schemes import SReview, SMarkedReview
from taa_model.model.use import mark_review


app = FastAPI()

@app.post("/mark_review")
def mark_feedback (review:SReview):
    marks = mark_review(review=review.phrase)
    return SMarkedReview(marks=marks)