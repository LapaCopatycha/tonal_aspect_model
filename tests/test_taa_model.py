import pytest
from taa_model.model.use import mark_review

phrase_lst = [
    "Этот телефон работает. Батарейку держит. Упакованно круто",
    "Этот телефон не работает. Батарейку не держит. Упакованно ужасно",
    "Херня",
    "",
    "гцщвоцвлцжцьцжюфюфьф фдвоцзвщжй вадожвдцзц"
]

@pytest.mark.parametrize("phrase", phrase_lst)
def test_mark_review_dict(phrase):
    assert type(mark_review(review=phrase)) == dict

@pytest.mark.parametrize("phrase", phrase_lst)
def test_mark_review_len(phrase):
    marked_values = mark_review(review=phrase).values()
    assert len(marked_values) != 0

@pytest.mark.parametrize("phrase", phrase_lst)
def test_mark_review_values(phrase):
    acceptable_values = [0,1]
    marked_values = mark_review(review=phrase).values()
    res = [True if v in acceptable_values else False for v in marked_values]
    assert  (len(res) == sum(res)) and (len(res) != 0)

