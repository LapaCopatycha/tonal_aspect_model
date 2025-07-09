import json
import torch

from configurations import BASE_DIR
from taa_model.transformer_model.preparation import prep_review
from taa_model.transformer_model.model import tokenizer, model

def result_to_dict(y, header):
    return {name: int(y[i]) for i, name in enumerate(header[:-1])}


def mark_review_transformer(review):

    # Вообще надо подумать как это закешировать так на загрузку этих заголовков будет тратиться очень много времени
    header_path = f'{BASE_DIR}/taa_model/data/header.json'
    with open(header_path, 'r', encoding='utf-8') as f:
        header = json.load(f)

    phrase_clear = prep_review(phrase=review)

    if len(phrase_clear.split()) < 2 :
        return result_to_dict(y=([0] * len(header)), header=header)

    tokenized_text = tokenizer(phrase_clear,truncation=True,padding='max_length',max_length=384,return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized_text) # Кстати ** это вроде тоже не эффективно по времени

    predict = torch.sigmoid(outputs.logits)
    y = (predict>0.5).int()

    return result_to_dict(y=y.squeeze(0), header=header)

def mark_review_lst_transformer (reviews:list):
    return [mark_review_transformer(review) for review in reviews]

# Нужно будет подумать над функцей активации так как есть проблема в том что два противоречивых класса могут выпасть
# Если оба больше 0.5 выбирается масксимальный из двух
# print(mark_review_transformer("Телефон работает отлично. Батарею не держит. Упакованно плохо"))