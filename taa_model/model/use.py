import csv
import json

import torch
import torch.nn.functional as F

from paths import BASE_DIR
from taa_model.model.model import model, navec
from taa_model.model.preparation import prep_review


def mark_review(review):
    model_path = f'{BASE_DIR}/taa_model/model_history/model_lstm_bidir.tar'
    st = torch.load(model_path, weights_only=True)
    model.load_state_dict(st)

    # Вообще надо подумать как это закешировать так на загрузку этих заголовков будет тратиться очень много времени
    headers_path = f'{BASE_DIR}/taa_model/data/headers.json'
    with open(headers_path, 'r', encoding='utf-8') as f:
        header = json.load(f)

    phrase_lst = prep_review(phrase=review, navec_emb=navec)
    _data_batch = torch.stack(phrase_lst)


    predict = model(_data_batch.unsqueeze(0)).squeeze(0)
    p = F.sigmoid(predict)
    y = (p>0.5).int()

    return {name: int(y[i]) for i, name in enumerate(header[:-1])}

def mark_review_lst (reviews:list):
    return [mark_review(review) for review in reviews]

# phrase = "Этот телефон не работает. Батарейку не держит. Упакованно ужасно"
phrase = "Этот телефон работает. Батарейку держит. Упакованно круто"
print(mark_review(review=phrase))

# Нужно будет подумать над функцей активации так как есть проблема в том что два противоречивых класса могут выпасть
# Если оба больше 0.5 выбирается масксимальный из двух