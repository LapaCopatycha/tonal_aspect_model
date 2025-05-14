import csv
import numpy as np

import torch.utils.data as data
from model import PhraseDataset, model, navec
from taa_model.model.metrics import model_eval
from taa_model.model.use import mark_review

# Что то тут ни как нормально не работает
# def t_model():
#     test_dataset = PhraseDataset("../data/test.csv", navec)
#     test_data = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
#     test_metrics = model_eval(model, test_data, loss_func=False)
#     return test_metrics
#
# print(t_model())

def t_model():
    path = '../data/test.csv'
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        dataset = list(reader)
        phrase_all = dataset[1:].copy()

    all_len = 0
    all_positive = 0
    for row in phrase_all:
        real_marks = np.array(row[0:-1], dtype=int)
        review = row[-1]

        model_marks = mark_review(review)
        model_marks = np.array(list(model_marks.values()))
        all_positive += sum(real_marks == model_marks)
        all_len += len(real_marks)
    hamming_loss = 1 - (all_positive / all_len)
    return hamming_loss

print(t_model())
