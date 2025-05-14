import csv
import numpy as np

from taa_model.model.use import mark_review
from configurations import BASE_DIR

def t_model():
    path = f'{BASE_DIR}/taa_model/data/test.csv'
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