import re
import csv
import json

from navec import Navec
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

from paths import BASE_DIR
from taa_model.model.preparation import prep_review



class PhraseDataset(data.Dataset):
    def __init__(self, path, navec_emb, batch_size=8):
        self.navec_emb = navec_emb
        self.batch_size = batch_size

        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            dataset = list(reader)
            header = dataset[0].copy()
            phrase_all = dataset[1:].copy() # Нужно ли копировать
            self.save_header(header=header)
            self._clear_phrase(phrase_all)

        self.phrase_lst = [_pr for _pr in phrase_all if len(_pr[-1]) > 1] # Добавил чтобы не было однословных отзывов
        self.phrase_lst.sort(key=lambda _x: len(_x[-1]))
        self.dataset_len = len(self.phrase_lst)

    def save_header(self, header):
        with open('../data/headers.json', 'w') as file:
            json.dump(header, file)

    def _clear_phrase(self, p_lst):
        for _i, _p in enumerate(p_lst):
            _words = prep_review(phrase=_p[-1], navec_emb=self.navec_emb)
            p_lst[_i][-1] = _words

    def __getitem__(self, item):
        item *= self.batch_size
        item_last = item + self.batch_size
        if item_last > self.dataset_len:
            item_last = self.dataset_len

        _data = []
        _target = []
        max_length = len(self.phrase_lst[item_last-1][-1])

        for i in range(item, item_last):
            words_emb = []
            phrase = self.phrase_lst[i]
            length = len(phrase[-1])

            for k in range(max_length):
                t = phrase[-1][k] if k < length else torch.zeros(300)
                words_emb.append(t)

            _data.append(torch.vstack(words_emb))
            _target.append(torch.tensor(list(map(int, phrase[0:-1])), dtype=torch.float64))

        _data_batch = torch.stack(_data)
        _target = torch.vstack(_target)
        return _data_batch, _target

    def __len__(self):
        last = 0 if self.dataset_len % self.batch_size == 0 else 1
        return self.dataset_len // self.batch_size + last

class WordsRNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 20
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.LSTM(in_features, self.hidden_size, batch_first=True, bidirectional=True, dropout=0.5, num_layers=1)
        self.out = nn.Linear(self.hidden_size * 4, out_features)
    def forward(self, x):
        x, (h, c) = self.rnn(x)
        hc = torch.cat((h[-2, :, :], h[-1, :, :], c[-2, :, :], c[-1, :, :]), dim=1)
        y = self.out(hc)
        return y

path = f'{BASE_DIR}/taa_model/emb/navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)

model = WordsRNN(300, 14)