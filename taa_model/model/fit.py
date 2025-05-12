from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from metrics import model_eval
from model import PhraseDataset, model, navec
from config import BASE_DIR


def fit(epochs=10, p_valid=0.3, save_model=False):

    p_dataset = PhraseDataset("../data/prepared_reviews.csv", navec)
    d_train , d_valid = data.random_split(p_dataset, [1 - p_valid, p_valid])
    train_data = data.DataLoader(d_train, batch_size=1, shuffle=True)
    train_data_val = data.DataLoader(d_valid, batch_size=1, shuffle=False)


    optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001) # Мб его тоже заменить
    loss_func = nn.CrossEntropyLoss()


    loss_lst_val = [] # список значений потерь при валидации
    loss_lst = [] # список значений потерь при обучении
    hamming_loss_lst = []
    modifed_hamming_loss_lst = []

    for _e in range(epochs):
        model.train()
        loss_mean = 0
        lm_count = 0

        train_tqdm = tqdm(train_data, leave=True)
        for x_train, y_train in train_tqdm:
            predict = model(x_train.squeeze(0)).squeeze(0)
            loss = loss_func(predict, y_train.squeeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lm_count += 1
            loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean # А правильно ли она считается
            train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")

        # Валидация
        model_val = model_eval(model=model, data=train_data_val, loss_func=loss_func)
        Q_val = model_val['Q_val']
        hamming_loss = model_val['hamming_loss']
        modifed_hamming_loss =  model_val['modifed_hamming_loss']

        print(f" | loss_mean={loss_mean:.3f}, Q_val={Q_val:.3f}, hamming_loss={hamming_loss:.3f}"
              f", modifed_hamming_loss={modifed_hamming_loss:.3f}")

        loss_lst.append(loss_mean)
        loss_lst_val.append(Q_val)
        hamming_loss_lst.append(hamming_loss)
        modifed_hamming_loss_lst.append(modifed_hamming_loss)

    if save_model is True:
        st = model.state_dict()
        torch.save(st, f'{BASE_DIR}/taa_model/model_history/model_lstm_bidir.tar') # Изменить путь

    return {'loss_lst_val' : loss_lst_val, 'loss_lst' : loss_lst, 'hamming_loss_lst' : hamming_loss_lst,
            'modifed_hamming_loss_lst':modifed_hamming_loss_lst}

results = fit(epochs=10, p_valid=0.3, save_model=True)