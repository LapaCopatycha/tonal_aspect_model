import torch
import torch.nn.functional as F

# Мб переименовать функцию
def model_eval(model, data, loss_func):
    model.eval() # Точно оно здесь нужно
    Q_val = 0
    count_val = 0

    if loss_func is False:
        Q_val = None
    total_incorrect_hl = 0
    total_incorrect_mhl = 0
    total_samples = 0

    for x, y in data:
        with torch.no_grad():
            predict = model(x.squeeze(0)).squeeze(0)

            if loss_func is not False:
                loss = loss_func(predict, y.squeeze(0))
                Q_val += loss.item()
                count_val += 1


            total_incorrect_hl += (((predict)>0.5).int() != y.squeeze(0)).sum().item()
            total_incorrect_mhl += (features_activation_f(predict) != y.squeeze(0)).sum().item()
            total_samples += y.squeeze(0).size(dim=0) * y.squeeze(0).size(dim=1)
            # Как то топорно
            # мб есть лучше посчитать колличество элементов

    if loss_func is not False:
        Q_val /= count_val
    hamming_loss = total_incorrect_hl / total_samples
    modifed_hamming_loss = total_incorrect_mhl / total_samples
    return {'Q_val':Q_val, 'hamming_loss' : hamming_loss, 'modifed_hamming_loss' : modifed_hamming_loss}

def features_activation_f(predict, activate=0.5 , q_features=7):
    new_prediction = []
    for element in torch.reshape(predict, (len(predict), q_features, 2)):
        new_element = []
        for feature in element:
            if (feature>activate).int().sum() == 2:
                if feature[0] > feature[1]:
                    res = torch.tensor([1,0])
                else:
                    res = torch.tensor([0,1])
            else:
                res = (feature>activate).int()
            new_element.extend(res)
        new_prediction.append(new_element)
    return torch.tensor(new_prediction)