import torch
import torch.nn.functional as F
from sympy.physics.units import micro

def update_crosstab(crosstab, p, y):
    crosstab['true_positive'] += ((p == 1) & (y == 1)).sum(dim=0)
    crosstab['false_positive'] += ((p == 1) & (y == 0)).sum(dim=0)
    crosstab['false_negative'] += ((p == 0) & (y == 1)).sum(dim=0)
    crosstab['true_negative'] += ((p == 0) & (y == 0)).sum(dim=0)
    return crosstab

def macro_averange_metrics(crosstab, prefix=''):
    true_positive = crosstab['true_positive']
    false_positive = crosstab['false_positive']
    false_negative = crosstab['false_negative']
    true_negative = crosstab['true_negative']

    e = 0.001
    precision = true_positive / (true_positive + false_positive + e)
    recall = true_positive / (true_positive + false_negative+ e)
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)

    f1 = 2 * (precision * recall) / (precision + recall + e)

    precision = precision.mean().item()
    recall = recall.mean().item()
    accuracy = accuracy.mean().item()
    f1 = f1.mean().item()
    return {f'{prefix}precision' : precision, f'{prefix}recall' : recall,
            f'{prefix}accuracy' : accuracy, f'{prefix}f1' : f1}

def micro_averange_metrics(crosstab, prefix=''):
    true_positive = crosstab['true_positive']
    false_positive = crosstab['false_positive']
    false_negative = crosstab['false_negative']
    true_negative = crosstab['true_negative']

    e = 0.001
    # Микро среднее:
    precision = true_positive.sum() / (true_positive.sum() + false_positive.sum() + e)
    recall = true_positive.sum() / (true_positive.sum() + false_negative.sum() + e)
    accuracy = ((true_positive.sum() + true_negative.sum()) /
                (true_positive.sum() + false_positive.sum() + false_negative.sum() + true_negative.sum()))

    f1 = 2 * (precision * recall) / (precision + recall)

    precision = precision.item()
    recall = recall.item()
    accuracy = accuracy.item()
    f1 = f1.item()
    return {f'{prefix}precision' : precision, f'{prefix}recall' : recall,
            f'{prefix}accuracy' : accuracy, f'{prefix}f1' : f1}

# Мб переименовать функцию
def model_eval(model, data, loss_func):
    model.eval() # Точно оно здесь нужно
    Q_val = 0
    count_val = 0

    if loss_func is False:
        Q_val = None

    crosstab = { 'true_positive' : 0, 'false_positive' : 0,
                'false_negative' : 0, 'true_negative' : 0}

    modified_crosstab = crosstab.copy()

    for x, y in data:
        with torch.no_grad():
            predict = model(x.squeeze(0)).squeeze(0)
            y = y.squeeze(0)

            if loss_func is not False:
                loss = loss_func(predict, y)
                Q_val += loss.item()
                count_val += 1

            p = ((predict)>0.5).int()
            modified_p = features_activation_f(predict)
            crosstab = update_crosstab(crosstab, p, y)
            modified_crosstab = update_crosstab(modified_crosstab, modified_p, y)

    if loss_func is not False:
        Q_val /= count_val

    metrics = macro_averange_metrics(crosstab)
    modified_metrics = macro_averange_metrics(modified_crosstab, prefix='modified_')

    return {'Q_val':Q_val, **metrics, **modified_metrics}

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