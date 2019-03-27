import numpy as np

import fire
from ner_data import label2idx


def eval(filename, eps=1e-9):
    y_true = np.array([label2idx[line.split()[0]]
                       for line in open(filename, 'r').read().splitlines()
                       if len(line.split()) > 2])
    y_pred = np.array([label2idx[line.split()[1]]
                       for line in open(filename, 'r').read().splitlines()
                       if len(line.split()) > 2])

    all_pos = len(y_pred[y_pred>1])
    actual_pos = len(y_true[y_true>1])
    correct_pos =(np.logical_and(y_true==y_pred, y_true>1)).sum().item()
    print(f'{all_pos} - {actual_pos} -> {correct_pos}')
    prec = correct_pos / (all_pos + eps)
    rec = correct_pos / (actual_pos + eps)
    f1 = (2*prec*rec)/(prec+rec+eps)
    print('f1 ',f1)

if __name__=="__main__":
    fire.Fire(eval)
