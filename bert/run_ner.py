import logging
from functools import partial
from pathlib import Path

import fire
import torch
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.metrics import fbeta
from learner import BertForNER, OneHotCallBack, ner_loss_func
from ner_data import NerDataset
from optimizer import BertAdam
from pytorch_pretrained_bert import BertModel
from torch.utils.data import DataLoader


def conll_f1(oh_pred, oh_true):
    # mask of all 0, 1 elements (e.g padding and 0 label)
    mask = torch.ByteTensor([[1, 1]+[0]*(oh_pred.size(-1)-2)]*oh_pred.size(-2))
    logging.info(f'mask: {mask.size()}')
    oh_pred.masked_fill_(mask, 0)
    oh_true.masked_fill_(mask, 0)
    logging.info(f'oh pred: {oh_pred}')
    logging.info(f'oh true: {oh_true}')
    return fbeta(oh_pred, oh_true, beta=1, sigmoid=False)

def run_ner(batch_size:int=1,
            lr:float=0.0001,
            epochs:int=1,
            trainset:str='data/conll-2003/eng/train.txt',
            devset:str='data/conll-2003/eng/dev.txt',
            testset:str='data/conll-2003/eng/test.txt',
            bert_model:str='bert-base-uncased',
            ds_size:int=None,
            data_bunch_path:str='data/conll-2003/db'):

    train_dl = DataLoader(
        dataset=NerDataset(trainset, ds_size=ds_size),
        batch_size=batch_size,
        shuffle=True
    )

    dev_dl = DataLoader(
        dataset=NerDataset(devset, ds_size=ds_size),
        batch_size=batch_size,
        shuffle=False
    )

    test_dl = DataLoader(
        dataset=NerDataset(testset, ds_size=ds_size),
        batch_size=batch_size,
        shuffle=False
    )

    data = DataBunch(
        train_dl= train_dl,
        valid_dl= dev_dl,
        test_dl = test_dl,
        path = Path(data_bunch_path)
    )

    model = BertForNER.from_pretrained(bert_model)

    f1 = partial(fbeta, beta=1, sigmoid=False)

    learn = Learner(data, model, BertAdam,
                    loss_func=ner_loss_func,
                    metrics=[OneHotCallBack(f1), OneHotCallBack(conll_f1)]
    )

    # learn.lr_find()
    # learn.recorder.plot(skip_end=15)

    learn.fit(epochs, lr)

if __name__ == '__main__':
    fire.Fire(run_ner)
