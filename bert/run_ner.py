import logging
import random
from functools import partial
from pathlib import Path

import numpy as np

import fire
import torch
from fastai.basic_data import DataBunch, DatasetType
from fastai.basic_train import Learner
from fastai.metrics import fbeta
from fastai.torch_core import flatten_model, to_device
from fastai.train import to_fp16
from learner import (Conll_F1, OneHotCallBack, conll_f1, create_fp16_cb,
                     ner_loss_func, write_eval)
from ner_data import VOCAB, NerDataset, idx2label, pad
from optimizer import BertAdam
from pytorch_pretrained_bert import BertForTokenClassification
from torch.utils.data import DataLoader

NER = 'conll-2003'

def init_logger(log_dir, name):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=log_dir / (name+'.log'),
                    filemode='w',
                    format='%(asctime)s, %(message)s',
                    datefmt='%H:%M%S',
                    level=logging.INFO
    )

def bert_layer_list(model):
    ms = torch.nn.ModuleList()

    flm = flatten_model(model)
    # embedding = [0:5] layer
    ms.append(torch.nn.ModuleList(flm[0:5]))
    # encoder (12 layers) = [5:16] [16:27] ... [126:136]
    for i in range(5, 137, 11):
        ms.append(torch.nn.ModuleList(flm[i: i+11]))
    # pooling layer = [137:139]
    ms.append(torch.nn.ModuleList(flm[-4:-2]))
    # head = [-2:]
    ms.append(torch.nn.ModuleList(flm[-2:]))
    return ms

def train(learn, epochs, lr, name, freez, discr, one_cycle, save):
    lrs = lr if not discr else learn.lr_range(slice(start_lr, end_lr))
    for epoch in range(epochs):
        if freez:
            if epoch==0: learn.freeze()
            elif epoch==epochs-1: learn.unfreeze()
            else:
                lay = (15//(epochs-1)) * epoch * -1
                print('freez top ', lay, ' off ', 15)
                learn.freeze_to(lay)
        if one_cycle:
            learn.fit_one_cylce(1, lrs, mom=(0.8, 0.7))
        else:
            learn.fit(1, lrs)
        if save: m_path = learn.save(f"{name}_{epoch}_model", return_path=True)
    if save: print(f'Saved model to {m_path}')

def run_ner(lang:str='eng',
            log_dir:str='logs',
            task:str=NER,
            batch_size:int=1,
            lr:float=5e-5,
            epochs:int=1,
            trainset:str='data/conll-2003/',
            devset:str='data/conll-2003/',
            testset:str='data/conll-2003/',
            max_seq_len:int=128,
            do_lower_case:bool=False,
            warmup_proportion:float=0.1,
            grad_acc_steps:int=1,
            rand_seed:int=None,
            fp16:bool=False,
            loss_scale:float=None,
            ds_size:int=None,
            data_bunch_path:str='data/conll-2003/db',
            freez:bool=False,
            one_cycle:bool=False,
            discr:bool=False,
            tuned_learner:str=None,
            do_train:str=False,
            do_eval:str=False,
	        save:bool=False,
):
    name = "_".join(map(str,[task, lang, batch_size, lr, max_seq_len,do_train, do_eval]))
    init_logger(log_dir, name)

    if rand_seed:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)

    if grad_acc_steps < 1:
        raise ValueError(f"""Invalid grad_acc_steps parameter:
                         {grad_acc_steps}, should be >= 1""")

    # TODO proper training with grad accum step??
    batch_size //= grad_acc_steps

    trainset += lang + '/train.txt'
    devset += lang + '/dev.txt'
    testset += lang + '/test.txt'

    bert_model = 'bert-base-cased' if lang=='eng' else 'bert-base-multilingual-cased'
    print(f'Lang: {lang}\nModel: {bert_model}\nRun: {name}')
    model = BertForTokenClassification.from_pretrained(bert_model, num_labels=len(VOCAB), cache_dir='bertm')

    if batch_size>16: model = torch.nn.DataParallel(model)

    train_dl = DataLoader(
        dataset=NerDataset(trainset,bert_model,max_seq_len=max_seq_len, ds_size=ds_size),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad
    )

    dev_dl = DataLoader(
        dataset=NerDataset(devset, bert_model,max_seq_len=max_seq_len, ds_size=ds_size),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad
    )

    test_dl = DataLoader(
        dataset=NerDataset(testset, bert_model,max_seq_len=max_seq_len, ds_size=ds_size),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad
    )

    data = DataBunch(
        train_dl= train_dl,
        valid_dl= dev_dl,
        test_dl = test_dl,
        collate_fn=pad,
        path = Path(data_bunch_path)
    )

    optim = BertAdam

    train_opt_steps = int(len(train_dl.dataset) / batch_size / grad_acc_steps) * epochs
    f1 = partial(fbeta, beta=1, sigmoid=False)
    loss_fun = ner_loss_func
    metrics = [conll_f1, Conll_F1()]
    fp16_cb_fns = partial(create_fp16_cb,
                          train_opt_steps = train_opt_steps,
                          gradient_accumulation_steps = grad_acc_steps,
                          warmup_proportion = warmup_proportion,
                          fp16 = fp16)

    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                              "to use distributed and fp16 training.")
        optim, dynamic=(FusedAdam, True) if not loss_scale else (FP16_Optimizer,False)

    learn = Learner(data, model, optim,
                    loss_func=loss_fun,
                    metrics=metrics
                    true_wd=False,
                    callback_fns=fp16_cb_fns,
                    layer_groups=None if not freez else bert_layer_list(model),
                    path='learn',
                    )
    # load fine-tuned learner
    if tuned_learner:
        print('Loading pretrained learner: ', tuned_learner)
        learn.load(tuned_learner)

    if fp16: learn.to_fp16(loss_scale=loss_scale, dynamic=dynamic)

    # learn.lr_find()
    # learn.recorder.plot(skip_end=15)

    if do_train:
        train(learn, epochs, lr, name, freez, discr, one_cycle,save)
    if do_eval:
        res = learn.validate(test_dl, metrics=metrics)
        print(f'Validation on test set:\nloss {res[0]}, scores: {res[1:]}')

if __name__ == '__main__':
    fire.Fire(run_ner)
