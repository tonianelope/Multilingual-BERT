import csv
import logging
import random
from functools import partial
from pathlib import Path

import numpy as np

import fire
import torch
from fastai.basic_data import DataBunch, DatasetType
from fastai.basic_train import Learner
from fastai.callback import OptimWrapper
from fastai.metrics import fbeta
from fastai.torch_core import flatten_model, to_device
from fastai.train import to_fp16
from learner import Conll_F1, conll_f1, ner_loss_func
from ner_data import VOCAB, NerDataset, idx2label, pad
from optimizer import BertAdam, initBertAdam
from pytorch_pretrained_bert import BertForTokenClassification
from torch.utils.data import DataLoader

NER = 'conll-2003'

def init_logger(log_dir, name):
    logging.basicConfig(filename=log_dir / (name+'.log'),
                    filemode='w',
                    format='%(asctime)s, %(message)s',
                    datefmt='%H:%M%S',
                    level=logging.INFO
    )

def apply_freez(learn, layers, lay):
    if lay==0: learn.freeze()
    if lay==layers: learn.unfreeze()
    else: learn.freeze_to(lay)
    print('Freezing layers ', lay, ' off ', layers)

def bert_layer_list(model):
    ''' Break a bert base model in to a list of layers'''
    ms = torch.nn.ModuleList()

    flm = flatten_model(model)
    # embedding = [0:5] layer
    ms.append(torch.nn.ModuleList(flm[0:5]))
    # encoder (12 layers) = [5:16] [16:27] ... [126:136]
    bert_layergroup_size = 11#33
    for i in range(5, 137, bert_layergroup_size):
        ms.append(torch.nn.ModuleList(flm[i: i+bert_layergroup_size]))
    # pooling layer = [137:139]
    ms.append(torch.nn.ModuleList(flm[-4:-2]))
    # head = [-2:]
    ms.append(torch.nn.ModuleList(flm[-2:]))
    return ms

def run_ner(lang:str='eng',
            log_dir:str='logs',
            task:str=NER,
            batch_size:int=1,
            lr:float=5e-5,
            epochs:int=1,
            dataset:str='data/conll-2003/',
            loss:str='cross',
            max_seq_len:int=128,
            do_lower_case:bool=False,
            warmup_proportion:float=0.1,
            grad_acc_steps:int=1,
            rand_seed:int=None,
            fp16:bool=False,
            loss_scale:float=None,
            ds_size:int=None,
            data_bunch_path:str='data/conll-2003/db',
            bertAdam:bool=False,
            freez:bool=False,
            one_cycle:bool=False,
            discr:bool=False,
            lrm:int=2.6,
            div:int=None,
            tuned_learner:str=None,
            do_train:str=False,
            do_eval:str=False,
	        save:bool=False,
            name:str='ner',
            mask:tuple=('s','s'),
):
    name = "_".join(map(str,[name,task, lang, mask[0],mask[1], loss, batch_size, lr, max_seq_len,do_train, do_eval]))
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    init_logger(log_dir, name)

    if rand_seed:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rand_seed)

    trainset = dataset + lang + '/train.txt'
    devset = dataset +lang + '/dev.txt'
    testset = dataset + lang + '/test.txt'

    bert_model = 'bert-base-cased' if lang=='eng' else 'bert-base-multilingual-cased'
    print(f'Lang: {lang}\nModel: {bert_model}\nRun: {name}')
    model = BertForTokenClassification.from_pretrained(bert_model, num_labels=len(VOCAB), cache_dir='bertm')

    model = torch.nn.DataParallel(model)
    model_lr_group = bert_layer_list(model)
    layers = len(model_lr_group)
    kwargs = {'max_seq_len':max_seq_len, 'ds_size':ds_size, 'mask':mask}

    train_dl = DataLoader(
        dataset=NerDataset(trainset,bert_model,train=True, **kwargs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(pad, train=True)
    )

    dev_dl = DataLoader(
        dataset=NerDataset(devset, bert_model, **kwargs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad
    )

    test_dl = DataLoader(
        dataset=NerDataset(testset, bert_model, **kwargs),
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

    loss_fun = ner_loss_func if loss=='cross' else partial(ner_loss_func, zero=True)
    metrics = [Conll_F1()]

    learn = Learner(data, model, BertAdam,
                    loss_func=loss_fun,
                    metrics=metrics,
                    true_wd=False,
                    layer_groups=None if not freez else model_lr_group,
                    path='learn',
                    )

    # initialise bert adam optimiser
    train_opt_steps = int(len(train_dl.dataset) / batch_size) * epochs
    optim = BertAdam(model.parameters(),
                     lr=lr,
                     warmup=warmup_proportion,
                     t_total=train_opt_steps)

    if bertAdam: learn.opt = OptimWrapper(optim)
    else: print("No Bert Adam")

    # load fine-tuned learner
    if tuned_learner:
        print('Loading pretrained learner: ', tuned_learner)
        learn.load(tuned_learner)

    # Uncomment to graph learning rate plot
    # learn.lr_find()
    # learn.recorder.plot(skip_end=15)

    # set lr (discriminative learning rates)
    if div: layers=div
    lrs = lr if not discr else learn.lr_range(slice(lr/lrm**(layers), lr))

    results = [['epoch', 'lr', 'f1', 'val_loss', 'train_loss', 'train_losses']]

    if do_train:
        for epoch in range(epochs):
            if freez:
                lay= (layers//(epochs-1)) * epoch * -1
                if lay==0:print('Freeze'); learn.freeze()
                elif lay==layers: print('unfreeze');learn.unfreeze()
                else: print('freeze2');learn.freeze_to(lay)
                print('Freezing layers ', lay, ' off ', layers)

            # Fit Learner - eg train model
            if one_cycle: learn.fit_one_cycle(1, lrs, moms=(0.8, 0.7))
            else: learn.fit(1, lrs)

            results.append([
                epoch, lrs,
                learn.recorder.metrics[0][0],
                learn.recorder.val_losses[0],
                np.array(learn.recorder.losses).mean(),
                learn.recorder.losses,
            ])

            if save:
                m_path = learn.save(f"{lang}_{epoch}_model", return_path=True)
                print(f'Saved model to {m_path}')
    if save: learn.export(f'{lang}.pkl')

    if do_eval:
        res = learn.validate(test_dl, metrics=metrics)
        met_res = [f'{m.__name__}: {r}' for m, r in zip(metrics, res[1:])]
        print(f'Validation on TEST SET:\nloss {res[0]}, {met_res}')
        results.append([
            'val', '-', res[1], res[0], '-','-'
        ])

    with open(log_dir / (name+'.csv'), 'a') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerows(results)

if __name__ == '__main__':
    fire.Fire(run_ner)
