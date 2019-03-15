import logging
import random
from functools import partial
from pathlib import Path

import numpy as np

import fire
import torch
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.metrics import fbeta
from fastai.train import to_fp16
from learner import (BertForNER, OneHotCallBack, conll_f1, create_fp16_cb,
                     ner_loss_func)
from ner_data import NerDataset
from optimizer import BertAdam
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader

logging.basicConfig(filename='run_ner.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M%S',
                    level=logging.INFO
)

def run_ner(bert_model:str='bert-base-uncased',
            batch_size:int=1,
            lr:float=0.0001,
            epochs:int=1,
            trainset:str='data/conll-2003/eng/train.txt',
            devset:str='data/conll-2003/eng/dev.txt',
            testset:str='data/conll-2003/eng/test.txt',
            max_seq_length:int=512,
            do_lower_case:bool=False,
            warmup_proportion:float=0.1,
            grad_acc_steps:int=1,
            rand_seed:int=42,
            fp16:bool=False,
            loss_scale:float=None,
            ds_size:int=None,
            data_bunch_path:str='data/conll-2003/db'):

    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

    if grad_acc_steps < 1:
        raise ValueError(f"""Invalid grad_acc_steps parameter:
                         {grad_acc_steps}, should be >= 1""")

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    # TODO proper training with grad accum step??
    batch_size //= grad_acc_steps

    train_dl = DataLoader(
        dataset=NerDataset(trainset, tokenizer=tokenizer, ds_size=ds_size),
        batch_size=batch_size,
        shuffle=True
    )

    dev_dl = DataLoader(
        dataset=NerDataset(devset, tokenizer=tokenizer, ds_size=ds_size),
        batch_size=batch_size,
        shuffle=False
    )

    test_dl = DataLoader(
        dataset=NerDataset(testset, tokenizer=tokenizer, ds_size=ds_size),
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
    optim = BertAdam

    train_opt_steps = int(len(train_dl.dataset) / batch_size / grad_acc_steps) * epochs
    f1 = partial(fbeta, beta=1, sigmoid=False)
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
                    loss_func=ner_loss_func,
                    metrics=[OneHotCallBack(conll_f1)],
                    callback_fns=fp16_cb_fns)

    if fp16: learn.to_fp16(loss_scale=loss_scale, dynamic=dynamic)

    # learn.lr_find()
    # learn.recorder.plot(skip_end=15)

    learn.fit(epochs, lr)

    m_path = learn.save("ner_trained_model", return_path=True)
    logging.info(f'Saved model to {m_path}')

if __name__ == '__main__':
    fire.Fire(run_ner)
