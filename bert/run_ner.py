from functools import partial
from pathlib import Path

import numpy as np

import fire
import torch
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.metrics import fbeta
from learner import BertForNER, OneHotCallBack, ner_loss_func
from ner_data import NerDataset
from optimizer import BertAdam
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizerimport, logging
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

def run_ner(bert_model:str='bert-base-uncased',
            output_dir:str,
            batch_size:int=1,
            lr:float=0.0001,
            epochs:int=1,
            trainset:str='data/conll-2003/eng/train.txt',
            devset:str='data/conll-2003/eng/dev.txt',
            testset:str='data/conll-2003/eng/test.txt',
            max_seq_length:int=512,
            do_lower_case:bool=False,
            warmup_proportion:float=0.1,
            gradient_accumulation_steps:int=1,
            rand_seed:int=42,
            fp16:bool=False,
            loss_scale:float=None,
            ds_size:int=None,
            data_bunch_path:str='data/conll-2003/db'):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True) # exist_ok=True 

    if gradient_accumulation_steps < 1:
        raise ValueError(f"""Invalid gradient_accumulation_steps parameter:
                         {gradient_accumulation_steps}, should be >= 1""")

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    batch_size //= gradient_accumulation_steps

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

    f1 = partial(fbeta, beta=1, sigmoid=False)

    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                              "to use distributed and fp16 training.")

        optim, dynamic=(FuseAdam, True) if not loss_scale else (FP16_Optimizer,False)
        # TODO call optim backwards(loss) and add special bert warm up on loss calc

        learn = Learner(data, model, optim,
                        loss_func=ner_loss_func,
                        metrics=[OneHotCallBack(f1), OneHotCallBack(conll_f1)]
        ).to_fp16(loss_scale=loss_scale, dynamic=dynamic)
    else:
        learn = Learner(data, model, BertAdam,
                        loss_func=ner_loss_func,
                        metrics=[OneHotCallBack(f1), OneHotCallBack(conll_f1)]
        )

    # learn.lr_find()
    # learn.recorder.plot(skip_end=15)

    learn.fit(epochs, lr)

    m_path = learn.save("ner_trained_model", return_path=True)
    logging.info(f'Saved model to {m_path}')

if __name__ == '__main__':
    fire.Fire(run_ner)
