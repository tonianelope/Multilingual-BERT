import logging

import numpy as np

import torch
from fastai.basic_train import Learner, LearnerCallback
from fastai.callback import Callback
from fastai.core import is_listy
from fastai.metrics import fbeta
from fastai.torch_core import add_metrics, num_distrib
from ner_data import VOCAB, idx2label
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.optimization import warmup_linear

EPOCH =0

def write_eval(msg, epoch=EPOCH):
    global EPOCH
    EPOCH=epoch
    with open(f'logs/eval_{epoch}.log', 'a') as f:
        f.write(msg+'\n')

def write_eval_lables(pred, true):
    for p, t in zip(pred, true):
        t = idx2label[t.item()]
        p = idx2label[p.item()]
        write_eval(f"{t} {p} {t==p}")
    write_eval("\n")

def write_log(msg):
    with open('logs/out.log', 'a') as f:
        f.write(msg)
        f.write('\n')

def ner_loss_func(out, *ys, cross_ent=False):
    if out.shape<=torch.Size([1]):
        loss = out
    else:
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
        _, labels, attention_mask = ys
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = out.view(-1, len(VOCAB))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            try:
                loss = loss_fct(active_logits, active_labels)
            except Exception as e:
                loss = loss_fct(out.view(-1, len(VOCAB)), labels.view(-1))
        else:
            loss = loss_fct(out.view(-1, len(VOCAB)), labels.view(-1))
    return loss

class OneHotCallBack(Callback):

    def __init__(self, func):
        # If it's a partial, use func.func
        name = getattr(func,'func', func).__name__
        self.func, self.name = func, name
        self.epoch = 1
        self.world = num_distrib()
        self.i = 0

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        write_log(f"E {kwargs['epoch']}")
        self.val, self.count = 0.,0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."

        #print(f"step: loss: {kwargs['last_loss'].item()}")
        logging.info(f'masked target: {target_masked}')
        logging.info(f'masked output: {out_masked}')

        if not is_listy(target_masked): target_masked=[target_masked]
        self.count += target_masked[0].size(0)
        val = self.func(out_masked, *target_masked)
        write_eval(f'F1={val}', self.epoch)
        self.i +=1

        if self.world:
            val = val.clone()
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            val /= self.world
        self.val += target_masked[0].size(0) * val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        self.epoch +=1
        return add_metrics(last_metrics, self.val/self.count)

def conll_f1(pred, *true, eps:float = 1e-9):
    pred = pred.argmax(-1)
    _, label_ids, label_mask = true
    mask = label_mask.view(-1)
    pred = pred.view(-1)
    labels = label_ids.view(-1)
    y_pred = torch.masked_select(pred, mask)
    y_true = torch.masked_select(labels, mask)
    write_eval_lables(y_pred, y_true)
    logging.info('EVAL')
    logging.info(y_pred)
    logging.info(y_true)

    all_pos = len(y_pred[y_pred>1])
    actual_pos = len(y_true[y_true>1])
    correct_pos =(np.logical_and(y_true==y_pred, y_true>1)).sum().item()
    logging.info(f'{all_pos} - {actual_pos} -> {correct_pos}')
    prec = correct_pos / (all_pos + eps)
    rec = correct_pos / (actual_pos + eps)
    f1 = (2*prec*rec)/(prec+rec+eps)
    logging.info(f'f1: {f1}')
    write_log(f'f1: {f1}')

    return torch.Tensor([f1])

def create_fp16_cb(learn, **kwargs):
    return FP16_Callback(learn, **kwargs)

class FP16_Callback(LearnerCallback):

    def __init__(self,
                 learn: Learner,
                 train_opt_steps: int,
                 gradient_accumulation_steps: int = 1,
                 warmup_proportion: float = 0.1,
                 fp16: bool = True,
                 global_step: int = 0):
        super().__init__(learn)
        self.train_opt_steps = train_opt_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_proportion = warmup_proportion
        self.fp16 = fp16
        self.global_step = global_step

    def on_batch_begin(self, last_input, last_target, train, **kwards):
#        if not train:
         return {'last_input': last_input[:2], 'last_target': last_target}

    def on_backward_begin(self, last_loss, **kwargs):
        '''
        returns loss, skip_backward
        '''
        loss = last_loss
        if self.gradient_accumulation_steps > 1:
            loss /= gradient_accumulation_steps

        if self.fp16:
            learn.opt.backwards(loss)
            # modify learning rate with special BERT warm up

            lr_this_step = learn.opt.get_lr() * warmup_linear(
               self.global_step/self.train_opt_steps, warmup_proportion)
            for param_group in learn.opt.param_groups:
                param_group['lr'] = lr_this_step
            global_step += 1
        return {'last_loss': loss, 'skip_bwd': self.fp16}
