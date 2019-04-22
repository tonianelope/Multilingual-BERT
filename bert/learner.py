import logging

import numpy as np

import torch
from fastai.basic_train import Learner, LearnerCallback
from fastai.callback import Callback
from fastai.core import is_listy
from fastai.metrics import fbeta
from fastai.torch_core import add_metrics, num_distrib, to_device
from ner_data import VOCAB, idx2label
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.optimization import warmup_linear

EPOCH =0
WEIGHTS = torch.tensor([0.2, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
i = 0

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

def ner_loss_func(out, *ys, zero=False): 
    #print(out.size(),out)
    ys = to_device(ys, torch.cuda.current_device())
    if out.size()<=torch.Size([2]):
     #   print('train')
        loss = out.mean()
    else:
      #  print('val')
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        if zero:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0 , reduction='none')
        #loss_fct = torch.nn.LogSoftmax(-1)
        one, labels, attention_mask = ys
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = out.view(-1, len(VOCAB))[active_loss]
           # active_labels = one.view(-1, len(VOCAB))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            #print('o',out)
            #print('a',active_logits)
            #print('l',active_labels)
            #
            #print('losses',loss)
            logging.info(active_labels)
            logging.info(loss)
            l = loss.sum(-1)
            logging.info(l)
            loss = loss.mean(-1)
            logging.info(loss)
        else:
            loss = loss_fct(out.view(-1, len(VOCAB)), labels.view(-1))
       # loss = loss_fct(out.view(-1, len(VOCAB)))
       # loss = -torch.sum(loss*labels.view(-1))
       # loss = torch.masked_select(loss, attention_mask)
       # loss = torch.mean(loss)
    return loss

def tf_loss_func(out, *ys):
    ys = to_device(ys, torch.cuda.current_device())
    one_hot_labels, labels, attention_mask = ys    
    p = torch.nn.functional.softmax(out, -1)
    ex = torch.exp(out)
    #print('out',out[0][0])
    #print('l',labels[0])
    #out = torch.nn.functional.softmax(out, -1)
    #print(out[1][0])
    #p = torch.nn.functional.log_softmax(out, -1)
    losses = -torch.log(torch.sum(one_hot_labels * p, -1))
    # part = torch.log(torch.sum(ex, 1))
    
    #losses = -(torch.sum(one_hot_labels * p, -1))
    losses = torch.masked_select(losses, attention_mask)
    #return torch.sum(losses)
    print('losses',losses)
    return torch.mean(losses)

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
        #write_log(f"E {kwargs['epoch']}")
        self.val, self.count = 0.,0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."

        #logging.info(f"step: loss: {kwargs['last_loss'].item()}")
        logging.info(f'masked target: {target_masked}')
        logging.info(f'masked output: {out_masked}')

        if not is_listy(target_masked): target_masked=[target_masked]
        self.count += target_masked[0].size(0)
        val = self.func(out_masked, *target_masked)
        #write_eval(f'F1={val}', self.epoch)
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
            loss /= self.gradient_accumulation_steps

        if self.fp16:
            learn.opt.backwards(loss)
            # modify learning rate with special BERT warm up

            lr_this_step = learn.opt.get_lr() * warmup_linear(
               self.global_step/self.train_opt_steps,self.warmup_proportion)
            for param_group in learn.opt.param_groups:
                param_group['lr'] = lr_this_step
            global_step += 1
        return {'last_loss': loss, 'skip_bwd': self.fp16}

def conll_f1(pred, *true, eps:float = 1e-9):
    true = to_device(true, torch.cuda.current_device())
    pred = pred.argmax(-1)
    _, label_ids, label_mask = true
    mask = label_mask.view(-1)==1 
    y_pred = pred.view(-1)[mask]
    y_true = label_ids.view(-1)[mask]
    #y_pred = torch.masked_select(pred, mask)
    #y_true = torch.masked_select(labels, mask)
    #y_pred, y_true = pred, labels
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
    logging.info(f'f1: {f1}   prec: {prec}, rec: {rec}')
    #write_log(f'f1: {f1}')

    return torch.Tensor([f1])

class Conll_F1(Callback):

    def __init__(self):
        super().__init__()
        self.__name__='Total F1'
        self.name = 'Total F1'

    def on_epoch_begin(self, **kwargs):
        self.correct, self.predict, self.true, self.predict2 = 0,0,0,0
        global i 
        i = 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        pred = last_output.argmax(-1)
        true = to_device(last_target, torch.cuda.current_device())
        _, label_ids, label_mask = true
        y_pred = pred.view(-1) 
        y_true = label_ids.view(-1)
        self.predict2 += len(y_pred[y_pred>1])
        preds = y_pred[y_true!=0]
        logging.info(y_true)
        logging.info(y_pred)
        logging.info(preds)
        self.predict += len(preds[preds>1])
        self.true += len(y_true[y_true>1])
        self.correct +=(np.logical_and(y_true==y_pred, y_true>1)).sum().item()

    def on_epoch_end(self, last_metrics, **kwargs):
        eps = 1e-9
        prec = self.correct / (self.predict + eps)
        rec = self.correct / (self.true + eps)
        logging.info(f"====epoch {kwargs['epoch']}====")
        logging.info(f'num pred2: {self.predict2}')
        logging.info(f'num pred: {self.predict}')
        logging.info(f'num corr: {self.correct}')
        logging.info(f'num true: {self.true}')
        logging.info(f'prec: {prec}')
        logging.info(f'rec: {rec}')
        f1 =(2*prec*rec)/(prec+rec+eps)
        logging.info(f'f1: {f1}')
        return add_metrics(last_metrics,f1)
