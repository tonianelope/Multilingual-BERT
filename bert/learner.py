import logging

import numpy as np

import torch
from fastai.basic_train import Learner, LearnerCallback
from fastai.callback import Callback
from fastai.core import is_listy
from fastai.metrics import fbeta
from fastai.torch_core import add_metrics, num_distrib
from ner_data import TOKENIZER, VOCAB, idx2label
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.optimization import warmup_linear

EPOCH = 1

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

def write_eval_text(last_input, pred, true, epoch):
    last_input = last_input[0]
    for text, label, pred in zip(last_input, true, pred):
        text = TOKENIZER.convert_ids_to_tokens(text.tolist())
        org_text = []
        tok = ''
        for i in range(1, len(text)):
            if not text[i].startswith('##'):
                org_text.append(tok.replace('##', ''))
                tok = ''
            tok += text[i]

        for w, t, p in zip(org_text[1:-1], label, pred):
            t = idx2label[t.item()]
            p = idx2label[p.item()]
            write_eval(f"{w} {t} {p} {t==p}" ,epoch)
        write_eval("\n", epoch)

def write_log(msg):
    with open('logs/out.log', 'a') as f:
        f.write(msg+'\n')


class BertForNER(torch.nn.Module):

    def __init__(self, lang):
        super().__init__()

        bert_model = 'bert-base-cased' if lang=='eng' else 'bert-base-multilingual-cased'
        self.bert = BertModel.from_pretrained(bert_model)

        self.num_labels = len(VOCAB)
        self.fc = torch.nn.Linear(768, self.num_labels)

        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, input_ids, segment_ids, input_mask ):
        self.bert.train()
        enc_layer, _ = self.bert(input_ids, segment_ids, input_mask)
        bert_layer = enc_layer[-1]

        # if one_hot_labels is not None:
        #     bert_layer = self.dropout(bert_layer) # TODO comapre to without dropout
        logits = self.fc(bert_layer)
        y_hat = logits.argmax(-1)
        #y_hat = torch.tensor(np.eye(self.num_labels, dtype=np.float32)[y_hat])
        return logits #, y_hat

def ner_loss_func(out, *ys, cross_ent=False):
    
    write_log("===========\n\tLOSS")
    #_ = ner_ys_masked(out, ys, log=True)

    logits = out
    one_hot_labels, label_ids, label_mask = ys

    if cross_ent: # use torch cross entropy loss
        logits = logits.view(-1, logits.shape[-1])
        y = label_ids.view(-1)

        fc =  torch.nn.CrossEntropyLoss(ignore_index=0)
        # need mask???
        loss =  fc(logits, y)
        #print(f"loss: {loss}")
        return loss


    else:
        p = torch.nn.functional.softmax(logits, -1)
        losses = -torch.log(torch.sum(one_hot_labels * p, -1))
        losses = torch.masked_select(losses, label_mask) # TODO compare with predict mask
        return torch.sum(losses)

def ner_ys_masked(output, target, log=True):
    _, label_ids, label_mask  = target

    out = output.argmax(-1)
    out_masked, target_masked = [], []
    for i in range(len(out)):
        o = torch.masked_select(out[i], label_mask[i])
        t = torch.masked_select(label_ids[i], label_mask[i])
        if log:
          write_log(f'T: {t}')
          write_log(f'P: {o}')
        out_masked.extend(o.tolist())
        target_masked.extend(t.tolist())
    return torch.tensor(out_masked), torch.tensor(target_masked)

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
        write_log(f"""====================\n\t E {kwargs['epoch']}\n====================""")
        self.val, self.count = 0.,0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        out_masked, target_masked = ner_ys_masked(last_output, last_target )
        write_eval_text(kwargs['last_input'], out_masked, target_masked, self.epoch)

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
    
    pred, true = ner_ys_masked(pred, true)
    #print('EVAL')
    y_pred, y_true = pred.view(-1), true.view(-1)
    write_eval_lables(y_pred, y_true)
    #print(y_pred)
    #print(y_true)
    all_pos = len(y_pred[y_pred>1])
    actual_pos = len(y_true[y_true>1])
    correct_pos =(np.logical_and(y_true==y_pred, y_true>1)).sum().item()
    logging.info(f'{all_pos} - {actual_pos} -> {correct_pos}')
    prec = correct_pos / (all_pos + eps)
    rec = correct_pos / (actual_pos + eps)
    f1 = (prec*rec)/(prec+rec+eps)
    logging.info(f'f1: {f1}')
    write_log(f'===============\nscores: {f1}')
    print('f1 ',f1) 
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
