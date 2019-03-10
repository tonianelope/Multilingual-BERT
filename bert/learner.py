import logging

import numpy as np

import torch
from fastai.callback import Callback, LearnerCallback
from fastai.core import is_listy
from fastai.torch_core import add_metrics, num_distrib
from ner_data import VOCAB
from pytorch_pretrained_bert import warmup_linear
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel


class BertForNER(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForNER, self).__init__(config)
        self.num_labels = len(VOCAB)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(0.2)
        self.hidden2label = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids, input_mask):
        bert_layer, _ = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)

        # if one_hot_labels is not None:
        #     bert_layer = self.dropout(bert_layer) # TODO comapre to without dropout
        logits = self.hidden2label(bert_layer)
        y_hat = logits.argmax(-1)
        #logging.info(f'Y_hat {y_hat.size()}\n{y_hat}\n')
        y_hat = torch.tensor(np.eye(self.num_labels, dtype=np.float32)[y_hat])
        #logging.info(f'One_hat {y_hat.size()}\n{y_hat}\n')
        #logging.info(f'LOGITS {logits.size()}\n{logits}\n')
        return logits #, y_hat

def ner_loss_func(out, *ys, cross_ent=False):

    logits = out
    one_hot_labels, label_ids, label_mask = ys

    if cross_ent: # use torch cross entropy loss
        logits.view(-1, logits.shape[-1])
        y = ys[0].view(-1)
        fc =  torch.nn.CrossEntropyLoss(ignore_index=0)
        # need mask???
        return fc(logits, y)

    else:
        p = torch.nn.functional.softmax(logits, -1)
        losses = -torch.log(torch.sum(one_hot_labels * p, -1))
        losses = torch.masked_select(losses, label_mask) # TODO compare with predict mask
        return torch.sum(losses)

class OneHotCallBack(Callback):

    def __init__(self, func):
        # If it's a partial, use func.func
        name = getattr(func,'func', func).__name__
        self.func, self.name = func, name
        self.world = num_distrib()

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0.,0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        _, label_ids, label_mask = last_target
        out = last_output.argmax(-1)
        logging.info(f'last target {label_ids[0][:20]}')
        logging.info(f'last output {out[0][:20]}')
        out_masked = torch.masked_select(out, label_mask)
        last_output = torch.tensor(np.eye(10, dtype=np.float32)[out_masked])
        target_masked = torch.masked_select(label_ids, label_mask)
        one_hot_labels = torch.tensor(np.eye(10, dtype=np.float32)[target_masked])
        logging.info(f'last target {one_hot_labels.size()}')
        logging.info(f'last output {last_output.size()}')
        logging.info(f'target {target_masked}')
        logging.info(f'output {out_masked}')


        if not is_listy(one_hot_labels): one_hot_labels=[one_hot_labels]
        self.count += one_hot_labels[0].size(0)
        val = self.func(last_output, *one_hot_labels)
        if self.world:
            val = val.clone()
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            val /= self.world
        self.val += one_hot_labels[0].size(0) * val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val/self.count)


def conll_f1(oh_pred, oh_true):
    pass

@dataclass
class FP16Callback(LearnerCallback):

    def __init__(self, learn, fp16, gradient_accumulation_steps, train_opt_steps, warmup_proportion, global_step=0):

    '''
    returns loss, skip_backward
    '''
    def on_backward_begin(self, loss, **kwargs):
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
        return loss, self.fp16
