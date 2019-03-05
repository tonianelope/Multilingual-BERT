import torch
from ner_data import VOCAB
# from fastai.basic_train import Learner, LearnerCallback
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel


def ner_loss(output, *ys):
    # bert_layer, x = output

    # hidden_size =  768
    # num_labels = 12 # TODO might be off
    # hidden2label = torch.nn.Linear(hidden_size, num_labels)
    # logits = hidden2label(bert_layer)
    return output

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

        return logits, y_hat

def ner_loss_func(out, *ys, cross_ent=False):
    print(ys)
    logits, y_hat = out
    one_hot_labels, label_mask = ys
    print(one_hot_labels.size())
    print(label_mask.size())
    print(logits.size())
    print(y_hat.size())

    if cross_ent: # use torch cross entropy loss
        logits.view(-1, logits.shape[-1])
        y = ys[0].view(-1)
        fc =  torch.nn.CrossEntropyLoss(ignore_index=0)
        return fc(logits, *ys)

    else:
        p = torch.nn.functional.softmax(logits, -1)
        losses = -torch.log(torch.sum(one_hot_labels * p, -1))
        # losses = torch.masked_select(losses, predict_mask) # TODO compare with predict mask
        return torch.sum(losses)

# class NERLearner(Learner):

#     def __init__(self, data:DataBunch, model:nn.Module,
#                  split_func:OptSplotFunc=None, clip:float=None, alpha:float=2.0, beta:float=1.0,
#                  metrics=None, **learn_kwargs):
#         super().__init__(data, model, **learn_kwargs)

#         # TODO create callback
#         self.callbacks.append(NERTrainer(self, alpha=alpha, beta=beta))
#         if clip: self.callback_fns.append(partial(GradientClipping, clip=clip))
#         if split_func: self.split(split_func)
#         is_class = (hasattr(self.data.train_ds, 'y') and (isinstance(self.data.train_ds.y, CategoryList) or
#                                                           isinstance(self.data.train_ds.y, LMLabelList)))
#         self.metrics = ifnone(metrics, ([accuracy] if is_class else []))

#     # need load encoder save encoder files??

# @dataclass
# class NERCallBack(LearnerCallback):
#     learn: Learner
#     # ?? other att?

#     def on_batch_begin(**kwargs):
#         # TODO prob not requiered
#         #  batch = tuple(t.to(device) for t in batch)
#         # input_ids, input_mask, segment_ids, predict_mask, one_hot_labels = batch
#         pass

#     def on_loss_begin(**kwargs):
#         pass

#     def on_backward_begin(**kwargs):
#         # if config['train']['gradient_accumulation_steps'] > 1:
#         #     loss = loss / config['train']['gradient_accumulation_steps']
#         pass

#     def on_backward_end(**kwargs):
#         if (step + 1) % config['train']['gradient_accumulation_steps'] == 0:
#             # modify learning rate with special warm up BERT uses
#             lr_this_step = config['train']['learning_rate'] * warmup_linear(global_step/num_train_steps, config['train']['warmup_proportion'])
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_this_step
