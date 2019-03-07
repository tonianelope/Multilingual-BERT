from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from conll_preprocessing import read_conll_data
from fastai.basic_data import DataBunch
from fastai.text import BaseTokenizer, TextDataBunch, Tokenizer
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset

PAD = '[PAD]'
VOCAB = (PAD, 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG')
label2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2label = {idx: tag for idx, tag in enumerate(VOCAB)}

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

class BertNerDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        xb = tuple(tensor[0] for tensor in self.x)
        yb = self.y[0]# [tensor[index] for tensor in self.y]
        print('DATA:', xb[0][:10])
        return xb, yb

    def __len__(self):
        return self.x[0].size(0)


@dataclass
class InputFeatures(object):
    input_ids :list      # input sentence as token ids
    input_mask :list     # mask length of input sentence (cause padding)
    segment_ids :list    # ids of sent A & B
    label_ids :list      # same as input but for labels
    label_mask :list
    one_hot_labels :list # one hot encoded labels for loss calculation

def convert_sentence(text:list, labels:list, tokenizer, max_seq_len:int):
    "Convert a list of tokens and their corresponding labels"
    bert_tokens = [ "[CLS]" ]
    bert_labels = [ PAD ]

    orig_tokens = str(text).split()
    labels = labels.split()
    assert len(orig_tokens) == len(labels)
    prev_label = ""

    for i, (orig_token, label) in enumerate(zip(orig_tokens, labels)):
        prefix = ['', '']
        if label != 'O':
            label = label.split("-")[1]
            prefix[0] = 'I-' if label==prev_label else 'B-'
            prefix[1] = 'I-'
        prev_label = label
        cur_tokens = tokenizer.tokenize(orig_token)
        if not cur_tokens: print(orig_token)
        cur_labels = [f'{prefix[0]}{label}']+[f'{prefix[1]}{label}']*(len(cur_tokens)-1)

        # TODO log to long examples
        if max_seq_len - 1 < len(bert_tokens) + len(cur_tokens):
            break

        bert_tokens.extend(cur_tokens)
        bert_labels.extend(cur_labels)

    bert_tokens.append("[SEP]")
    bert_labels.append( PAD )
    return bert_tokens, bert_labels

def convert_data(data :list, tokenizer, max_seq_len=424):

    features = []

    for idx, (labels, text) in tqdm(enumerate(data), total=len(data)):
        bert_tokens, bert_labels, = convert_sentence(text, labels, tokenizer, max_seq_len)

        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        labels = bert_labels # TODO why uncommented????
        label_ids = [label2idx[l] for l in labels]

        input_mask = [1] * len(input_ids) + [0] * (max_seq_len - len(input_ids))
        label_mask =[0] + [1] * (len(label_ids)-2) + [0] * (max_seq_len - len(label_ids)+1)
        segment_ids = [0] * max_seq_len # all sent A no sent B
        label_ids += [label2idx[PAD]] * (max_seq_len - len(label_ids))
        input_ids += [0] * (max_seq_len - len(input_ids))

        # print(len(label2idx))
        one_hot_labels = np.eye(len(label2idx), dtype=np.float32)[label_ids]

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
            label_mask =label_mask,
            one_hot_labels =one_hot_labels
        )
        features.append(feature)

        assert len(input_ids) == max_seq_len
        assert len(input_ids) == max_seq_len
        assert len(input_ids) == max_seq_len
        assert len(input_ids) == max_seq_len
    return features

def get_data_bunch(data_bunch_path:Path, files:dict, batch_size=32):
    # TODO pass path via input
    DATA_PATH = Path('./data/conll-2003/eng')
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    CSV_PATH = Path('./data/conll-2003/csv')
    CSV_PATH.mkdir(parents=True, exist_ok=True)

    lower_case = True

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=lower_case)

    # TODO merge for loops?
    features = {}
    for key in files:
        data = read_conll_data(DATA_PATH/files[key])
        data = data[:1] # if key!=TRAIN else data[:1]
        features[key] = convert_data(data, tokenizer, max_seq_len=512)

    dls = {}
    for key in files:
        all_input_ids = torch.tensor([f.input_ids for f in features[key]])
        all_input_mask = torch.tensor([f.input_mask for f in features[key]])
        all_segment_ids = torch.tensor([f.segment_ids for f in features[key]])
        all_one_hot_labels = torch.tensor([f.one_hot_labels for f in features[key]])

        all_label_ids = torch.tensor([f.label_ids for f in features[key]])
        all_label_mask = torch.tensor([f.label_mask for f in features[key]])

        print(len(all_label_ids))
        x = [all_input_ids, all_segment_ids, all_input_mask]
        y = all_one_hot_labels #, all_label_mask]

        data = BertNerDataset(x, y)
        sampler = None#RandomSampler(data)
        dls[key] = DataLoader(data, sampler=sampler, batch_size=batch_size)

    dataBunch = DataBunch(
        train_dl= dls[TRAIN],
        valid_dl= dls[DEV],
        test_dl = dls[TEST],
        path = data_bunch_path
    )

    # dataBunch.save() TODO figure out the issue with saving??
    return dataBunch
