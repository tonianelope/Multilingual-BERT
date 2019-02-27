from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import torch
from conll_preprocessing import read_conll_data
from fastai.basic_data import DataBunch
from fastai.text import BaseTokenizer, TextDataBunch, Tokenizer
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset

# TODO create DataBunch ...

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'


class BertNerDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        xb = tuple(tensor[index] for tensor in self.x)
        yb = tuple(tensor[index] for tensor in self.y)
        return (xb, yb)

    def __len__(self):
        return self.x[0].size(0)


@dataclass
class InputFeatures(object):
    input_ids :list    # input sentence as token ids
    input_mask :list   # mask length of input sentence (cause padding)
    segment_ids :list  # ids of sent A & B
    label_ids :list    # same as input but for labels
    label_mask :list


def convert_sentence(text:list, labels:list, tokenizer, max_seq_len:int):
    "Convert a list of tokens and their corresponding labels"
    bert_tokens = [ "[CLS]" ]
    bert_labels = [ "[CLS]" ]

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
    bert_labels.append("[SEP]")
    return bert_tokens, bert_labels

def convert_data(data :list, tokenizer, label2idx=None, max_seq_len=424, pad='[PAD]'):
    if label2idx is None:
        label2idx = {pad: 0, '[CLS]': 1, '[SEP]': 2}

    features = []

    for idx, (labels, text) in tqdm(enumerate(data), total=len(data)):
        bert_tokens, bert_labels = convert_sentence(text, labels, tokenizer, max_seq_len)

        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        labels = bert_labels # TODO why uncommented????
        for l in labels:
            if l not in label2idx:
                # print(f'new label {l}')
                label2idx[l] = len(label2idx)
        label_ids = [label2idx[l] for l in labels]

        input_mask = [1] * len(input_ids) + [0] * (max_seq_len - len(input_ids))
        label_mask = [1] * len(label_ids) + [0] * (max_seq_len - len(label_ids))
        segment_ids = [0] * max_seq_len # all sent A no sent B
        label_ids += [label2idx[pad]] * (max_seq_len - len(label_ids))
        input_ids += [0] * (max_seq_len - len(input_ids))

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
            label_mask =label_mask
        )
        features.append(feature)

        assert len(input_ids) == max_seq_len
        assert len(input_ids) == max_seq_len
        assert len(input_ids) == max_seq_len
        assert len(input_ids) == max_seq_len

    return features

def get_data_bunch(data_bunch_path:Path, files:dict):
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
        data = data[:10] if key!=TRAIN else data[:50]
        features[key] = convert_data(data, tokenizer, max_seq_len=512)

    batch_size = 32
    dls = {}
    for key in files:
        all_input_ids = torch.tensor([f.input_ids for f in features[key]])
        all_input_mask = torch.tensor([f.input_mask for f in features[key]])
        all_segment_ids = torch.tensor([f.segment_ids for f in features[key]])
        all_label_ids = torch.tensor([f.label_ids for f in features[key]])
        all_label_mask = torch.tensor([f.label_mask for f in features[key]])

        print(len(all_label_ids))
        x = [all_input_ids, all_segment_ids, all_input_mask]
        y = [all_label_ids, all_label_mask]

        data = BertNerDataset(x, y)
        sampler = RandomSampler(data)
        batch_size = 32
        dls[key] = DataLoader(data, sampler=sampler, batch_size=batch_size)

    dataBunch = DataBunch(
        train_dl= dls[TRAIN],
        valid_dl= dls[DEV],
        test_dl = dls[TEST],
        path = data_bunch_path
    )
    
    # dataBunch.save() TODO figure out the issue with saving??
    return dataBunch
