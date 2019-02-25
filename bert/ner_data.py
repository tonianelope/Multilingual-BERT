from dataclasses import dataclass

from tqdm import tqdm

import torch
from fastai.text import BaseTokenizer, TextDataBunch, Tokenizer
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

# TODO create DataBunch ...

@dataclass
class InputFeatures(object):
    input_ids :list
    input_mask :list
    segment_ids :list
    label_ids :list
    label_mask :list

    '''
`input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
    '''

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

def dataloader(data:list, tokenizer, batch_size):
    features = convert_data(data, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.long)
    
    # all_predict_mask = torch.ByteTensor([f.predict_mask for f in features])
    # all_one_hot_labels = torch.tensor([f.one_hot_labels for f in features], dtype=torch.float)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_mask) # all_predict_mask, all_one_hot_labels)
    sampler = RandomSampler(data)
    return DataLoader(data, sampler=sampler, batch_size=batch_size)
