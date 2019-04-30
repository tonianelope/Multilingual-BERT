import codecs
import logging
from pathlib import Path

import numpy as np

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import Dataset

PAD = '[PAD]'
VOCAB = (PAD, 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG')
label2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2label = {idx: tag for idx, tag in enumerate(VOCAB)}
b2i = {'B-PER':'I-PER', 'B-LOC':'I-LOC','B-ORG':'I-ORG', 'B-MISC':'I-MISC'}

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

class NerDataset(Dataset):
    """
    creates a conll Dataset
    filepath:     path to conll file
    tokenizer:    default is BertTokenizer from pytorch pretrained bert
    max_seq_len:  max length for examples, shorter ones are padded longer discarded
    ds_size:      for debug peruses: truncates the dataset to ds_size examples
    mask:         experiment with the masking type mask[0] is the input mask, mask[1] the label mask
    """
    def __init__(self, filepath, bert_model, max_seq_len=512, ds_size=None, train=False, mask=('s','s')):
        self.xmask, self.ymask = mask
        self.train= train
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)

        #data = read_conll_data(filepath)
        data = open(filepath, 'r').read().strip().split("\n\n")
        if ds_size: data = data[:ds_size]
        size = len(data)
        skipped=0
        sents, labels = [],[]

        for entry in data:
            words = [line.split()[0] for line in entry.splitlines()] #words.split()
            tags = ([line.split()[-1] for line in entry.splitlines()]) #tags.split()

            if words[0]=='-DOCSTART-': continue
            # account for [cls] [sep] token
            #if (len(tokens)+2) > max_seq_len:
            #    skipped +=1
            #    continue

            sents.append(["[CLS]"]+words+["[SEP]"])
            labels.append([PAD]+tags+[PAD])

        org_size = len(sents)
        self.labels, self.sents = labels, sents
        print()
        print(filepath)
        print(f'lines {size} sents {org_size}	style: x={self.xmask} y={self.ymask}')
        # print(f'Truncated examples: {(skipped/org_size)*100:.2}% => {skipped}/{org_size} ')

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        text, labels = self.sents[index], self.labels[index]

        x, y = [], []
        # is heads counts the words (disregarding worpiece sub tokens)
        is_heads, is_labels = [], []

        for w, t in zip(text, labels):
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + [PAD] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [label2idx[each] for each in t]  # (T,)
            is_label = [1] if yy[0]>0 else [0]
            is_label += [0] * (len(tokens)-1)

            x.extend(xx)
            y.extend(yy)
            is_heads.extend(is_head)
            is_labels.extend(is_label)

        one_hot_labels = np.eye(len(label2idx), dtype=np.float32)[y]

        seqlen = len(y)
        segment_ids = [0] * seqlen
        seq_mask = [1] * seqlen
        masks = {'s':seq_mask, 'h':is_heads, 'l': is_labels}
        x_mask = masks[self.xmask]
        y_mask = masks[self.ymask]
        assert_str = f"len(x)={len(x)}, len(y)={len(y)}, len(x_mask)={len(x_mask)}, len(y_mask)={len(y_mask)},"
        assert len(x)==len(y)==len(x_mask)==len(y_mask), assert_str

        xb = (x, segment_ids, x_mask)
        yb = (one_hot_labels, y, y_mask)

        return xb, yb

def pad(batch, bertmax=512, train=False):
    ''' Function to pad samples in batch to the same length '''
    seqlens = [len(x[0]) for x,_ in batch]
    maxlen = np.array(seqlens).max()

    pad_fun = lambda sample: (sample+[0]*(maxlen-len(sample)))
    t = torch.tensor

    input_ids, segment_ids, input_mask, texts =  [],[],[], []
    label_ids, label_mask, one_hot_labels, labels = [],[],[],[]

    for x, y in batch:
        input_ids.append( pad_fun(x[0]) )
        segment_ids.append( pad_fun(x[1]))
        input_mask.append( pad_fun(x[2]))

        label_id = pad_fun(y[1])
        label_ids.append(label_id)
        label_mask.append( pad_fun(y[2]))
        one_hot_labels.append(np.eye(len(label2idx), dtype=np.float32)[label_id])
    x = ( t(input_ids), t(segment_ids), t(input_mask))
    # add labels to x if they are used in the sample.
    if len(batch[0][0])>3: x = ( t(input_ids), t(segment_ids), t(input_mask), t(label_ids))
    y =  ( t(one_hot_labels), t(label_ids), t(label_mask).byte()) 

    return x,y

# from https://github.com/sberbank-ai/ner-bert/blob/master/examples/conll-2003.ipynb
def read_conll_data(input_file:str):
    """Read CONLL-2003 format data."""
    with codecs.open(input_file, "r", encoding="utf-8") as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue

            if len(contends) == 0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([l, w])
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label)
        return lines

def conll_to_docs(input_file:str, output_file:str):
    '''
    Converts a conll-style file to a doc file used to generate the pre-training data
    input_file:    Input file - must follow the conll data format
    output_file:   the resulting file, collecting the sentences/documents
    '''
    with codecs.open(output_file, 'w', encoding="utf-8") as outfile:
        data = open(input_file, 'r').read().strip().split("\n\n")
        for entry in data:
            words = [line.split()[0] for line in entry.splitlines()]
            if words[0]=='-DOCSTART-':
               outfile.write("\n")
               continue
            else:
                w = ' '.join([word for word in words if len(word) > 0])
                outfile.write(w+"\n")

def conll_to_csv(file:str):
    """Write CONLL-2003 to csv"""
    csv_dir = Path('./csv')
    csv_dir.mkdir(parents=True, exist_ok=True)

    filepath = Path(file)
    if(filepath.is_file()):
        data = read_conll_data(filepath)
        df = pd.DataFrame(data, columns=['labels', 'text'])

        csv_path = csv_dir / (filepath.name + '.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f'Wrote {csv_path}')
        return csv_path
    else:
        raise ValueError(f'{file} does not exist, or is not a file')
