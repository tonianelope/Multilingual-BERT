import codecs
from pathlib import Path

import numpy as np

from conll_preprocessing import read_conll_data
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import Dataset

PAD = '[PAD]'
VOCAB = (PAD, 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG')
label2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2label = {idx: tag for idx, tag in enumerate(VOCAB)}

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

class NerDataset(Dataset):
    def __init__(self, filepath, tokenizer=TOKENIZER, max_seq_len=512):
        data = read_conll_data(filepath)
        self.labels, self.sents = zip(*data)
        self.tokenizer = tokenizer
        print(self.labels[:2])
        print(self.sents[:2])

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        bert_tokens, bert_labels = self.get_bert_tl(index)

        input_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        labels = bert_labels # TODO why uncommented????
        label_ids = [label2idx[l] for l in labels]

        input_mask = [1] * len(input_ids) + [0] * (max_seq_len - len(input_ids))
        label_mask =[0] + [1] * (len(label_ids)-2) + [0] * (max_seq_len - len(label_ids)+1)
        segment_ids = [0] * max_seq_len # all sent A no sent B
        label_ids += [label2idx[PAD]] * (max_seq_len - len(label_ids))
        input_ids += [0] * (max_seq_len - len(input_ids))

        one_hot_labels = np.eye(len(label2idx), dtype=np.float32)[label_ids]

        assert max_seq_len == len(input_ids) == len(segment_ids) == len(one_hot_labels)

        return (input_ids, segment_ids, input_mask), (one_hot_labels, label_mask)

    def get_bert_tl(self, index):
        "Convert a list of tokens and their corresponding labels"
        text, labels = self.sents[index], self.labels[index]
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
                cur_tokens = self.tokenizer.tokenize(orig_token)
            if not cur_tokens: print(orig_token)
            cur_labels = [f'{prefix[0]}{label}']+[f'{prefix[1]}{label}']*(len(cur_tokens)-1)

            # TODO log to long examples
            if self.max_seq_len - 1 < len(bert_tokens) + len(cur_tokens):
                break

            bert_tokens.extend(cur_tokens)
            bert_labels.extend(cur_labels)

        bert_tokens.append("[SEP]")
        bert_labels.append( PAD )
        return bert_tokens, bert_labels

# from https://github.com/sberbank-ai/ner-bert/blob/master/examples/conll-2003.ipynb
def read_conll_data(input_file:str):
    """Reads CONLL-2003 format data."""
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

def conll_to_csv(file):
    """Write CONLL-2003 to csv"""

    csv_dir = Path('./csv')
    csv_dir.mkdir(parents=True, exist_ok=True)

    filepath = Path(file)
    if(filepath.is_file()):
        data = read_conll_data(filepath)
        df = pd.DataFrame(data, columns=['labels', 'text'])

        csv_path = csv_dir / (filepath.name + '.csv')
        df.to_csv(csv_path, index=False)
        print(f'Wrote {csv_path}')
        return csv_path
    else:
        raise ValueError(f'{file} does not exist, or is not a file')
