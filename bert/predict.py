from pathlib import Path

import fire
import torch
from fastai.basic_train import load_learner
from ner_data import VOCAB, idx2label
from pytorch_pretrained_bert import BertForTokenClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer


def to_feature(sent, bert_model):
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
    words = ['[CLS]']+sent+['[SEP]']

    x, mask = [], []
    for w in words:
        w = w.strip()
        tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
        xx = tokenizer.convert_tokens_to_ids(tokens)
        m = [1] + [0]*(len(tokens)-1)
        x.extend(xx)
        mask.extend(m)
    t = torch.LongTensor
    print('input: ',tokenizer.convert_ids_to_tokens(x))
    return t([x]), t(mask)

def predict(name, lang='eng', path='learn', model_dir='models'):
    path, model_dir = Path(path), Path(model_dir)
    print('Loading model...')
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    state = torch.load(path/model_dir/f'{name}.pth', map_location=device)
    bert_model = 'bert-base-cased' if lang=='eng' else 'bert-base-multilingual-cased'
    print(f'Lang: {lang}\nModel: {bert_model}\nRun: {name}')
    model = BertForTokenClassification.from_pretrained(bert_model, num_labels=len(VOCAB), cache_dir='bertm')
    model.load_state_dict(state['model'], strict=True)
    print('Done')

    try:
        while True:
            # get sentence
            sent = input('Enter sentence: ')
            words = sent.split()
            x, mask = to_feature(words, bert_model)
            with torch.no_grad():
                # predict named entities
                out = model(x)
                pred = out.argmax(-1).view(-1)
                print(pred)
                active_pred = pred[mask==1]
                print('Named Entities')
                active_pred = active_pred.tolist()
                for w,l in zip(words,active_pred[1:-1]):
                    print(f'{w} {idx2label[l]}')

    except Exception as e:
        print('See ya')

if __name__ == '__main__':
    fire.Fire(predict)
