from pathlib import Path

import fire
from fastai.basic_train import Learner
from ner_data import DEV, TEST, TRAIN, get_data_bunch
from optimizer import BertAdam
from pytorch_pretrained_bert import BertModel

ENG = {
    TRAIN: 'train.txt',
    DEV: 'dev.txt',
    TEST: 'test.txt'
}

def run_ner():
    DATA_BUNCH_PATH = Path('./data/conll-2003/data_bunch')
    DATA_BUNCH_PATH.mkdir(parents=True, exist_ok=True)

    # if [p for f in DATA_BUNCH_PATH.rglob('*') if p.is_file()]:
    data = get_data_bunch(DATA_BUNCH_PATH, ENG)
    # data = load_data(DATA_BUNCH_PATH)

    model = BertModel.from_pretrained('bert-base-uncased')
    learn = Learner(data, model, BertAdam)
    learn.lr_find()
    learn.recorder.plot(skip_end=15)

    learn.fit(1, 0.003)

if __name__ == '__main__':
    fire.Fire(run_ner)
