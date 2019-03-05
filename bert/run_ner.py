from pathlib import Path

import fire
from fastai.basic_train import Learner
from learner import BertForNER, ner_loss
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

    data = get_data_bunch(DATA_BUNCH_PATH, ENG, batch_size=1)

    model = BertForNER.from_pretrained('bert-base-uncased', num_labels=12)
    learn = Learner(data, model, BertAdam, loss_func=ner_loss)
    # learn.lr_find()
    # learn.recorder.plot(skip_end=15)

    learn.fit(7, 1e-05)

if __name__ == '__main__':
    fire.Fire(run_ner)
