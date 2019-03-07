from functools import partial
from pathlib import Path

import fire
from fastai.basic_train import Learner
from fastai.metrics import fbeta
from learner import BertForNER, ner_loss_func
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

    model = BertForNER.from_pretrained('bert-base-uncased.tar.gz')

    f1 = partial(fbeta, beta=1, sigmoid=False)

    learn = Learner(data, model, BertAdam, loss_func=ner_loss_func, metrics=[f1])
    # learn.lr_find()
    # learn.recorder.plot(skip_end=15)

    learn.fit(1, 1e-05)

if __name__ == '__main__':
    fire.Fire(run_ner)

    # change logits to one_hots y
    #   y_hat = y_pred.argmax(-1)
    # y_pred = torch.tensor(np.eye(10, dtype=np.float32)[y_hat])
