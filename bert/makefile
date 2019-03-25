D_TRAIN=https://raw.githubusercontent.com/Nidhi-K/Natural-Language-Processing-Projects/master/Sequential%20CRF%20for%20NER/data/deu.train
D_VALID=https://raw.githubusercontent.com/Nidhi-K/Natural-Language-Processing-Projects/master/Sequential%20CRF%20for%20NER/data/deu.testa
D_TEST=https://raw.githubusercontent.com/Nidhi-K/Natural-Language-Processing-Projects/master/Sequential%20CRF%20for%20NER/data/deu.testb
TRAIN=https://raw.githubusercontent.com/Emrys-Hong/fastai_sequence_tagging/master/data/coNLL/train.txt
VALID=https://raw.githubusercontent.com/Emrys-Hong/fastai_sequence_tagging/master/data/coNLL/valid.txt
TEST=https://raw.githubusercontent.com/Emrys-Hong/fastai_sequence_tagging/master/data/coNLL/test.txt

EXEC=python run_ner.py


help:
	$(EXEC) --help

test:
	$(EXEC) --ds-size=1 $(W)

apex-test:
	$(EXEC) --ds-size=1 --fp16 $(W)

run:
	$(EXEC) --batch-size=32 --epochs=4 --lr=5e-5 --cross-ent=True $(W)

run-c:
	$(EXEC) --batch-size=8 --epochs=3 --lr=5e-5 --cross-ent=True $(W)

run-nc:
	$(EXEC) --batch-size=32 --epochs=3 --lr=5e-5 --cross-ent=False $(W)

apex-run:
	$(EXEC) --batch_size=16 --epochs=1 --fp16 $(W)

datasets:
	make dataset-eng
	make dataset-deu

dataset-eng:
	mkdir -p data/conll-2003/eng
	wget --show-progress $(TRAIN) && mv train.txt data/conll-2003/eng
	wget --show-progress $(VALID) && mv valid.txt dev.txt && mv dev.txt data/conll-2003/eng
	wget --show-progress $(TEST) && mv test.txt data/conll-2003/eng

dataset-deu:
	mkdir -p data/conll-2003/deu
	wget --show-progress $(D_TRAIN) && mv deu.train train.txt && mv train.txt data/conll-2003/deu
	wget --show-progress $(D_VALID) && mv deu.testa dev.txt && mv dev.txt data/conll-2003/deu
	wget --show-progress $(D_TEST) && mv deu.testb test.txt && mv test.txt data/conll-2003/deu

model:
	wget https://s3.amazonaws.com/models.huggingface.co/bert/$(BERT).tar.gz

clean-logs:
	rm *.log
	rm out.txt
	rm eval_*

clean-models:
	rm *.tar.gz

rm-dataset:
	rm -rf data

fastai:
	pip install git+https://github.com/fastai/fastai.git
