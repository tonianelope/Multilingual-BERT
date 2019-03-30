D_TRAIN=https://raw.githubusercontent.com/Nidhi-K/Natural-Language-Processing-Projects/master/Sequential%20CRF%20for%20NER/data/deu.train
D_VALID=https://raw.githubusercontent.com/Nidhi-K/Natural-Language-Processing-Projects/master/Sequential%20CRF%20for%20NER/data/deu.testa
D_TEST=https://raw.githubusercontent.com/Nidhi-K/Natural-Language-Processing-Projects/master/Sequential%20CRF%20for%20NER/data/deu.testb
TRAIN=https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/NERdata/train.txt
VALID=https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/NERdata/dev.txt
TEST=https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/NERdata/test.txt

EXEC=python run_ner.py


help:
	$(EXEC) --help

basic-bert:
	$(EXEC) --batch-size=$(BS) --epochs=4 --lr=5e-5 $(W)
	$(EXEC) --batch-size=$(BS) --epochs=4 --lr=3e-5 $(W)
	$(EXEC) --batch-size=$(BS) --epochs=4 --lr=2e-5 $(W)

test1:
	$(EXEC) --ds-size=1 $(W)

apex-test:
	$(EXEC) --ds-size=1 --fp16 $(W)

run:
	$(EXEC) --batch-size=32 --epochs=4 --lr=5e-5 $(W)

run-c:
	$(EXEC) --batch-size=8 --epochs=3 --lr=5e-5 $(W)

run-b:
	$(EXEC) --batch-size=16 --epochs=3 --lr=5e-5 $(W)

apex-run:
	$(EXEC) --batch_size=16 --epochs=1 --fp16 $(W)

datasets:
	make dataset-eng
	make dataset-deu

dataset-eng:
	mkdir -p data/conll-2003/eng
	wget --progress=bar $(TRAIN) && mv train.txt data/conll-2003/eng
	wget --progress=bar $(VALID) && mv dev.txt data/conll-2003/eng
	wget --progress=bar $(TEST) && mv test.txt data/conll-2003/eng

dataset-deu:
	mkdir -p data/conll-2003/deu
	wget --progress=bar $(D_TRAIN) && mv deu.train train.txt && mv train.txt data/conll-2003/deu
	wget --progress=bar $(D_VALID) && mv deu.testa dev.txt && mv dev.txt data/conll-2003/deu
	wget --progress=bar $(D_TEST) && mv deu.testb test.txt && mv test.txt data/conll-2003/deu

model:
	wget https://s3.amazonaws.com/models.huggingface.co/bert/$(BERT).tar.gz

mv-logs:
	mkdir -p $(DIR)
	mv logs/* $(DIR)/

clean-logs:
	rm logs/*

clean-models:
	rm *.tar.gz

rm-dataset:
	rm -rf data

fastai:
	pip install git+https://github.com/fastai/fastai.git
