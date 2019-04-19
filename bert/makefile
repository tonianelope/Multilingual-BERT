GERM_URL=https://raw.githubusercontent.com/Nidhi-K/Natural-Language-Processing-Projects/master/Sequential%20CRF%20for%20NER/data/
GERMI_URL=https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/
ENG_URL=https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/NERdata/
ENGI_URL=https://raw.githubusercontent.com/smb564/NLP/master/src/


D_TRAIN=$(GERM_URL)deu.train
D_VALID=$(GERM_URL)deu.testa
D_TEST=$(GERM_URL)deu.testb
TRAIN=$(ENG_URL)train.txt
VALID=$(ENG_URL)dev.txt
TEST=$(ENG_URL)test.txt

EXEC=python run_ner.py

help:
	$(EXEC) --help

basic-bert:
	$(EXEC) --batch-size=$(BS) --epochs=4 --lr=5e-5 $(W)
	$(EXEC) --batch-size=$(BS) --epochs=4 --lr=3e-5 $(W)
	$(EXEC) --batch-size=$(BS) --epochs=4 --lr=2e-5 $(W)

icetown:
	$(EXEC) --do-train --do-eval --batch-size=32 --epochs=4 --lr=$(LR) --freez --discr --one-cycle --lang=$(L)

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

2bert:
	python bert_train_data.py --train_corpus $(DIR)docs.txt --bert_model $(M) --output_dir $(DIR) --epochs_to_generate $(E) --max_seq_len 256

pretrain_lm:
	python $(FILE) --pregenerated_data $(DIR) --bert_model $(M) --output_dir pretrain --epochs $(E) --train_batch_size 16

datasets:
	make dataset-eng
	make dataset-deu

dataset-eng:
	mkdir -p data/conll-2003/eng
	wget --progress=bar $(TRAIN) && mv train.txt data/conll-2003/eng
	wget --progress=bar $(VALID) && mv dev.txt data/conll-2003/eng
	wget --progress=bar $(TEST) && mv test.txt data/conll-2003/eng

dataset-engI:
	mkdir -p data/conll-2003/ieng
	wget --progress=bar $(ENGI_URL)conll2003.eng.train && mv conll2003.eng.train data/conll-2003/ieng/train.txt
	wget --progress=bar $(ENGI_URL)conll2003.eng.testa && mv conll2003.eng.testa data/conll-2003/ieng/dev.txt
	wget --progress=bar $(ENGI_URL)conll2003.eng.testb && mv conll2003.eng.testb data/conll-2003/ieng/test.txt

dataset-deu:
	mkdir -p data/conll-2003/deu
	wget --progress=bar $(D_TRAIN) && mv deu.train train.txt && mv train.txt data/conll-2003/deu
	wget --progress=bar $(D_VALID) && mv deu.testa dev.txt && mv dev.txt data/conll-2003/deu
	wget --progress=bar $(D_TEST) && mv deu.testb test.txt && mv test.txt data/conll-2003/deu

dataset-deuI:
	mkdir -p data/conll-2003/deu
	wget --progress=bar $(D_TRAIN)deuutf.train && mv deuutf.train data/conll-2003/deu/train.txt
	wget --progress=bar $(D_VALID)deuutf.testa && mv deuutf.testa data/conll-2003/deu/dev.txt
	wget --progress=bar $(D_TEST)deuutf.testb && mv deuutf.testb data/conll-2003/deu/test.txt


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
