EXEC=python run_ner.py

help:
	$(EXEC) --help

test:
	$(EXEC) --ds-size=1 $(W)

apex-test:
	$(EXEC) --ds-size=1 --fp16 $(W)

run:
	$(EXEC) --batch_size=16 --epochs=1 $(W)

apex-run:
	$(EXEC) --batch_size=16 --epochs=1 --fp16 $(W)

datasets:
	make dataset-eng
	make dataset-deu

dataset-eng:
	mkdir -p data/conll-2003/eng
	svn co 'https://github.com/kyzhouhzau/BERT-NER/trunk/NERdata/' data/conll-2003/eng

dataset-deu:
	mkdir -p data/conll-2003/deu
	svn co 'https://github.com/Nidhi-K/Natural-Language-Processing-Projects/trunk/Sequential CRF for NER/data/' data/conll-2003/deu

model:
	wget https://s3.amazonaws.com/models.huggingface.co/bert/$(BERT).tar.gz

clean-logs:
	rm *.log
	rm out.txt

clean-models:
	rm *.tar.gz

rm-dataset:
	rm -rf data

fastai:
	pip install git+https://github.com/fastai/fastai.git
