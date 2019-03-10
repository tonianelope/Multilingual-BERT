EXEC=python run_ner.py

test:
	$(EXEC) --ds-size=1

apex-test:
	$(EXEC) --ds-size=1 --fp16

run:
	$(EXEC) --batch_size=16 --epochs=7

apex-run:
	$(EXEC) --batch_size=16 --epochs=7 --fp16

datasets:
	make dataset-eng
	make dataset-deu

dataset-eng:
	mkdir -p data/conll-2003/eng
	svn co 'https://github.com/kyzhouhzau/BERT-NER/trunk/NERdata/' data/conll-2003/eng

dataset-deu:
	mkdir -p data/conll-2003/deu
	svn co 'https://github.com/Nidhi-K/Natural-Language-Processing-Projects/trunk/Sequential CRF for NER/data/' data/conll-2003/deu
