# GermLM
Exploring Multilingual Language Models and their effectinves for NER in German and English

## Requierements
Requierements can be installed via the `requierements.txt`.
We use 
* pytorch
* [pytorch_pretrained_bert](https://github.com/huggingface/pytorch-pretrained-BERT/)
* [fastai]

It is recommended to run this on at least 1 GPU. Our experiments were conducted using 2.

#### (Example) Replicating English BERT NER experiment
Create the appropriate datasets using the makefile

Run `run_ner.py`. Usage (listing the most important options) :
* `lang`: select the language to train. Supported languages are `eng`, `deu`, and `engm` using the english data on the multilingual models
* `batch_size`
* `lr`: define learning rate
* `epochs`: define epochs to train
* `dataset`: path to dataset. Note: `lang` will be appended to this path to access the language specific dataset.
* `loss`: set to `zero` to mask of all padding during loss calculation
* `ds_size`: limit the dataset loaded for testing
* `bertAdam`: if flag set uses the BertAdam optimiser

### Replicate English BERT NER
Create the dataset:
```shell
make dataset-engI
```
Train the NER model:
```
python run_ner.py --do-train --do-eval --lr=3e-5 --batch-size=16 --epochs=4 --bertAdam --dataset=data/conll-2003-I/
```

### Try out the model
If you use `run_ner.py` with the `save` flag, the saved model can be loaded in `predict.py` and it will recognise the named entities of the senteces provided. Note, you just need to proved the file name, the learner will automatically look for it in it's directory and append to correct extension.

```
python predict.py eng_3_model
```

## Fine-tuning

### LM - pretraining

Use `conl_to_docs` from `ner_data.py` to convert the trainings set to a document of sentences.

Use the output file you specified as input to the data generation:
```
make 2bert DIR='data/conll-2003/eng/' M='bert-base-cased' E=20
```

Then fine-tune the language model on the task data:
```
make pretrain_lm FILE='lm_finetune.py' DIR='data/conll-2003/deu/' M='bert-base-multilingual-cased' E=20 
```

### Task-finetuning

Learnig rates were selected using the jupter notebooks.

Run `task-finetuning.py` to fine-tuning using the tuning methods from [ULMFIT]. Add `tuned_learner` to load the fine-tuned LM:
```
python task-finetuning.py --batch-size=16 --epochs=4 --lr=5e-5 --do-train --do-eval --dataset=data/conll-2003-I/ --lang=deu --tuned-learner='pretrain/pytorch_fastai_model_i_bert-base-multilingual-cased_10.bin'
```

[BERT]:https://arxiv.org/pdf/1810.04805.pdf
[ULMFiT]: https://arxiv.org/pdf/1801.06146.pdf
[ELMo]: https://arxiv.org/abs/1802.05365
[OpenAi]: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
[AWD-LSTM]: TODO
[Wikitext-103]: https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
[Bookcorpus]: http://yknzhu.wixsite.com/mbweb


[CoNLL 2003]:https://www.clips.uantwerpen.be/conll2003/ner/
[Peters.]:https://www.aclweb.org/anthology/P/P17/P17-1161.pdf
[SNLI 2015]:https://nlp.stanford.edu/projects/snli/
[ROCStories]:http://cs.rochester.edu/nlp/rocstories/

[SB-10K]:http://www.spinningbytes.com/resources/
[GermEval2014]:https://sites.google.com/site/germeval2014ner/data
[CoNLL2011]:http://conll.cemantix.org/2011/data.html

[Twitter Corpus+Benchmark]:http://www.aclweb.org/anthology/W17-1106
[NER Shootout]:http://aclweb.org/anthology/P18-2020.pdf
[fastai]:https://github.com/fastai/fastai
[imdb_scripts]:https://github.com/fastai/fastai/tree/master/courses/dl2/imdb_scripts
