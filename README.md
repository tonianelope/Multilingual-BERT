# GermLM
Exploring Language Models and their effectinves (for German)

1. [Language Models](#LM)
2. [NLP Tasks](#Tasks)
3. [Code](#Code)

- [Timeline](https://docs.google.com/spreadsheets/d/1qDNQsrnsflI8x8Fy9NzZflGyXgI0rxfZDUDpufEVzw0/edit?usp=sharing)
- [Notes](https://docs.google.com/document/d/1VUu5cna6MblNheDGa7tRTsYjlheZEMigvjyFjUEROPE/edit?usp=sharing)

## Language Models <a name="LM"></a>

|Paper | LM Architecture | Corpus | Tasks |
|------|----------------|--------|-------|
|Universal Language Model Fine-tuning<br/> for Text Classification ([ULMFiT]) | [AWD-LSTM] | [Wikitext-103] | - Sentiment Analysis<br/>- Questino Classification <br/>- Topic classification |
|Deep contextualized word<br/> representations ([ELMo]) | biLM | 1B Word Benchmark | - Question Answering<br/>- Textual entailment<br/>- Semantic role labeling <br/>- Coreference resolution<br/>- Named entity extraction<br/>- Sentiment analysis |
|Improving Language Understanding<br/>by Generative Pre-Training ([OpenAi])| Transformer | [Bookcorpus] | - Natural Language Inference  <br/>- Question Answering <br/>- Sentence similarity<br/>- Classificatino

[ULMFiT]: https://arxiv.org/pdf/1801.06146.pdf
[ELMo]: https://arxiv.org/abs/1802.05365
[OpenAi]: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
[AWD-LSTM]: TODO
[Wikitext-103]: https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
[Bookcorpus]: http://yknzhu.wixsite.com/mbweb

## Creating a Language Model

ULMFiT LM [imdb_scripts] for creating a LM (based on Wiki data) and fine-tuning it:

- Create/Load Wiki Corpus
- Tokenise text using [spacy](http://spacy.io/) includes fixup function <-
- Mapp tokens to ids
- (Pre)Train LM

Further example of using LM for IMDb Sentiment Analysis with more detailed explanations/comments: [IMDb ipython](https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb)

### ULMFit for other languages

- German [forum](https://forums.fast.ai/t/ulmfit-german/22529) [github](https://github.com/n-waves/ulmfit4de/blob/master/TRAINING.md)
- ...

### German Corpa

- [Wikipedia](http://www1.ids-mannheim.de/kl/projekte/korpora/archiv/wp.html)
- [COSMA II](http://www.ids-mannheim.de/cosmas2/uebersicht.html)


## Tasks - Applying the Language Model <a name="Tasks"></a>

### English Tasks

| Task                  | Model       | Corpus       | Score | Paper/Source |
|-----------------------|-------------|--------------|-------|--------------|
| Sentiment Analysis    | Transformer | [STT-2][STT] |  91.3 | [OpenAi]     |
|                       | biLSTM      | STT-5        |  54.7 | [ELMo]       |
| Named Entity Rec      | biLSTM      | [CoNNL 2003] | 90.15 | [ElMo]       |
|                       |             |              | 91.93 | [Peters.]    |
| Natural Language Inf  | Transformer | [SNLI 2015]  |  89.9 | [OpenAi]     |
|                       | biLSTM      |              |  88.0 | [ELMo]       |
| Commensense Reasoning | Transformer | [ROCStories] |  86.5 | [OpenAi]     |


[STT]:https://nlp.stanford.edu/sentiment/index.html
[CoNLL 2003]:https://www.clips.uantwerpen.be/conll2003/ner/
[Peters.]:https://www.aclweb.org/anthology/P/P17/P17-1161.pdf
[SNLI 2015]:https://nlp.stanford.edu/projects/snli/
[ROCStories]:http://cs.rochester.edu/nlp/rocstories/

### German Tasks

| Task               | Model  | Corpus         | Score       | Paper/Source               |
| --                 | ---    | ---            | ---         | ---                        |
| Sentiment Analysis | CNN    | [SB-10k]       | 65.09       | [Twitter Corpus+Benchmark] |
|                    | CNN    | MGS            | 59.90       |                            |
|                    | SVM    | DAI            | 61.85       |                            |
| Named Entity Rec   | biLSTM | [GermEval2014] | 81.83 (F1)* | [NER Shootout]             |
|                    | biLSTM | [CoNNL2003]    | 82.99 (F1)  |                            |
|                    |        |                |             |                            |
*Outer chunks evaluation

Related papers:
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)
- [Modular Classifier Ensemble Architecture for Named Entity Recognition on Low Resource Systems](http://asv.informatik.uni-leipzig.de/publication/file/300/GermEval2014_ExB.pdf)


[SB-10K]:http://www.spinningbytes.com/resources/
[GermEval2014]:https://sites.google.com/site/germeval2014ner/data
[CoNLL2011]:http://conll.cemantix.org/2011/data.html


[Twitter Corpus+Benchmark]:http://www.aclweb.org/anthology/W17-1106
[NER Shootout]:http://aclweb.org/anthology/P18-2020.pdf


## Code <a name="Code"></a>

`gc_setup.sh` installs the [fastai] library v0.7
`wt103.sh` downloads the pre-trained [ULMFiT] weights
create a directory `fastai-scripts` and link it to `fastai/courses/dl2/imbd_scripts/*`
use the [imdb_scripts] to train/run/eval the LM

[fastai]:https://github.com/fastai/fastai
[imdb_scripts]:https://github.com/fastai/fastai/tree/master/courses/dl2/imdb_scripts
