# GermLM
Exploring Language Models and their effectinves (for German)

## Language Models
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

## German Language Model
### German Corpa

- [Wikipedia](http://www1.ids-mannheim.de/kl/projekte/korpora/archiv/wp.html)
- [COSMA II](http://www.ids-mannheim.de/cosmas2/uebersicht.html)

## Tasks
### Sentiment Analysis
#### Papers
- [A Twitter Corpus and Benchmark Resources for German Sentiment Analysis](http://www.aclweb.org/anthology/W17-1106)

#### Corpa 
- German Twitter Sentiment [SB-10K](http://www.spinningbytes.com/resources/)
- MGS corpus (link?) 
- DAI corpus (link?) 

### NER

#### Papers
- [A Named Entity Recognition Shootout for German](http://aclweb.org/anthology/P18-2020.pdf)
- [Modular Classifier Ensemble Architecture for Named Entity Recognition on Low Resource Systems](http://asv.informatik.uni-leipzig.de/publication/file/300/GermEval2014_ExB.pdf)
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)

#### Corpa
- [GermEval2014](https://sites.google.com/site/germeval2014ner/data)
- [CoNLL2011](http://conll.cemantix.org/2011/data.html)
