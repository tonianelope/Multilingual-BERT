from pytorch_pretrained_bert.tokenization import BertTokenizer as BT

# constants for bert vars
BERT_BASE_UNCASED = 'bert-base-uncased'
BERT_BASE_UNCASED_LOWER = 'bert-base-uncased-lower'

class BertTokenizer(BaseTokenizer):
    "Wrapper around huggingfaces BertTokenizer to make it a fastai `BaseTokenizer`"
    "See fastai docs: https://docs.fast.ai/text.transform.html#SpacyTokenizer"

    toks = {
        BERT_BASE_UNCASED_LOWER: BT.from_pretrained(BERT_BASE_UNCASED, do_lower_case=True),
        BERT_BASE_UNCASED: BT.from_pretrained(BERT_BASE_UNCASED, do_lower_case=False)
    }

    def __init__(self, lang:str):
        # lang repurposed for bert type
        self.tok = self.toks[lang]

    def tokenizer(self, t:str):
        return [t for t in self.tok.tokenize(t)]

    def add_special_cases(self, toks): pass
