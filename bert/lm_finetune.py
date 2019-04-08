"""
modefified from: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/lm_finetuning/finetune_on_pregenerated.py#L281
"""
import json
import logging
import random
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tqdm import tqdm

import fastai.train
import torch
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.callback import Callback
from fastai.torch_core import flatten_model, split_no_wd_params, to_device
from optimizer import BertAdam
from pytorch_pretrained_bert.modeling import BertForPreTraining
from pytorch_pretrained_bert.optimization import warmup_linear  # ,BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

def bert_layer_list(model):
    ms = torch.nn.ModuleList()

    flm = flatten_model(model)
    print(f'Modules Len : {len(flm)}')

    # embedding = [0:5] layer
    ms.append(torch.nn.ModuleList(flm[0:5]))
    # encoder (12 layers) = [5:16] [16:27] ... [126:136]
    bert_layergroup_size = 11#33
    for i in range(5, 137, bert_layergroup_size):
        ms.append(torch.nn.ModuleList(flm[i: i+bert_layergroup_size]))
    # pooling layer = [137:139]
    ms.append(torch.nn.ModuleList(flm[-6:-4]))
    # head = [-2:]
    #ms.append(torch.nn.ModuleList(flm[-4:]))
    print(ms)
#    for i,m in enumerate(split_no_wd_params(ms)):
#        print(f'model{i}')
#        print(m)
#
    return ms

# this PR for ref https://github.com/fastai/fastai/commit/16a858751fc3bd37153a8ada434042f7a53df111
class PregeneratedData(Callback):
    def __init__(self, path, tokenizer, epochs, batch_size, epoch=0):
        self.path = path
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.batch_size = batch_size
        data = PregeneratedDataset(
            epoch=epoch,
            training_path=self.path,
            tokenizer=self.tokenizer,
            num_data_epochs=self.epochs)
        self.dataset = data #DataLoader(data, shuffle=True, batch_size=self.batch_size)

    def __len__(self): return len(self.dataset)
    def __getattr(self, k:str): return getattr(self.dataset, k)

    def on_epoch_begin(self, **kwargs):
        epoch = kwargs['epoch']
        print(epoch)
        data = PregeneratedDataset(
            epoch=epoch,
            training_path=self.path,
            tokenizer=self.tokenizer,
            num_data_epochs=self.epochs)
        self.dataset = data# DataLoader(data, shuffle=True, batch_size=self.batch_size)

    def __getitem__(self, idx:int):
        return self.dataset[idx]

    # def on_batch_begin(self, last_input, last_target, **kwargs):
    #     for batch in self.epoch_dataset:
    #         batch = to_device(batch, torch.cuda.current_device())
    #         t = to_device(torch.tensor([]), torch.cuda.current_device())
    #         yield {'last_input': batch, 'last_target': t}

def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir/'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
        logging.info(f"Loading training examples for epoch {epoch}")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return ((torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64))),
                torch.tensor([]))


def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual", "bert-base-chinese"])
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")

    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help="random seed for initialization")
    args = parser.parse_args()

    assert args.pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"

    samples_per_epoch = []
    for i in range(args.epochs):
        epoch_file = args.pregenerated_data / f"epoch_{i}.json"
        metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.epochs
    print(samples_per_epoch)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    total_train_examples = 0
    for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)

    # Prepare model
    model = BertForPreTraining.from_pretrained(args.bert_model)
    model = torch.nn.DataParallel(model)

    # Prepare optimizer
    optimizer = BertAdam

    train_dataloader = DataLoader(
        PregeneratedData(args.pregenerated_data,  tokenizer,args.epochs, args.train_batch_size),
        batch_size=args.train_batch_size,
    )

    data = DataBunch(train_dataloader,train_dataloader)
    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {total_train_examples}")
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)

    learn = Learner(data, model, optimizer,
                    loss_func=lambda x: x,
                    true_wd=False,
                    path='learn',
                    layer_groups=bert_layer_list(model),
    )
    lr= args.learning_rate
    lrs = learn.lr_range(slice(lr/(1.3**15), lr))
    learn.fit_one_cycle(args.epochs, lrs, wd=1e-4)

    m_name = 'Pretrained_lm_'+lang
    m_path = learn.save(m_name, return_path=True)
    print(f'Saved model to {m_path}')

if __name__ == '__main__':
    main()
