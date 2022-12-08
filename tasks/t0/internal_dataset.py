import json
import os
import random
import time

import numpy as np
import sklearn.utils
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from tasks.data_utils import InputExample
from tasks.t0.t0_global_vars import P3_TASK_LIST
from utils import print_rank_0, tqdm_rank_0
from SwissArmyTransformer import mpu

t0_task_to_prompt_number = {
        'ag_news': 7,
        'app_reviews': 4,
        'wiki_bio': 5,
        'cnn_dailymail/3.0.0': 9,
        'gigaword': 9,
        'wiki_hop/original': 9,
        'glue/mrpc': 7,
        'glue/qqp': 6,
        'amazon_polarity': 9,
        'paws/labeled_final': 12,
        'dbpedia_14': 4,
        'dream': 5,
        'kilt_tasks/hotpotqa': 5,
        'trec': 18,
        'multi_news': 6,
        'samsum': 7,
        'xsum': 10,
        'imdb': 11,
        'rotten_tomatoes': 10,
        'yelp_review_full': 7,
        'wiki_qa': 11,
        'common_gen': 9,
        'adversarial_qa/dbidaf': 5,
        'adversarial_qa/dbert': 5,
        'adversarial_qa/droberta': 5,
        'quoref': 11,
        'ropes': 12,
        'duorc/SelfRC': 9,
        'duorc/ParaphraseRC': 9,
        'sciq': 5,
        'quarel': 5,
        'qasc': 8,
        'cosmos_qa': 13,
        'wiqa': 8,
        'social_i_qa': 6,
        'quail': 13,
        'quartz': 8,
        'cos_e/v1.11': 11
        }


def get_task_prompt_number(prompted_task_name, multi_task_set):
    # assert multi_task_set == "t0"
    for task_name in list(t0_task_to_prompt_number.keys()):
        cur_task_name = task_name.replace("/", "_")
        if prompted_task_name.startswith(cur_task_name):
            return t0_task_to_prompt_number[task_name]
    return None


class InternalDataProcessor:
    def __init__(self, task_name, args, tokenizer, **kwargs):
        self.task_name = task_name
        self.args = args
        self.tokenizer = tokenizer
        self.max_src_length, self.max_tgt_length = self.args.multi_src_seq_length, self.args.multi_tgt_seq_length
        # assert self.max_src_length == 512
        # assert self.max_tgt_length == 256
        print_rank_0(f'InternalDataProcessor: self.max_src_length == {self.max_src_length}, self.max_tgt_length = {self.max_tgt_length}')
        self.data_dir = args.multi_cache_dir
        self.tlm_data_cache_dir = args.tlm_data_cache_dir
        self.max_task_dataset_size = args.max_task_dataset_size

    def create_examples(self, split):
        source_texts, target_texts = [], []
        current_idx = mpu.get_data_parallel_rank() % self.args.loader_scatter  # [0,1,2,3,4,5,6,7]

        filepath = os.path.join(self.data_dir, self.task_name, split + ".json")
        dataset = load_dataset("json",
                               data_files={split: filepath},
                               cache_dir=self.tlm_data_cache_dir,
                               split=split)
        # """
        # filepath = os.path.join(self.data_dir, self.task_name, split)
        # dataset = load_from_disk(filepath)
        total_number = len(dataset)
        print_rank_0(f"Original total number: {total_number} | split: {split} | task: {self.task_name}.")

        if total_number >= self.max_task_dataset_size:
            prompt_number = get_task_prompt_number(self.task_name, self.args.multi_task_set)
            assert prompt_number is not None
            current_number = int(self.max_task_dataset_size / prompt_number)
            rng = random.Random(1234)
            random_list = rng.sample(population=list(range(total_number)), k=current_number)
        else:
            current_number = total_number
            random_list = list(range(total_number))

        start_number = int(current_idx / self.args.loader_scatter * current_number)
        end_number = int((current_idx + 1) / self.args.loader_scatter * current_number)
        idx_list = random_list[start_number:end_number] if (start_number < end_number) else [random_list[start_number]]
        cur_data = dataset.select(idx_list)
        print_rank_0(f"Truncated Data Per-rank number: {len(cur_data)} | split: {split} | task: {self.task_name}.")
        print_rank_0("\n")
        for data_example in cur_data:
            source_texts.append(data_example["inputs_pretokenized"])
            target_texts.append(data_example["targets_pretokenized"])
        return source_texts, target_texts


class InternalDataset(Dataset):
    def __init__(self, split, args, tokenizer, source_texts):
        self.args = args
        self.tokenizer = tokenizer
        self.max_src_length = args.multi_src_seq_length
        self.example_list = self.create_packed_examples(source_texts)
        print_rank_0(f"Internal | split: {split} | Number of Packed Examples: {len(self.example_list)}.")

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        sample = {
            'text': np.array(example, dtype=np.int64)
        }
        return sample

    def create_packed_examples(self, source_texts):
        example_list = []
        prev_toks = []
        # idx = 0
        for text in tqdm_rank_0(source_texts):
            tok = self.tokenizer.EncodeAsIds(text).tokenization
            length = self.max_src_length - len(prev_toks)
            if len(tok) < length:
                prev_toks = prev_toks + tok
            elif len(tok)  == length:
                prev_toks = prev_toks + tok
                example_list.append(prev_toks)
                # idx += 1
                prev_toks = []
            else:
                prev_toks = prev_toks + tok[:length]
                example_list.append(prev_toks)
                # idx += 1
                prev_toks = []
        return example_list
