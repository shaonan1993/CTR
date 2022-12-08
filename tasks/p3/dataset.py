import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm

from data_utils import exists_lazy, LazyLoader, LazyWriter
from tasks.data_utils import InputExample
from tasks.p3.p3 import large_t0_task_dict, _MAX_DATASET_SIZE
from tasks.p3.pvp import P3PVP
from tasks.superglue.dataset import TEST_SET
from utils import print_rank_0
from SwissArmyTransformer import mpu

task_to_prompts = {
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

class DataProcessor:
    def __init__(self, args, task_name, tokenizer, lazy_seq2seq_loader=False, **kwargs):
        self.args = args
        self.data_dir = args.multi_cache_dir
        self.max_src_len = self.args.multi_src_seq_length
        self.max_tgt_len = self.args.multi_tgt_seq_length
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.lazy_seq2seq_loader = lazy_seq2seq_loader
        self.loader_scatter = args.loader_scatter
        """
        if args.cache_dir is not None:
            self.cache_dir = os.path.join(args.cache_dir, "datasets")
        else:
            self.cache_dir = None
        """

    def _yield_examples(self, split, dataset):
        raise NotImplementedError

    def create_examples(self, split):
        print_rank_0(f"Creating {split} dataset from {self.data_dir} for task {self.task_name}.")
        if not self.lazy_seq2seq_loader:
            assert self.loader_scatter == 8, "--loader_scatter should be fixed to be 8."
            current_idx = mpu.get_data_parallel_rank() % self.loader_scatter  # [0,1,2,3,4,5,6,7]
            filepath = os.path.join(self.data_dir, self.task_name, split + ".json")
            print_rank_0(self.task_name)
            dataset = load_dataset("json", data_files={split: filepath}, split=split,
                                   cache_dir=self.args.tlm_data_cache_dir)

            total_number = len(dataset)
            # if self.task_name in list(large_t0_task_dict.keys()) and split == "train":
            #    assert total_number >= _MAX_DATASET_SIZE
            if total_number >= 50000:
                # current_number = large_t0_task_dict[self.task_name]
                # prefix = [name for name in T0_TRAIN_TASK_NAME if name.replace("/", "_").startswith(self.task_name)]
                # print(prefix)
                print(self.task_name)
                prompt_key=None
                for task_name in list(task_to_prompts.keys()):
                    new_task_name = task_name.replace("/", "_")
                    if self.task_name.startswith(new_task_name):
                        prompt_key = task_name

                current_number = int(50000 / task_to_prompts[prompt_key])
                rng = random.Random(1234)
                random_list = rng.sample(population=list(range(total_number)), k=current_number)
            else:
                current_number = total_number
                random_list = list(range(total_number))

            start_number = int(current_idx / self.loader_scatter * current_number)
            end_number = int((current_idx + 1) / self.loader_scatter * current_number)
            idx_list = random_list[start_number: end_number]
            dataset = dataset.select(idx_list)
            example_list = []
            for idx, example in enumerate(self._yield_examples(split, dataset)):
                if (idx + 1) % 20000 == 0:
                    print_rank_0(f"Complete {idx + 1} examples")
                example_list.append(example)
        else:
            """
            data_dir = os.path.join(self.data_dir, self.task_name)
            if not exists_lazy(data_dir, data_type=split) and torch.distributed.get_rank() == 0:
                example_writer = LazyWriter(self.data_dir, data_type=split, is_array=False)
                filepath = os.path.join(data_dir, split + ".json")
                dataset = load_dataset("json", data_files={split: filepath}, split=split, cache_dir=self.cache_dir)
                for idx, example in enumerate(self._yield_examples(split, dataset)):
                    if (idx + 1) % 20000 == 0:
                        print_rank_0(f"Complete {idx + 1} examples")
                    example_writer.write(example)
                example_writer.close()
            else:
                while not os.path.exists(LazyWriter.get_len_path(data_dir, data_type=split)):
                    time.sleep(1)
            loader_range = None
            if self.loader_scatter is not None:
                scatter_rank = mpu.get_data_parallel_rank() % self.loader_scatter
                loader_range = (
                    1.0 * scatter_rank / self.loader_scatter, 1.0 * (scatter_rank + 1) / self.loader_scatter)
            example_list = LazyLoader(data_dir, data_type=split, map_fn=InputExample.from_json_string,
                                      mem_map=True, is_array=False, loader_range=loader_range)
            """
            raise ValueError("")
        print_rank_0(f"Creating {len(example_list)} examples for {split} of task {self.task_name}.")
        return example_list


class P3Processor(DataProcessor):
    def _yield_examples(self, split, dataset):
        source_texts, target_texts = [], []
        assert "inputs_pretokenized" in dataset.features
        assert "targets_pretokenized" in dataset.features
        for data_example in dataset:
            source_text = data_example["inputs_pretokenized"]
            source_texts.append(source_text)
            target_text = data_example["targets_pretokenized"]
            target_texts.append(target_text)
        assert len(source_texts) == len(target_texts)

        eos_token = self.tokenizer.get_command('eos').token

        def exceed_maximum_length(prev_inputs, inputs, max_seq_len):
            assert isinstance(prev_inputs, str) and isinstance(inputs, str)
            prev_tok = self.tokenizer.EncodeAsIds(prev_inputs).tokenization
            assert len(prev_tok) <= max_seq_len
            tok = self.tokenizer.EncodeAsIds(inputs).tokenization

            if len(tok) >= max_seq_len - 2:
                tok = tok[:(max_seq_len - 2)]
                inputs = self.tokenizer.DecodeIds(tok)

            if len(prev_tok) + len(tok) < max_seq_len - 2:
                ret_inputs = prev_inputs + (inputs + eos_token)
                return False, ret_inputs
            else:
                ret_inputs = prev_inputs
                return True, ret_inputs

        if True:
            for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
                guid = "%s-%s" % (split, idx)
                meta = {"ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(target_text).tokenization)}
                example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
                if idx < 3:
                    print_rank_0(
                        (source_text.encode('utf-8'), target_text.encode('utf-8'), meta["ref"].encode('utf-8')))
                yield example


class P3Dataset(Dataset):
    def __init__(self, args, task_name, split, tokenizer):
        self.args = args
        self.task_name = task_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_src_length, self.max_tgt_length = args.multi_src_seq_length, args.multi_tgt_seq_length

        self.processor = P3Processor(self.args, self.task, tokenizer, lazy_seq2seq_loader=False)
        example_list = self.processor.create_examples(split)
        self.example_list = example_list
        self.examples = {example.guid: example for example in example_list}
        print_rank_0(f"Return {len(self.examples)} {split} examples for task {task_name}.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        pad_id = self.tokenizer.get_command('pad').Id
        sop_id = self.tokenizer.get_command('sop').Id

        if self.t5_model:
            eos_id = self.tokenizer.get_command('eos').Id
            source_text, target_text = example.text_a, example.text_b

            if True:
                source_tokens = self.tokenizer.EncodeAsIds(source_text).tokenization
                if len(source_tokens) > self.max_src_length - 1:
                    source_tokens = source_tokens[: (self.max_src_length - 1)]
                source_tokens = source_tokens + [eos_id]

            attention_mask = [1] * len(source_tokens)
            if len(source_tokens) < self.max_src_length:
                pad_length = self.max_src_length - len(source_tokens)
                source_tokens = source_tokens + [pad_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length

            # if self.split == "train" or self.split == "validation":
            if self.is_training:
                if True:
                    target_tokens = self.tokenizer.EncodeAsIds(target_text).tokenization
                    if len(target_tokens) > self.max_tgt_length - 1:
                        target_tokens = target_tokens[: (self.max_tgt_length - 1)]
                    target_tokens = target_tokens + [eos_id]

                loss_mask = [1] * len(target_tokens)
                if len(target_tokens) < self.max_tgt_length:
                    pad_length = self.max_tgt_length - len(target_tokens)
                    target_tokens = target_tokens + [pad_id] * pad_length
                    loss_mask = loss_mask + [0] * pad_length

                sample = {'text': np.array(source_tokens, dtype=np.int64),
                          'target': np.array(target_tokens, dtype=np.int64),
                          'attention_mask': np.array([[attention_mask]], dtype=np.int64),
                          'loss_mask': np.array(loss_mask, dtype=np.int64),
                          "uid": example.guid}
            else:  ### TODO: test sets of training tasks are not used.
                sample = {
                    'text': np.array(source_tokens, dtype=np.int64),
                    'attention_mask': np.array([[attention_mask]], dtype=np.int64),
                    "uid": example.guid
                }
        else:
            eop_id = self.tokenizer.get_command('eop').Id
            pvp = P3PVP(self.tokenizer,
                        max_src_length=self.max_src_length,
                        max_tgt_length=self.max_tgt_length,
                        task_mask=self.args.task_mask)
            mask_id = pvp.mask_id
            source_tokens, target_text = pvp.encode(example)

            if len(source_tokens) < self.max_src_length:
                source_tokens = source_tokens + [pad_id] * (self.max_src_length - len(source_tokens))
            sep = len(source_tokens)
            position_ids = list(range(len(source_tokens)))
            block_position_ids = [0] * len(source_tokens)
            mask_pos = source_tokens.index(mask_id)

            # if self.split == 'train' or self.split == "validation":
            if self.is_training:
                target_tokens = self.tokenizer.EncodeAsIds(" " + target_text).tokenization
                target_tokens = target_tokens + [eop_id]

                if len(target_tokens) > self.max_tgt_length:
                    target_tokens = target_tokens[:self.max_tgt_length]
                loss_mask = [1] * len(target_tokens)

                if len(target_tokens) < self.max_tgt_length:
                    loss_mask += [0] * (self.max_tgt_length - len(target_tokens))
                    target_tokens += [pad_id] * (self.max_tgt_length - len(target_tokens))

                tokens = source_tokens + [sop_id] + target_tokens[:-1]
                loss_mask = [0] * len(source_tokens) + loss_mask
                target_ids = [0] * len(source_tokens) + target_tokens
                position_ids += [mask_pos] * len(target_tokens)
                if self.args.no_block_position:
                    block_position_ids += [1] * len(target_tokens)
                else:
                    block_position_ids += list(range(1, len(target_tokens) + 1))
                position_ids = [position_ids, block_position_ids]
                sample = {'text': np.array(tokens, dtype=np.int64),
                          'target': np.array(target_ids, dtype=np.int64),
                          'attention_mask': np.array(sep, dtype=np.int64),
                          'loss_mask': np.array(loss_mask, dtype=np.int64),
                          "position_id": np.array(position_ids, dtype=np.int64),
                          "uid": example.guid}
            else:
                tokens = source_tokens + [sop_id]
                position_ids = position_ids + [mask_pos]
                block_position_ids = block_position_ids + [1]
                position_ids = [position_ids, block_position_ids]
                sample = {'text': np.array(tokens, dtype=np.int64),
                          'attention_mask': np.array(sep, dtype=np.int64),
                          "position_id": np.array(position_ids, dtype=np.int64),
                          "uid": example.guid}
        return sample
