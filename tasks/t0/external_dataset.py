import json
import os
import random
import time

import h5py
import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm
import pickle as pkl

from data_utils import LazyLoader
from tasks.data_utils import InputExample
from utils import print_rank_0, tqdm_rank_0
from SwissArmyTransformer import mpu


class ExternalDataProcessor:
    def __init__(self, args, tokenizer, lazy_seq2seq_loader=False, **kwargs):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_length = args.multi_src_seq_length
        self.lazy_seq2seq_loader = lazy_seq2seq_loader
        self.loader_scatter = args.loader_scatter

        self.external_data_dir = args.external_multi_cache_dir
        self.tlm_data_cache_dir = args.tlm_data_cache_dir
        self.split="external"

    def create_example(self, data):
        return data["text"]

    def create_examples(self):
        if not self.lazy_seq2seq_loader:
            texts = []
            current_idx = mpu.get_data_parallel_rank() % self.args.loader_scatter  # [0,1,2,3,4,5,6,7]
            filepath = os.path.join(self.external_data_dir, f"{self.max_seq_length}_unique_external_data.csv")
            if not os.path.exists(filepath):
                raise ValueError(f"The file {filepath} does not exist.")
            start = int(current_idx / self.args.loader_scatter * 100)
            end = int((current_idx + 1) / self.args.loader_scatter * 100)
            split_str = self.split + f"[{start}%:{end}%]"

            alldata = load_dataset("csv",data_files={self.split: filepath},split=split_str, cache_dir=self.tlm_data_cache_dir)
            print_rank_0(f"Per-rank number of external: {len(alldata)} | current_rank: {mpu.get_data_parallel_rank()}.")

            """
            data_list = [idx for idx in list(range(32)) if (idx % self.args.loader_scatter == current_idx)]
            dataset_list = []
            for idx in data_list:
                cur_path = os.path.join(self.external_data_dir, "external", str(idx))
                dataset = load_from_disk(cur_path)
                dataset_list.append(dataset)
            assert len(dataset_list) == int(32/self.args.loader_scatter)
            alldata = concatenate_datasets(dataset_list)
            """

            for data_example in tqdm_rank_0(alldata):
                if data_example["text"] is None:
                    print(data_example)
                    continue
                texts.append(data_example["text"])
            print_rank_0(f"Per-rank number of external: {len(texts)} | current_rank: {mpu.get_data_parallel_rank()}.")
        else:
            data_dir = self.external_data_dir
            name = f"{self.max_seq_length}_unique_external_data"
            lenpath = os.path.join(data_dir, name + ".len.pkl")
            datapath = os.path.join(data_dir, name + ".json")
            if not os.path.exists(lenpath) and torch.distributed.get_rank() == 0:
                lengths = []
                with open(datapath) as file:
                    for line in file:
                        lengths.append(len(line.encode('utf-8')))
                with open(lenpath, 'wb') as f:
                    pkl.dump(lengths, f)
            else:
                while not os.path.exists(lenpath):
                    time.sleep(1)
            loader_range = None
            if self.loader_scatter is not None:
                scatter_rank = mpu.get_data_parallel_rank() % self.loader_scatter
                loader_range = (
                    1.0 * scatter_rank / self.loader_scatter, 1.0 * (scatter_rank + 1) / self.loader_scatter)
            map_fn = lambda x: self.create_example(json.loads(x))
            texts = LazyLoader(data_type=self.split, map_fn=map_fn, mem_map=True,
                                      is_array=False, loader_range=loader_range, lenpath=lenpath, datapath=datapath)
            print_rank_0(f"Creating {len(texts)} examples for external dataset.")

        return texts

    """
    def create_examples(self):
        texts = []
        print_rank_0(f"Creating {self.split} dataset for task {self.task_name}.")
        assert self.args.loader_scatter == 8, "--loader_scatter should be fixed to be 8."
        current_idx = mpu.get_data_parallel_rank() % 8  # [0,1,2,3,4,5,6,7]

        filepath = os.path.join(self.external_data_dir, self.task_name, "selected_unique.csv")
        if not os.path.exists(filepath):
            raise ValueError(f"The file {filepath} does not exist.")
        dataset = load_dataset("csv", data_files={self.split: filepath}, split=self.split, cache_dir=self.tlm_data_cache_dir)

        total_data_num = len(dataset)
        print_rank_0(f"Original total number of data: {total_data_num}.")
        start = int(current_idx / 8 * total_data_num)
        end = int((current_idx + 1) / 8 * total_data_num)
        cur_data = dataset[start:end]
        print_rank_0(f"Per-rank number of data: {len(cur_data)}.")
        for idx, data_example in enumerate(cur_data):
            texts.append(data_example["text"])
        return texts
    """

class ExternalDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.max_src_length = args.multi_src_seq_length
        self.tokenizer = tokenizer

        self.external_processor = ExternalDataProcessor(args, tokenizer)
        text_list = self.external_processor.create_examples()
        # text_list = sklearn.utils.shuffle(text_list)

        self.example_list = self.create_packed_examples(text_list)

        """
        save_path = os.path.join(args.external_data_dir, "packed_external")
        assert os.path.exists(save_path)
        current_rank = mpu.get_data_parallel_rank()
        current_idx = current_rank % self.args.loader_scatter
        if current_rank == current_idx:
            save_file_path = os.path.join(save_path, f"{current_idx}.h5")
            if not os.path.exists(save_file_path):
                hf = h5py.File(save_file_path, 'w')
                current_data = np.array(self.example_list, dtype=np.int64)
                print(current_data.shape)
                hf.create_dataset(f'packed_external_{current_rank}',
                                  data=current_data,
                                  dtype="int64")
                print(f"Rank {current_rank} done saving external data !!!")
        """

        print_rank_0("\n")
        print_rank_0(f"External | Number of Packed Examples: {len(self.example_list)}.")
        print_rank_0("\n")

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        sample = {
            'text': np.array(example, dtype=np.int64)
        }
        return sample

    def create_packed_examples(self, texts):
        example_list = []
        prev_toks = []
        idx = 0
        for text in tqdm_rank_0(texts):
            toks = self.tokenizer.EncodeAsIds(text).tokenization
            length = self.max_src_length - len(prev_toks)
            if len(toks) < length:
                prev_toks = prev_toks + toks
            elif len(toks) == length:
                prev_toks = prev_toks + toks
                example_list.append(prev_toks)
                idx += 1
                prev_toks = []
            else:
                prev_toks = prev_toks + toks[:length]
                example_list.append(prev_toks)
                idx += 1
                prev_toks = []
        return example_list