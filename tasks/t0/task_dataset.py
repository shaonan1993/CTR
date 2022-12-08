import numpy as np
from torch.utils.data import Dataset

from utils import print_rank_0


class TaskDataset(Dataset):
    def __init__(self, task_name, args, split, tokenizer, source_texts, target_texts):
        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.max_src_length, self.max_tgt_length = args.multi_src_seq_length, args.multi_tgt_seq_length
        self.source_texts = source_texts
        self.target_texts = target_texts

        assert len(self.source_texts) == len(self.target_texts)
        print_rank_0(f"Task-Data | Name: {task_name} | Split: {split} | # Packed-Data: {len(self.source_texts)}.")

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text, target_text = self.source_texts[idx], self.target_texts[idx]

        pad_id = self.tokenizer.get_command('pad').Id
        eos_id = self.tokenizer.get_command('eos').Id

        source_tokens = self.tokenizer.EncodeAsIds(source_text).tokenization
        if len(source_tokens) >= self.max_src_length - 1:
            source_tokens = source_tokens[: (self.max_src_length - 1)]
        source_tokens = source_tokens + [eos_id]
        attention_mask = [1] * len(source_tokens)
        if len(source_tokens) < self.max_src_length:
            pad_length = self.max_src_length - len(source_tokens)
            source_tokens = source_tokens + [pad_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        target_tokens = self.tokenizer.EncodeAsIds(target_text).tokenization
        if len(target_tokens) >= self.max_tgt_length - 1:
            target_tokens = target_tokens[: (self.max_tgt_length - 1)]
        target_tokens = target_tokens + [eos_id]
        loss_mask = [1] * len(target_tokens)
        if len(target_tokens) < self.max_tgt_length:
            pad_length = self.max_tgt_length - len(target_tokens)
            target_tokens = target_tokens + [pad_id] * pad_length
            loss_mask = loss_mask + [0] * pad_length

        sample = {
            'text': np.array(source_tokens, dtype=np.int64),
            'target': np.array(target_tokens, dtype=np.int64),
            'attention_mask': np.array([[attention_mask]], dtype=np.int64),
            'loss_mask': np.array(loss_mask, dtype=np.int64)
        }
        return sample


class TaskDatasetWithId(Dataset):
    def __init__(self, task_name, args, split, tokenizer, source_texts, target_texts, task_id):
        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.max_src_length, self.max_tgt_length = args.multi_src_seq_length, args.multi_tgt_seq_length
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.task_id = task_id
        self.task_ids = list(range(task_id*args.prompt_num_per_task, (task_id+1)*args.prompt_num_per_task))

        assert len(self.source_texts) == len(self.target_texts)
        print_rank_0(f"Task-Data | Name: {task_name} | Split: {split} | # Packed-Data: {len(self.source_texts)}.")

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text, target_text = self.source_texts[idx], self.target_texts[idx]

        pad_id = self.tokenizer.get_command('pad').Id
        eos_id = self.tokenizer.get_command('eos').Id

        source_tokens = self.tokenizer.EncodeAsIds(source_text).tokenization
        if len(source_tokens) >= self.max_src_length - 1:
            source_tokens = source_tokens[: (self.max_src_length - 1)]
        source_tokens = source_tokens + [eos_id]
        attention_mask = [1] * len(source_tokens)
        if len(source_tokens) < self.max_src_length:
            pad_length = self.max_src_length - len(source_tokens)
            source_tokens = source_tokens + [pad_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        target_tokens = self.tokenizer.EncodeAsIds(target_text).tokenization
        if len(target_tokens) >= self.max_tgt_length - 1:
            target_tokens = target_tokens[: (self.max_tgt_length - 1)]
        target_tokens = target_tokens + [eos_id]
        loss_mask = [1] * len(target_tokens)
        if len(target_tokens) < self.max_tgt_length:
            pad_length = self.max_tgt_length - len(target_tokens)
            target_tokens = target_tokens + [pad_id] * pad_length
            loss_mask = loss_mask + [0] * pad_length

        sample = {
            'text': np.array(source_tokens, dtype=np.int64),
            'target': np.array(target_tokens, dtype=np.int64),
            'attention_mask': np.array([[attention_mask]], dtype=np.int64),
            'loss_mask': np.array(loss_mask, dtype=np.int64),
            'task_id': np.array(self.task_id, dtype=np.int64),
            'task_ids': np.array(self.task_ids, dtype=np.int64)
        }
        return sample


class TaskDatasetWithCode(Dataset):
    def __init__(self, task_name, args, split, tokenizer, source_texts, target_texts, code_list):
        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.max_src_length, self.max_tgt_length = args.multi_src_seq_length, args.multi_tgt_seq_length
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.code_list = code_list

        assert len(self.source_texts) == len(self.target_texts)
        print_rank_0(f"Task-Data | Name: {task_name} | Split: {split} | # Packed-Data: {len(self.source_texts)}.")

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text, target_text = self.source_texts[idx], self.target_texts[idx]

        pad_id = self.tokenizer.get_command('pad').Id
        eos_id = self.tokenizer.get_command('eos').Id

        source_tokens = self.tokenizer.EncodeAsIds(source_text).tokenization
        if len(source_tokens) >= self.max_src_length - 1:
            source_tokens = source_tokens[: (self.max_src_length - 1)]
        source_tokens = source_tokens + [eos_id]
        attention_mask = [1] * len(source_tokens)
        if len(source_tokens) < self.max_src_length:
            pad_length = self.max_src_length - len(source_tokens)
            source_tokens = source_tokens + [pad_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        target_tokens = self.tokenizer.EncodeAsIds(target_text).tokenization
        if len(target_tokens) >= self.max_tgt_length - 1:
            target_tokens = target_tokens[: (self.max_tgt_length - 1)]
        target_tokens = target_tokens + [eos_id]
        loss_mask = [1] * len(target_tokens)
        if len(target_tokens) < self.max_tgt_length:
            pad_length = self.max_tgt_length - len(target_tokens)
            target_tokens = target_tokens + [pad_id] * pad_length
            loss_mask = loss_mask + [0] * pad_length

        sample = {
            'text': np.array(source_tokens, dtype=np.int64),
            'target': np.array(target_tokens, dtype=np.int64),
            'attention_mask': np.array([[attention_mask]], dtype=np.int64),
            'loss_mask': np.array(loss_mask, dtype=np.int64),
            'task_code': np.array(self.code_list, dtype=np.int64)
        }
        return sample


class TaskDatasetWithFeature(Dataset):
    def __init__(self, task_name, args, split, tokenizer, source_texts, target_texts, feature_ids):
        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.max_src_length, self.max_tgt_length = args.multi_src_seq_length, args.multi_tgt_seq_length
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.feature_ids = feature_ids

        # padding feature ids
        if len(self.feature_ids) < args.max_feature_num_per_task:
            self.feature_ids = self.feature_ids + [args.max_feature_num-1] * (args.max_feature_num_per_task - len(self.feature_ids))

        assert len(self.feature_ids) == args.max_feature_num_per_task, f'feature_ids: {self.feature_ids}'

        assert len(self.source_texts) == len(self.target_texts)
        print_rank_0(f"Task-Data | Name: {task_name} | Split: {split} | # Packed-Data: {len(self.source_texts)}.")

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text, target_text = self.source_texts[idx], self.target_texts[idx]

        pad_id = self.tokenizer.get_command('pad').Id
        eos_id = self.tokenizer.get_command('eos').Id

        source_tokens = self.tokenizer.EncodeAsIds(source_text).tokenization
        if len(source_tokens) >= self.max_src_length - 1:
            source_tokens = source_tokens[: (self.max_src_length - 1)]
        source_tokens = source_tokens + [eos_id]
        attention_mask = [1] * len(source_tokens)
        if len(source_tokens) < self.max_src_length:
            pad_length = self.max_src_length - len(source_tokens)
            source_tokens = source_tokens + [pad_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        target_tokens = self.tokenizer.EncodeAsIds(target_text).tokenization
        if len(target_tokens) >= self.max_tgt_length - 1:
            target_tokens = target_tokens[: (self.max_tgt_length - 1)]
        target_tokens = target_tokens + [eos_id]
        loss_mask = [1] * len(target_tokens)
        if len(target_tokens) < self.max_tgt_length:
            pad_length = self.max_tgt_length - len(target_tokens)
            target_tokens = target_tokens + [pad_id] * pad_length
            loss_mask = loss_mask + [0] * pad_length

        sample = {
            'text': np.array(source_tokens, dtype=np.int64),
            'target': np.array(target_tokens, dtype=np.int64),
            'attention_mask': np.array([[attention_mask]], dtype=np.int64),
            'loss_mask': np.array(loss_mask, dtype=np.int64),
            'feature_ids': np.array(self.feature_ids, dtype=np.int64)
        }
        return sample