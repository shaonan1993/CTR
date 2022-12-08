import torch
import torch.utils.data
from SwissArmyTransformer import mpu
import random
import copy
import numpy as np
import math
from utils import print_rank_0
from scipy.stats import poisson
from typing import List, Dict


def rindex(lst, val, start=None):
    if start is None:
        start = len(lst) - 1
    for i in range(start, -1, -1):
        if lst[i] == val:
            return i
    return -1


def index_in_list(lst, val, start=None):
    if start is None:
        start = 0
    for i in range(start, len(lst)):
        if lst[i] == val:
            return i
    return -1

class CollatorForTLM:
    def __init__(self, tokenizer, noise_density, mean_noise_span_length, min_src_length=32):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.min_src_length = min_src_length
        self.rank = mpu.get_data_parallel_rank()
        self.world_size = mpu.get_data_parallel_world_size()
        self.count = 0

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        rng = np.random.RandomState((self.count * num_workers + worker_id) * self.world_size + self.rank)
        self.count += 1
        data = {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}

        ### denoising_loss
        input_ids = data["text"]
        batch_size, expandend_input_length = input_ids.shape
        mask_indices = np.asarray(
            [self.random_spans_noise_mask(expandend_input_length, rng) for _ in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        denoising_text = torch.tensor(self.filter_input_ids(input_ids, input_ids_sentinel), dtype=torch.long)
        denoising_target = torch.tensor(self.filter_input_ids(input_ids, labels_sentinel), dtype=torch.long)
        denoising_attention_mask = torch.ones(1, 1, 1, denoising_text.size(1), dtype=torch.long)
        denoising_loss_mask = torch.ones(batch_size, denoising_target.size(1), dtype=torch.long)

        ### lm_loss
        lm_input_ids = data["text"]
        max_span_length = lm_input_ids.shape[1]
        span_length = rng.randint(1, max_span_length - self.min_src_length)

        lm_input_ids, lm_labels = lm_input_ids[:, :-span_length], lm_input_ids[:, -span_length:]
        lm_input_ids = np.concatenate(
            [lm_input_ids, np.full((batch_size, 1), self.tokenizer.get_command('eos').Id, dtype=np.int32)], axis=-1
        )

        lm_input_ids = torch.tensor(lm_input_ids, dtype=torch.long)
        lm_labels = torch.tensor(lm_labels, dtype=torch.long)
        lm_attention_mask = torch.ones(1, 1, 1, lm_input_ids.size(1), dtype=torch.long)
        lm_loss_mask = torch.ones(batch_size, lm_labels.size(1), dtype=torch.long)

        batch = {"denoising_encoder_inputs": denoising_text,
                 "denoising_targets": denoising_target,
                 "denoising_attention_mask": denoising_attention_mask,
                 "denoising_loss_mask": denoising_loss_mask,
                     "lm_encoder_inputs": lm_input_ids,
                     "lm_targets": lm_labels,
                     "lm_attention_mask": lm_attention_mask,
                     "lm_loss_mask": lm_loss_mask
                     }
        return batch

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        input_ids = input_ids_full[input_ids_full > 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.get_command('eos').Id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def random_spans_noise_mask(self, length, rng: np.random.RandomState):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            rng: numpy random state

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            rng.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

class CollatorForT5MLM:
    def __init__(self, tokenizer, noise_density, mean_noise_span_length, prefix_lm=False, min_src_length=32):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.prefix_lm = prefix_lm
        self.min_src_length = min_src_length
        self.rank = mpu.get_data_parallel_rank()
        self.world_size = mpu.get_data_parallel_world_size()
        self.count = 0

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        rng = np.random.RandomState((self.count * num_workers + worker_id) * self.world_size + self.rank)
        self.count += 1
        data = {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        input_ids = data["text"]
        batch_size, expandend_input_length = input_ids.shape

        if self.prefix_lm:
            max_span_length = input_ids.shape[1]
            span_length = rng.randint(1, max_span_length - self.min_src_length)
            input_ids, labels = input_ids[:, :-span_length], input_ids[:, -span_length:]
            input_ids = np.concatenate(
                [input_ids, np.full((batch_size, 1), self.tokenizer.get_command('eos').Id, dtype=np.int32)], axis=-1
            )
            text = torch.tensor(input_ids, dtype=torch.long)
            target = torch.tensor(labels, dtype=torch.long)
        else:
            mask_indices = np.asarray(
                [self.random_spans_noise_mask(expandend_input_length, rng) for _ in range(batch_size)])
            labels_mask = ~mask_indices

            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

            text_ids = self.filter_input_ids(input_ids, input_ids_sentinel)
            target_ids = self.filter_input_ids(input_ids, labels_sentinel)
            if (text_ids is None) or (target_ids is None):
                return None

            text = torch.tensor(text_ids, dtype=torch.long)
            target = torch.tensor(target_ids, dtype=torch.long)

        seq_length, decoder_seq_length = text.size(1), target.size(1)
        batch = {"text": text,
                 "target": target, "attention_mask": torch.ones(1, 1, 1, seq_length, dtype=torch.long),
                 "loss_mask": torch.ones(batch_size, decoder_seq_length, dtype=torch.long)}
        return batch

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)

        ## TODO: handle some data exception
        temp = input_ids_full[input_ids_full > 0]
        if temp.shape[0] % batch_size != 0:
            return None

        input_ids = input_ids_full[input_ids_full > 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.get_command('eos').Id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def random_spans_noise_mask(self, length, rng: np.random.RandomState):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            rng: numpy random state

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            rng.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


class CollatorForGLM:
    def __init__(self, tokenizer, max_seq_length, bert_prob=1.0, gap_sentence_prob=0.0, gpt_infill_prob=0.5,
                 gpt_min_ratio=0.5, bert_ratio=0.15, gap_sentence_ratio=0.15, average_block_length=3,
                 max_block_length=40, block_mask_prob=0.0, context_mask_ratio=0.0, context_mask_range=3,
                 short_seq_prob=0.0, single_span_prob=0.0, block_position_encoding=True, encoder_decoder=False,
                 shuffle_blocks=True, sentinel_token=False, task_mask=False, masked_lm=False):
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.get_command('eos').Id
        self.cls_token = tokenizer.get_command('ENC').Id
        self.sop_token = tokenizer.get_command('sop').Id
        self.eop_token = tokenizer.get_command('eop').Id
        if block_mask_prob > 0.0:
            self.block_mask = tokenizer.get_command('dBlock').Id
        self.count = 0
        self.max_seq_length = max_seq_length
        self.rank = mpu.get_data_parallel_rank()
        self.world_size = mpu.get_data_parallel_world_size()
        assert 0.0 <= bert_prob <= 1.0
        self.bert_prob = bert_prob
        self.gap_sentence_prob = gap_sentence_prob
        self.gpt_prob = 1 - bert_prob - gap_sentence_prob
        assert self.gpt_prob >= -1e-10
        self.infill_prob = gpt_infill_prob
        self.gpt_min_ratio = gpt_min_ratio
        self.bert_ratio = bert_ratio
        self.gap_sentence_ratio = gap_sentence_ratio
        self.block_length_distribution = [poisson.pmf(i, average_block_length) for i in range(1, max_block_length)]
        self.block_mask_prob = block_mask_prob
        self.context_mask_ratio = context_mask_ratio
        self.context_mask_range = context_mask_range
        self.short_seq_prob = short_seq_prob
        self.single_span_prob = single_span_prob
        self.block_position_encoding = block_position_encoding
        self.encoder_decoder = encoder_decoder
        self.shuffle_blocks = shuffle_blocks
        self.sentinel_token = sentinel_token
        self.mask_token = self.tokenizer.get_command('MASK').Id
        self.generation_mask = 'gMASK' if task_mask else 'MASK'
        self.generation_mask = self.tokenizer.get_command(self.generation_mask).Id
        self.gap_sentence_mask = 'sMASK' if task_mask else 'MASK'
        self.gap_sentence_mask = self.tokenizer.get_command(self.gap_sentence_mask).Id
        self.masked_lm = masked_lm
        print_rank_0(
            f"BERT prob {self.bert_prob}, gap sent prob {self.gap_sentence_prob}, GPT prob {self.gpt_prob}, infill prob {self.infill_prob}")
        print_rank_0(
            f"generation min ratio {self.gpt_min_ratio}, block ratio {self.bert_ratio}, gap sent ratio {self.gap_sentence_ratio}")
        print_rank_0(f"block length distribution {self.block_length_distribution}")
        print_rank_0(f"block mask prob {self.block_mask_prob}, context mask ratio {self.context_mask_ratio}")

    def contains_sentence_end(self, tok):
        tok = self.tokenizer.IdToToken(tok)
        if '.' in tok:
            return True
        if '?' in tok:
            return True
        if '!' in tok:
            return True
        if ';' in tok:
            return True
        if ':' in tok:
            return True
        if '。' in tok:
            return True
        if '？' in tok:
            return True
        if '！' in tok:
            return True
        if '；' in tok:
            return True
        if '…' in tok:
            return True
        if '\n' in tok:
            return True
        return False

    @staticmethod
    def sample_spans(span_lengths, total_length, rng, offset=0):
        blank_length = total_length - sum(span_lengths)
        m = blank_length - len(span_lengths) + 1
        places = [rng.randrange(m + 1) for _ in range(len(span_lengths))]
        places.sort()
        spans = []
        for place, span_length in zip(places, span_lengths):
            start = offset + place
            end = offset + place + span_length
            spans.append((start, end))
            offset += span_length + 1
        return spans

    def sample_span_in_document(self, tokens, masked_lengths, rng):
        rng.shuffle(masked_lengths)
        mask_spans = []
        mask_index = 0
        indices = [-1] + np.where(tokens == self.eos_token)[0].tolist()
        last_index = len(tokens)
        documents = []
        for index in reversed(indices):
            start_index = index
            if start_index + 1 < len(tokens) and tokens[start_index + 1] == self.cls_token:
                start_index += 1
            length = last_index - start_index - 1
            if last_index == len(tokens) and length > 0:
                length -= 1
            documents.append((start_index + 1, length))
            last_index = index
        documents.sort(key=lambda x: x[1])
        for i, (offset, length) in enumerate(documents):
            if i == len(documents) - 1:
                current_masked_length, current_count = 0, 0
                while mask_index + current_count < len(masked_lengths) and masked_lengths[
                    mask_index + current_count] + current_masked_length + current_count <= length:
                    current_masked_length += masked_lengths[mask_index + current_count]
                    current_count += 1
                if current_count > 0:
                    spans = self.sample_spans(masked_lengths[mask_index: mask_index + current_count], length, rng,
                                              offset=offset)
                    mask_spans += spans
                if mask_index + current_count < len(masked_lengths) - 1:
                    print(length, masked_lengths[mask_index:], masked_lengths[:mask_index], indices)
            else:
                current_masked_total = int(length * self.bert_ratio)
                current_masked_length, current_count = 0, 0
                while mask_index + current_count < len(masked_lengths) and masked_lengths[
                    mask_index + current_count] + current_masked_length <= current_masked_total:
                    current_masked_length += masked_lengths[mask_index + current_count]
                    current_count += 1
                if current_count > 0:
                    spans = self.sample_spans(masked_lengths[mask_index:mask_index + current_count], length,
                                              rng, offset=offset)
                    mask_spans += spans
                    mask_index += current_count
        return mask_spans

    def make_masked_data(self, tokens, loss_masks, block_spans, rng, task='bert'):
        position_ids = np.arange(len(tokens), dtype=np.long)
        targets = copy.deepcopy(tokens)
        mask_id = self.mask_token
        mlm_masks = np.zeros(len(tokens), dtype=np.long)
        for start, end in block_spans:
            for idx in range(start, end):
                tokens[idx] = mask_id
            mlm_masks[start: end] = 1
        loss_masks = loss_masks * mlm_masks
        return tokens, targets, loss_masks, position_ids, len(tokens)

    def make_block_data(self, tokens, loss_masks, block_spans, rng, task='bert'):
        text_length = len(tokens)
        position_ids = np.ones(len(tokens), dtype=np.long)
        for start, end in block_spans:
            position_ids[start + 1: end] = 0
        position_ids = np.cumsum(position_ids) - 1
        if self.encoder_decoder or not self.shuffle_blocks:
            block_spans.sort(key=lambda x: x[0])
        else:
            rng.shuffle(block_spans)
        if self.sentinel_token:
            block_spans = [(start, end, idx) for idx, (start, end) in enumerate(block_spans)]
        else:
            block_spans = [(start, end, 0) for start, end in block_spans]
        target_tokens, target_position_ids, target_block_position_ids, targets = [], [], [], []
        for start, end, idx in block_spans:
            sop_token = 'sop' if idx == 0 else f"sop{idx}"
            target_tokens.append([self.tokenizer.get_command(sop_token).Id])
            span_tokens = copy.deepcopy(tokens[start: end])
            if self.block_mask_prob > 0.0 and task == 'bert':
                for sub_idx in range(len(span_tokens)):
                    if random.random() < self.block_mask_prob:
                        span_tokens[sub_idx] = self.block_mask
            target_tokens.append(span_tokens)
            targets.append(tokens[start: end])
            targets.append([self.eop_token])
            if not self.sentinel_token:
                target_position_id = position_ids[start: end]
                target_position_ids.append(target_position_id)
                target_position_ids.append([target_position_id[0]])
            else:
                target_position_ids.append([self.max_seq_length] * (end - start + 1))
            if self.block_position_encoding:
                target_block_position_ids.append(np.arange(1, end - start + 2, dtype=np.long))
            else:
                target_block_position_ids.append([1] * (end - start + 1))
        block_spans.sort(key=lambda x: x[0])
        source_tokens, source_position_ids, local_spans = [], [], []
        last, current_length = 0, 0
        for start, end, idx in block_spans:
            if task == 'generation':
                mask_id = self.generation_mask
            elif task == 'gap_sentence':
                mask_id = self.gap_sentence_mask
            else:
                mask_token = 'MASK' if idx == 0 else f'MASK{idx}'
                mask_id = self.tokenizer.get_command(mask_token).Id
            local_spans.append((current_length, current_length + start - last))
            source_tokens.append(tokens[last: start])
            source_tokens.append([mask_id])
            source_position_ids.append(position_ids[last: start])
            source_position_ids.append([position_ids[start]])
            current_length += start - last + 1
            last = end
        if last < len(tokens):
            local_spans.append((current_length, current_length + len(tokens) - last))
            source_tokens.append(tokens[last:])
            source_position_ids.append(position_ids[last:])
        source_length = sum(map(len, source_tokens))
        if target_tokens and self.eos_token in np.concatenate(target_tokens).tolist():
            print("Found EOS in target", self.tokenizer.DecodeIds(tokens))
            raise RuntimeError
        if self.encoder_decoder:
            target_tokens = target_tokens + [self.eop_token]
            loss_masks = np.ones(len(target_tokens), dtype=np.long)
            return source_tokens, target_tokens, loss_masks
        else:
            tokens = np.concatenate(source_tokens + target_tokens)
            if task == 'bert' and self.context_mask_ratio > 0:
                mask_candidates = set()
                for start, end in local_spans:
                    if start != 0:
                        local_end = min(end, start + self.context_mask_range)
                        mask_candidates.update(range(start, local_end))
                    if end != 0:
                        local_start = max(start, end - self.context_mask_range)
                        mask_candidates.update(range(local_start, end))
                mask_pos = rng.sample(mask_candidates, int(self.context_mask_ratio * text_length))
                for pos in mask_pos:
                    tokens[pos] = self.block_mask
            targets = np.concatenate(source_tokens + targets)
            loss_masks = np.ones(len(tokens), dtype=np.long)
            loss_masks[:source_length] = 0
            position_ids = np.concatenate(source_position_ids + target_position_ids)
            block_position_ids = np.concatenate(
                [np.zeros(source_length, dtype=np.long)] + target_block_position_ids)
            position_ids = np.stack([position_ids, block_position_ids], axis=0)
            return tokens, targets, loss_masks, position_ids, source_length

    def generate_blank_data(self, sample, masked_lengths, rng, task='bert'):
        rng.shuffle(masked_lengths)
        tokens, loss_masks = sample['text'], sample['loss_mask']
        assert tokens[0] == self.cls_token
        block_spans = self.sample_span_in_document(tokens, masked_lengths, rng)
        if len(block_spans) < len(masked_lengths):
            return None
        if self.masked_lm:
            data = self.make_masked_data(tokens, loss_masks, block_spans, rng)
        else:
            data = self.make_block_data(tokens, loss_masks, block_spans, rng, task=task)
        return data

    def split_samples(self, samples, rng):
        target_length = rng.randrange(32, self.max_seq_length - 1)
        num_splits = (self.max_seq_length - 1) // target_length
        new_samples = []
        cls_id = self.cls_token
        for sample in samples:
            tokens, loss_masks = sample['text'][1:], sample['loss_mask'][1:]
            for _ in range(num_splits):
                if target_length >= len(tokens):
                    new_tokens, new_loss_masks = tokens, loss_masks
                else:
                    random_start = rng.randrange(0, len(tokens) - target_length)
                    while random_start > 0 and (tokens[random_start] == self.eos_token or not (
                            self.contains_sentence_end(tokens[random_start - 1]) or tokens[
                        random_start - 1] == self.eos_token)):
                        random_start -= 1
                    random_end = random_start + target_length
                    while random_end > random_start and not (
                            self.contains_sentence_end(tokens[random_end - 1]) or tokens[
                        random_end - 1] == self.eos_token):
                        random_end -= 1
                    if random_end - random_start < target_length // 2:
                        random_end = random_start + target_length
                    new_tokens, new_loss_masks = tokens[random_start: random_end], loss_masks[random_start: random_end]
                new_tokens = np.concatenate(([cls_id], new_tokens))
                new_loss_masks = np.concatenate(([0], new_loss_masks))
                new_samples.append({'text': new_tokens, 'loss_mask': new_loss_masks})
        return new_samples

    def __call__(self, samples):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        rng = random.Random((self.count * num_workers + worker_id) * self.world_size + self.rank)
        self.count += 1
        token_batch, target_batch, loss_mask_batch, position_id_batch = [], [], [], []
        source_batch, target_batch = [], []
        if rng.random() < self.short_seq_prob:
            samples = self.split_samples(samples, rng)
        rand = rng.random()
        single_span = rand < self.single_span_prob
        rand = 0.0 if single_span else rng.random()
        attention_mask = []
        if rand < self.bert_prob:
            mode = 'bert'
            for sample in samples:
                if single_span:
                    masked_lengths = [rng.choices(range(1, len(self.block_length_distribution) + 1),
                                                  weights=self.block_length_distribution)[0]]
                else:
                    masked_lengths, masked_count = [], 0
                    while masked_count < int(self.bert_ratio * len(sample['text'])):
                        block_length = rng.choices(range(1, len(self.block_length_distribution) + 1),
                                                   weights=self.block_length_distribution)[0]
                        masked_lengths.append(block_length)
                        masked_count += block_length
                data = self.generate_blank_data(sample, masked_lengths, rng, task='bert')
                if data is not None:
                    if self.encoder_decoder:
                        source_tokens, target_tokens, loss_masks, sep = data
                        source_batch.append(source_tokens)
                        target_batch.append(target_tokens)
                    else:
                        tokens, targets, loss_masks, position_ids, sep = data
                        token_batch.append(tokens)
                        target_batch.append(targets)
                        position_id_batch.append(position_ids)
                    loss_mask_batch.append(loss_masks)
                    attention_mask.append(sep)
        elif rand < self.bert_prob + self.gap_sentence_prob:
            mode = 'sentence'
            for sample in samples:
                tokens, loss_masks = sample['text'], sample['loss_mask']
                sentence_spans = []
                last_index = 1 if tokens[0] == self.cls_token else 0
                for i in range(len(tokens)):
                    if self.contains_sentence_end(tokens[i]):
                        if last_index < i + 1:
                            sentence_spans.append((last_index, i + 1))
                        last_index = i + 1
                    elif tokens[i] == self.eos_token:
                        last_index = i + 1
                if last_index < len(tokens):
                    sentence_spans.append((last_index, len(tokens)))
                if not sentence_spans and torch.distributed.get_rank() == 0:
                    try:
                        print(self.tokenizer.DecodeIds(tokens[1:].tolist()))
                    except IndexError:
                        print(tokens[1:])
                rng.shuffle(sentence_spans)
                block_spans, block_length = [], 0
                for start, end in sentence_spans:
                    block_spans.append((start, end))
                    block_length += end - start
                    if block_length >= int(self.gap_sentence_ratio * len(tokens)):
                        break
                data = self.make_block_data(tokens, loss_masks, block_spans, rng, task='gap_sentence')
                tokens, targets, loss_masks, position_ids, sep = data
                token_batch.append(tokens)
                target_batch.append(targets)
                loss_mask_batch.append(loss_masks)
                position_id_batch.append(position_ids)
                attention_mask.append(sep)
        else:
            mode = 'gpt'
            text_lens = list(map(lambda x: len(x['text']), samples))
            max_generation_length = rng.randint(int(self.gpt_min_ratio * min(text_lens)), max(text_lens) - 2)
            for sample in samples:
                generation_length = min(max_generation_length, len(sample['text']) - 2)
                multiple_doc = index_in_list(sample['text'], self.eos_token) not in [-1, len(
                    sample['text']) - 1]
                if multiple_doc or rng.random() < self.infill_prob:
                    division = len(sample['text']) - generation_length
                    tokens, loss_masks = sample['text'], sample['loss_mask']
                    source_tokens, target_tokens = tokens[:division], tokens[division:]
                    target_masks = loss_masks[division:]
                    tokens = np.concatenate((
                        source_tokens, [self.generation_mask, self.sop_token],
                        target_tokens[:-1]))
                    targets = np.concatenate((source_tokens, [self.generation_mask], target_tokens))
                    loss_masks = np.concatenate((np.zeros(len(source_tokens) + 1, dtype=np.long), target_masks))
                    token_batch.append(tokens)
                    target_batch.append(targets)
                    loss_mask_batch.append(loss_masks)
                    attention_mask.append(len(sample['text']) - generation_length + 1)
                    position_ids = np.arange(len(source_tokens) + len(target_tokens) + 1, dtype=np.long)
                    position_ids[len(source_tokens) + 1:] = len(source_tokens)
                    if self.block_position_encoding:
                        block_position_ids = np.concatenate(
                            (np.zeros(len(source_tokens), dtype=np.long),
                             np.arange(len(target_tokens) + 1, dtype=np.long)))
                    else:
                        block_position_ids = np.concatenate((np.zeros(len(source_tokens) + 1, dtype=np.long),
                                                             np.ones(len(target_tokens) + 1, dtype=np.long)))
                    position_id_batch.append(np.stack([position_ids, block_position_ids], axis=0))
                else:
                    tokens, targets, loss_masks, position_ids, sep = self.generate_blank_data(sample,
                                                                                              [generation_length], rng,
                                                                                              task='generation')
                    token_batch.append(tokens)
                    target_batch.append(targets)
                    loss_mask_batch.append(loss_masks)
                    position_id_batch.append(position_ids)
                    attention_mask.append(sep)
                    if tokens is None:
                        print(sample, generation_length, multiple_doc)
        if self.encoder_decoder:
            return {
                'text': torch.tensor(np.array(source_batch), dtype=torch.long),
                'target': torch.tensor(np.array(target_batch), dtype=torch.long),
                'loss_mask': torch.tensor(np.array(loss_mask_batch), dtype=torch.long)}
        else:
            token_batch, target_batch, loss_mask_batch, position_id_batch = self.pad_batch(token_batch, target_batch,
                                                                                           loss_mask_batch,
                                                                                           position_id_batch)
            return {'text': torch.tensor(np.array(token_batch), dtype=torch.long),
                    'target': torch.tensor(np.array(target_batch), dtype=torch.long),
                    'loss_mask': torch.tensor(np.array(loss_mask_batch), dtype=torch.long),
                    'position_id': torch.tensor(np.array(position_id_batch), dtype=torch.long),
                    'attention_mask': torch.tensor(np.array(attention_mask), dtype=torch.long),
                    'mode': mode}

    @staticmethod
    def pad_batch(token_batch, target_batch, loss_mask_batch, position_id_batch):
        seq_lengths = list(map(len, token_batch))
        if seq_lengths.count(seq_lengths[0]) != len(seq_lengths):
            max_length = max(seq_lengths)
            token_batch = [np.concatenate((tokens, np.zeros(max_length - len(tokens), dtype=np.long))) for tokens in
                           token_batch]
            target_batch = [np.concatenate((targets, np.zeros(max_length - len(targets), dtype=np.long))) for
                            targets in
                            target_batch]
            loss_mask_batch = [np.concatenate((loss_masks, np.zeros(max_length - len(loss_masks), dtype=np.long)))
                               for loss_masks in loss_mask_batch]
            position_id_batch = [
                np.concatenate((position_ids, np.zeros((2, max_length - position_ids.shape[1]), dtype=np.long)),
                               axis=1) for position_ids in position_id_batch]
        return token_batch, target_batch, loss_mask_batch, position_id_batch


# conventional transformer
def build_mask_matrix(separator, batch_size, seq_length, memory_length=0):
    dtype = torch.float
    m = torch.ones((1, seq_length, seq_length), dtype=dtype, device=separator.device)
    m = torch.tril(m)
    is_scalar = torch.numel(separator) == 1
    if is_scalar:
        m[0, :, :separator] = 1
    else:
        m = m.expand(batch_size, -1, -1)
        ids = torch.arange(seq_length, device=separator.device, dtype=separator.dtype).view(1, -1)
        mask = ids < separator.view(-1, 1)
        m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
    if memory_length > 0:
        m = m.expand(batch_size, -1, -1)
        m = torch.cat((torch.ones((batch_size, seq_length, memory_length), dtype=dtype), m), dim=2)
    m = m.unsqueeze(1)
    return m
