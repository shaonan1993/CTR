import json
import os
import argparse
import random
from datetime import datetime

import numpy as np
import torch.distributed
import torch

from arguments import get_args
from configure_data import make_tokenizer

from utils import get_checkpoint_iteration, get_checkpoint_name, print_rank_0, get_spare_port
from SwissArmyTransformer.training.deepspeed_training import initialize_distributed, set_random_seed, get_model
from SwissArmyTransformer import mpu

from model.T5_with_VAEmixin import T5VAEMixIn
from utils import load_pretrained, tqdm_rank_0
from configure_data import make_data_loader
from torch.utils.data import Dataset

from datasets import load_dataset, concatenate_datasets, Value
from templates import DatasetTemplates
from collections import Counter
import math

# T0 test tasks
TEST_TASK_NAME = [
    "super_glue/wsc.fixed",
    "super_glue/wic",
    "super_glue/copa",
    "story_cloze/2016",
    "super_glue/cb",
    "super_glue/rte",
    "hellaswag",
    "anli/r1",
    "anli/r2",
    "anli/r3",
    "winogrande/winogrande_xl",
]

# T0 sub train tasks
TRAIN_TASK_NAME = [
    'commonsense_qa',
    'cosmos_qa',
    'cos_e/v1.11',
    'qasc',
    'super_glue/boolq',
    'rotten_tomatoes',
    'ag_news',
    'glue/mrpc',
    'glue/qqp']


def build_ga_dataset(args, dataset_name, dataset_config_name, raw_datasets, distribution='uniform'):
    task_name = f'{dataset_name}/{dataset_config_name}' if dataset_config_name else f'{dataset_name}'
    dataset_distribution = {'anli/r1': {0: 334, 2: 333, 1: 333},
                            'anli/r2': {0: 334, 1: 333, 2: 333},
                            'anli/r3': {0: 402, 1: 402, 2: 396},
                            'super_glue/cb': {1: 28, 0: 23, 2: 5},
                            'super_glue/rte': {0: 146, 1: 131},
                            'super_glue/wsc.fixed': {0: 66, 1: 38},
                            'winogrande/winogrande_xl': {'2': 639, '1': 628},
                            'super_glue/copa': {0: 55, 1: 45},
                            'hellaswag': {'2': 2584, '0': 2515, '1': 2485, '3': 2458},
                            'super_glue/wic': {0: 319, 1: 319},
                            'story_cloze/2016': {1: 962, 2: 909}
                            }

    label_key = 'label'
    if dataset_name in ['winogrande']:
        label_key = 'answer'
    if dataset_name in ['story_cloze']:
        label_key = 'answer_right_ending'
    filtered_dataset = raw_datasets

    if dataset_name == 'anli':
        print(f'len of raw_dataset: {filtered_dataset}')
        filtered_dataset = filtered_dataset.filter(lambda x: len(x['reason']) > 0, load_from_cache_file=False)
        print(f'len of filtered_dataset: {filtered_dataset}')

    if dataset_name == 'winogrande':
        train_file_path = './data/T0_dataset/winogrande_winogrande_debiased'
        data_files = {
            'train': os.path.join(train_file_path, 'train.json')
        }
        filtered_dataset = load_dataset('json', data_files=data_files)['train']

    label_list = filtered_dataset[label_key]
    label_type_set = set(label_list)
    print(f'label_type_set: {label_type_set}')
    ga_dataset_list = []

    for label_type in label_type_set:
        single_label_dataset = filtered_dataset.filter(lambda x: x[label_key] == label_type, load_from_cache_file=False)
        single_label_dataset = single_label_dataset.shuffle(seed=args.fewshot_shuffle_seed, load_from_cache_file=False)

        if distribution == 'ratio':
            example_num_per_label = math.ceil(
                dataset_distribution[task_name][label_type] / sum(
                    dataset_distribution[task_name].values()) * args.shot_num)
        else:
            example_num_per_label = math.ceil(args.shot_num / len(label_type_set))

        ga_dataset_list.append(
            single_label_dataset.select(range(min(example_num_per_label, len(single_label_dataset)))))

    filtered_dataset = concatenate_datasets(ga_dataset_list)

    return filtered_dataset


def special_for_duplicat(dataset_config_name, raw_dataset):
    """remove duplicate examples"""
    context_key = 'inputs_pretokenized'

    counter = {}
    for example in raw_dataset:
        if example[context_key] in counter:
            counter[example[context_key]] += 1
        else:
            counter[example[context_key]] = 1

    choice_count = {}
    for key, value in counter.items():
        if value not in choice_count:
            choice_count[value] = 0
        choice_count[value] += 1

    valid_choice_num = sorted(choice_count.items(), key=lambda x: x[1], reverse=True)[0][0]

    valid_example_set = set()
    for key, value in counter.items():
        if value == valid_choice_num:
            valid_example_set.add(key)

    raw_dataset = raw_dataset.filter(lambda x: x[context_key] in valid_example_set, load_from_cache_file=False)

    return raw_dataset


def get_batch(batch, args):
    datatype = torch.int64
    keys = ["input_ids", "attention_mask", "candidate_token_ids", "candidate_attention_mask", "labels", "task_id"]

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, batch, datatype)

    task_id = data_b['task_id'].long()
    enc_input_ids = data_b["input_ids"].long()  # [batch_size, max_seq_len]
    enc_attention_mask = data_b["attention_mask"].long()  # [batch_size, 1, 1 max_seq_len]
    dec_target_ids = data_b["candidate_token_ids"].long()  # [batch_size, num_choices, max_tgt_len]
    dec_logit_mask = data_b['candidate_attention_mask'].long()  # [batch_size, num_choices, max_tgt_len]
    labels = data_b["labels"].long()

    decoder_input_ids = torch.ones((dec_target_ids.size()[0], dec_target_ids.size()[1], 1), dtype=torch.long).cuda() * \
                        args.decoder_start_token_id
    decoder_input_ids = torch.cat((decoder_input_ids, dec_target_ids[:, :, :-1]), dim=2)

    # Convert
    if args.fp16:
        enc_attention_mask = enc_attention_mask.half()
    elif args.bf16:
        enc_attention_mask = enc_attention_mask.bfloat16()

    return enc_input_ids, enc_attention_mask, decoder_input_ids, dec_target_ids, dec_logit_mask, labels, task_id


def decoder_shift_right(input_ids, args):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = args.decoder_start_token_id
    return shifted_input_ids


def get_loss_batch(data, args):
    keys = ['input_ids', 'loss_mask', 'target', 'attention_mask', 'task_id']
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['input_ids'].long()
    task_id = data_b['task_id'].long()
    # task_emb = data_b['task_emb'].float()
    labels = data_b['target'].long()
    decoder_tokens = decoder_shift_right(labels, args)  # decoder_tokens就是带偏移的target_id
    attention_mask = data_b['attention_mask'].long()
    loss_mask = data_b['loss_mask'].float()

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
    elif args.bf16:
        attention_mask = attention_mask.bfloat16()
    return tokens, decoder_tokens, labels, loss_mask, attention_mask, task_id


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)


def generate_single_dataset(task_name, args, tokenizer, dataset, task_id):
    input_ids = {}
    attention_mask = {}
    choices = {}
    choices_attention_mask = {}
    labels = {}
    pad_id = tokenizer.get_command('pad').Id

    eos_id = tokenizer.get_command('eos').Id

    for data_example in dataset:
        source_text = data_example["inputs_pretokenized"]
        target_text = data_example["targets_pretokenized"]
        is_correct = bool(data_example["is_correct"])

        # tokenize
        source_tokenized_inputs = \
            tokenizer.EncodeAsIds(source_text).tokenization[:(args.multi_src_seq_length - 1)] + [eos_id]
        source_attention_mask = [1] * len(source_tokenized_inputs)
        if len(source_tokenized_inputs) < args.multi_src_seq_length:
            source_pad_length = args.multi_src_seq_length - len(source_tokenized_inputs)
            source_tokenized_inputs = source_tokenized_inputs + [pad_id] * source_pad_length
            source_attention_mask = source_attention_mask + [0] * source_pad_length

        target_tokenized = tokenizer.EncodeAsIds(target_text).tokenization[:(args.multi_tgt_seq_length - 1)] + [eos_id]
        target_attention_mask = [1] * len(target_tokenized)
        if len(target_tokenized) < args.multi_tgt_seq_length:
            target_pad_length = args.multi_tgt_seq_length - len(target_tokenized)
            target_tokenized = target_tokenized + [pad_id] * target_pad_length
            target_attention_mask = target_attention_mask + [0] * target_pad_length

        if source_text not in input_ids:
            input_ids[source_text] = source_tokenized_inputs
            attention_mask[source_text] = source_attention_mask

        if source_text not in choices:
            choices[source_text] = []
            labels[source_text] = []
            choices_attention_mask[source_text] = []
        choices[source_text].append(target_tokenized)
        choices_attention_mask[source_text].append(target_attention_mask)

        if is_correct:
            labels[source_text].append(1)
        else:
            labels[source_text].append(0)

    keys = list(input_ids.keys())
    all_input_ids = torch.tensor([input_ids[key] for key in keys], dtype=torch.long)  # [data_num, max_source_seq_len]
    all_attention_mask = torch.tensor([attention_mask[key] for key in keys], dtype=torch.long).unsqueeze(1).unsqueeze(1)
    # [data_num, 1, 1, max_source_seq_len]
    all_labels = torch.tensor([labels[key] for key in keys], dtype=torch.long)  # [data_num, num_choices]
    feature_dict = {'input_ids': all_input_ids, 'attention_mask': all_attention_mask, "labels": all_labels}

    ### Check data choices
    all_choices = [choices[key] for key in keys]  # [data_num, num_choices, max_tgt_len]
    all_choices_attention = [choices_attention_mask[key] for key in keys]  # [data_num, num_choices, max_tgt_len]

    num_choices = 0
    for idx, item in enumerate(all_choices):
        if idx == 0:
            num_choices = len(item)
        assert len(item) == num_choices, "Different number of choices for the same dataset?"

    feature_dict['candidate_token_ids'] = torch.tensor(all_choices, dtype=torch.long)
    feature_dict['candidate_attention_mask'] = torch.tensor(all_choices_attention, dtype=torch.long)

    feature_dict["ids"] = torch.arange(len(all_choices), dtype=torch.long)

    print_rank_0(f'{task_name} use task id: {task_id}, task name: {args.t0_task_names[task_id]}')
    feature_dict['task_id'] = torch.tensor([task_id] * all_input_ids.shape[0], dtype=torch.long)

    return DictDataset(**feature_dict)


def generate_loss_dataset(task_name, args, tokenizer, dataset, task_id):
    pad_id = tokenizer.get_command('pad').Id

    eos_id = tokenizer.get_command('eos').Id

    input_ids_list = []
    attention_mask_list = []
    choices_ids_list = []
    choices_attention_mask_list = []
    for data_example in dataset:
        source_text = data_example["inputs_pretokenized"]
        target_text = data_example["targets_pretokenized"]
        is_correct = bool(data_example["is_correct"])

        # only use target
        if not is_correct:
            continue

        # tokenize
        source_tokenized_inputs = \
            tokenizer.EncodeAsIds(source_text).tokenization[:(args.multi_src_seq_length - 1)] + [eos_id]
        source_attention_mask = [1] * len(source_tokenized_inputs)
        if len(source_tokenized_inputs) < args.multi_src_seq_length:
            source_pad_length = args.multi_src_seq_length - len(source_tokenized_inputs)
            source_tokenized_inputs = source_tokenized_inputs + [pad_id] * source_pad_length
            source_attention_mask = source_attention_mask + [0] * source_pad_length

        target_tokenized = tokenizer.EncodeAsIds(target_text).tokenization[:(args.multi_tgt_seq_length - 1)] + [eos_id]
        target_attention_mask = [1] * len(target_tokenized)
        if len(target_tokenized) < args.multi_tgt_seq_length:
            target_pad_length = args.multi_tgt_seq_length - len(target_tokenized)
            target_tokenized = target_tokenized + [pad_id] * target_pad_length
            target_attention_mask = target_attention_mask + [0] * target_pad_length

        input_ids_list.append(source_tokenized_inputs)
        attention_mask_list.append(source_attention_mask)

        choices_ids_list.append(target_tokenized)
        choices_attention_mask_list.append(target_attention_mask)

    all_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    all_attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).unsqueeze(1).unsqueeze(1)
    feature_dict = {'input_ids': all_input_ids, 'attention_mask': all_attention_mask}

    feature_dict['target'] = torch.tensor(choices_ids_list, dtype=torch.long)
    feature_dict['loss_mask'] = torch.tensor(choices_attention_mask_list, dtype=torch.long)

    print_rank_0(f'{task_name} use task id: {task_id}, task name: {args.t0_task_names[task_id]}')
    feature_dict['task_id'] = torch.tensor([task_id] * all_input_ids.shape[0], dtype=torch.long)

    return DictDataset(**feature_dict)


def forward_step(model, enc_input_ids, enc_attention_mask, decoder_input_ids, dec_target_ids, dec_logit_mask,
                 length_penalty=0.0, take_softmax=True, task_id=0):
    cur_batch_size, seq_length = enc_input_ids.size()[:2]

    prompt_num_per_task = args.prompt_num_per_task

    more_attention_mask = torch.ones(cur_batch_size, 1, 1, prompt_num_per_task,
                                     device=enc_input_ids.device).type_as(enc_input_ids)
    enc_attention_mask = torch.cat((more_attention_mask, enc_attention_mask), dim=-1)

    encoder_outputs = model.encode(enc_input_ids, enc_attention_mask, task_id=task_id)

    hidden_size = encoder_outputs.size()[-1]
    num_choices, tgt_seq_length = decoder_input_ids.size()[1:]
    encoder_outputs = encoder_outputs.unsqueeze(1).expand(-1, num_choices, -1, -1)
    enc_attention_mask = enc_attention_mask.unsqueeze(1).expand(-1, num_choices, 1, 1, seq_length + prompt_num_per_task)
    encoder_outputs = encoder_outputs.reshape(cur_batch_size * num_choices, seq_length + prompt_num_per_task,
                                              hidden_size)
    enc_attention_mask = enc_attention_mask.reshape(cur_batch_size * num_choices, 1, 1,
                                                    seq_length + prompt_num_per_task)
    decoder_input_ids = decoder_input_ids.reshape(-1, tgt_seq_length)
    outputs, *_ = model.decode(decoder_input_ids, None, encoder_outputs=encoder_outputs,
                               cross_attention_mask=enc_attention_mask, task_id=task_id)
    outputs = outputs.float()

    # if take_softmax:
    #     outputs = torch.nn.functional.log_softmax(outputs, dim=-1)

    dec_target_ids = dec_target_ids.reshape(-1, dec_target_ids.size(-1))
    dec_logit_mask = dec_logit_mask.reshape(-1, dec_logit_mask.size(-1))
    """
    batch_ids = torch.arange(dec_target_ids.size(0), dtype=torch.long, device=dec_target_ids.device)
    batch_ids = batch_ids.unsqueeze(1).expand_as(dec_target_ids)
    seq_ids = torch.arange(dec_target_ids.size(-1), dtype=torch.long, device=dec_target_ids.device)
    seq_ids = seq_ids.unsqueeze(0).expand_as(dec_target_ids)
    logits = outputs[batch_ids, seq_ids, dec_target_ids]
    """
    logits = -mpu.vocab_parallel_cross_entropy(outputs, dec_target_ids)
    logits = (logits * dec_logit_mask).sum(dim=1)
    if length_penalty > 0.0:
        logits = logits / dec_logit_mask.sum(dim=1) ** length_penalty
    logits = logits.view(cur_batch_size, num_choices)
    return logits


def forward_step_for_loss(model, enc_input_ids, enc_attention_mask, decoder_input_ids, dec_target_ids, dec_logit_mask,
                          length_penalty=0.0, task_id=0):
    _, logits, vae_loss, *_ = model(enc_input_ids=enc_input_ids, dec_input_ids=decoder_input_ids,
                                    enc_attention_mask=enc_attention_mask, task_id=task_id)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), dec_target_ids)
    loss_mask = dec_logit_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask)
    return loss


def main(args):
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    if args.eval_task_set == 'train':
        CUR_EVAL_TASK_NAME = TRAIN_TASK_NAME
    elif args.eval_task_set == 'test':
        CUR_EVAL_TASK_NAME = TEST_TASK_NAME
    else:
        raise NotImplementedError(f'Invalid eval task set: {args.eval_task_set}')

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    load_dir, tag, _, _ = get_checkpoint_iteration(args.load)

    # Make tokenizer
    tokenizer = make_tokenizer(args)
    args.decoder_start_token_id = tokenizer.get_command('sop').Id

    # Load Model
    model_cls = T5VAEMixIn
    model = get_model(args, model_cls)
    if hasattr(model, "module"):
        module = model.module
    else:
        module = model
    load_pretrained(module, args.load, args)  # Load model
    torch.distributed.barrier()

    ### T0 test data Stuff
    world_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    eval_batch_size = args.batch_size * world_size
    if args.eval_batch_size is not None:
        eval_batch_size = args.eval_batch_size * world_size

    save_dir = os.path.join("eval_results", args.experiment_name)
    if torch.distributed.get_rank() == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    t0_combined_raw_data_dir = f'./data/t0_combined_raw_data_no_prompt/t0_combined_raw_data_8node'
    t0_task_names = json.load(open(os.path.join(t0_combined_raw_data_dir, 't0_task_names.json')))
    args.t0_task_names = t0_task_names

    task_name_to_id = dict()
    task_id_to_name = dict()
    for i, task_name in enumerate(t0_task_names):
        task_name_to_id[task_name] = i
        task_id_to_name[i] = task_name  # task_id -> task_name
    args.task_id_to_name = task_id_to_name
    args.task_num = len(task_name_to_id)

    print_rank_0(f'len of task_id_to_name: {len(task_id_to_name)}')

    def preprocess_function(examples, idx):
        bs = len(examples[column_names[0]])
        assert len(idx) == bs  # idx range from 0 to 999

        input_texts = []
        target_texts = []
        answer_choices_texts = []

        for i in range(bs):
            ex = {
                k: examples[k][i]
                for k in column_names
            }

            input, target = template.apply(ex)

            ex_answer_choices = template.get_answer_choices_list(ex)
            assert target in ex_answer_choices, f'task: {uniq_task_name}, target: {target}, ex_answer_choices: {ex_answer_choices}'
            input_texts.append(input)
            target_texts.append(target)  # correct answer
            answer_choices_texts.append(ex_answer_choices)  # candidate choices

        # check choice num
        choice_num = set([len(current_choice) for current_choice in answer_choices_texts])
        if len(choice_num) != 1:
            print_rank_0(f'ERROR: choice num: {choice_num}')

        features = {
            'idx': [],
            'inputs_pretokenized': [],
            'targets_pretokenized': [],
            'is_correct': [],
            'weight': []
        }

        for batch_idx, total_idx in zip(range(bs), idx):
            current_answer = target_texts[batch_idx]
            current_is_correct = [choice == current_answer for choice in answer_choices_texts[batch_idx]]

            if sum(current_is_correct) != 1:
                print_rank_0('Warning!!!')
                print_rank_0(f'each example have and only have one correct answer, inputs: {input_texts[batch_idx]}, \n' \
                             f'candidate: {answer_choices_texts[batch_idx]} \n' \
                             f'answer: {current_answer}')
                continue
            features['is_correct'].extend(current_is_correct)

            current_idx = [[total_idx, i] for i in range(len(answer_choices_texts[batch_idx]))]
            features['idx'].extend(current_idx)
            features['inputs_pretokenized'].extend([input_texts[batch_idx]] *
                                                   len(answer_choices_texts[batch_idx]))
            features['targets_pretokenized'].extend(answer_choices_texts[batch_idx])

            features['weight'].extend([1.0] * len(answer_choices_texts[batch_idx]))

        return features

    model.eval()
    # traverse each task
    all_best_results = {}
    for task_name in CUR_EVAL_TASK_NAME:
        task_tuple = task_name.split('/')
        task_name = task_name.replace("/", "_")

        if len(task_tuple) == 2:
            dataset_name, dataset_config_name = task_tuple[0], task_tuple[1]
        else:
            dataset_name, dataset_config_name = task_tuple[0], None

        train_file_path = './data/T0_dataset'
        if dataset_config_name:
            train_file_path = os.path.join(train_file_path, f'{dataset_name}_{dataset_config_name}')
        else:
            train_file_path = os.path.join(train_file_path, dataset_name)

        if dataset_name == 'story_cloze':
            data_files = {
                'validation': os.path.join(train_file_path, 'validation.json')
            }
            raw_datasets = load_dataset('json', data_files=data_files)['validation']

            new_feature = raw_datasets.features.copy()
            new_feature['answer_right_ending'] = Value('int64', id=None)
            raw_datasets = raw_datasets.cast(new_feature, load_from_cache_file=False)
        else:
            data_files = {
                'train': os.path.join(train_file_path, 'train.json')
            }
            raw_datasets = load_dataset('json', data_files=data_files)['train']

        column_names = raw_datasets.column_names

        print_rank_0(f'use template_dir: {args.template_dir}')
        prompts = DatasetTemplates(
            f"{dataset_name}" if dataset_config_name is None else f"{dataset_name}/{dataset_config_name}",
            template_dir=args.template_dir)

        template_list = prompts.templates.keys()
        print_rank_0(f'{dataset_name} contains templates: {template_list}')

        # 遍历每个被prompt封装过的task
        for template_id in template_list:
            template = prompts.templates[template_id]
            template_name = template.name

            uniq_task_name = f'{dataset_name}_{dataset_config_name}' if dataset_config_name else dataset_name
            norm_template_name = template_name.replace('/', '_')
            norm_template_name = norm_template_name.replace(' ', '_')
            uniq_task_name = f'{uniq_task_name}_{norm_template_name}'
            # ga_task_names.append(uniq_task_name)

            if template.metadata.original_task is not True:
                print_rank_0(f'skip {template_name}, not original task form')
                continue

            # some template can only apply to a part of example
            filtered_dataset = None
            if dataset_config_name == 'copa':
                if template_name in ["\u2026What could happen next, C1 or C2?", "\u2026As a result, C1 or C2?"]:
                    filtered_dataset = raw_datasets.filter(lambda example: example['question'] == 'effect', load_from_cache_file=False)
                if template_name in ["\u2026which may be caused by", "\u2026why? C1 or C2"]:
                    filtered_dataset = raw_datasets.filter(lambda example: example['question'] == 'cause', load_from_cache_file=False)

            if not filtered_dataset:
                filtered_dataset = raw_datasets

            filtered_dataset = build_ga_dataset(args, dataset_name, dataset_config_name, filtered_dataset,
                                                args.ga_dev_distribution)

            label_key = 'label'
            if dataset_name in ['winogrande']:
                label_key = 'answer'
            if dataset_name in ['story_cloze']:
                label_key = 'answer_right_ending'

            print(f'filtered_dataset: {filtered_dataset}')
            print(f'label distribution: {Counter(filtered_dataset[label_key])}')

            processed_dataset = filtered_dataset.map(
                preprocess_function, batched=True, with_indices=True, remove_columns=column_names,
                load_from_cache_file=False)

            for cur_task_id in range(318):
                # test_dataset = generate_single_dataset(uniq_task_name, args, tokenizer, processed_dataset, cur_task_id)
                test_dataset = generate_loss_dataset(uniq_task_name, args, tokenizer, processed_dataset, cur_task_id)

                dataloader = make_data_loader(test_dataset, tokenizer, batch_size=eval_batch_size, args=args,
                                              num_iters=None, shuffle=False, collator=None, keep_last=True)
                test_iterator = iter(dataloader)
                # preds, truths = [], []
                sum_loss = 0
                with torch.no_grad():
                    for batch in tqdm_rank_0(test_iterator, desc="Evaluating"):
                        enc_input_ids, decoder_input_ids, dec_target_ids, dec_logit_mask, enc_attention_mask, task_id = get_loss_batch(
                            batch, args)
                        loss = forward_step_for_loss(model, enc_input_ids, enc_attention_mask, decoder_input_ids,
                                                     dec_target_ids, dec_logit_mask, task_id=task_id)
                        sum_loss += loss

                torch.distributed.barrier()

                if uniq_task_name not in all_best_results:
                    all_best_results[uniq_task_name] = {'best_result': 99999.0, 'best_code': 0,
                                                        'best_task_name': args.task_id_to_name[0]}
                print_rank_0(f'Task: {uniq_task_name} use task code from {args.task_id_to_name[cur_task_id]}, loss: {sum_loss}')
                if sum_loss < all_best_results[uniq_task_name]['best_result']:
                    new_result = {'best_result': float(sum_loss), 'best_code': cur_task_id,
                                  'best_task_name': args.task_id_to_name[cur_task_id]}
                    print_rank_0(
                        f'find better task id for task: {uniq_task_name}, pre: {all_best_results[uniq_task_name]}, '
                        f'new: {new_result}')
                    all_best_results[uniq_task_name] = new_result

    # dump
    if torch.distributed.get_rank() == 0:
        with open(os.path.join(save_dir, "best_code.txt"), "w") as f:
            json.dump(all_best_results, f, indent=4)


if __name__ == "__main__":
    # Arguments.
    py_parser = argparse.ArgumentParser(add_help=False)
    T5VAEMixIn.add_vae_prompt_specific_args(py_parser)

    py_parser.add_argument("--eval_task_set", type=str, default='test', choices=['train', 'test'])

    py_parser.add_argument("--template_dir", type=str, default='./templates_test')
    py_parser.add_argument("--ga_dev_distribution", type=str, default='ratio')

    py_parser.add_argument("--shot_num", type=int, default=32)
    py_parser.add_argument("--fewshot_shuffle_seed", type=int, default=42)

    py_parser.add_argument('--t0_combined_data', type=str,
                           default=f'./data/t0_combined_raw_data_no_prompt/t0_combined_raw_data_8node')

    known, args_list = py_parser.parse_known_args()
    args, _ = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    assert (not args.finetune)

    main(args)
