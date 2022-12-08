import json
import logging
import os
import argparse
from datetime import datetime

import numpy as np
import torch.distributed
import torch

from arguments import get_args
from configure_data import make_tokenizer

from utils import get_checkpoint_iteration, get_checkpoint_name, print_rank_0, get_spare_port
from SwissArmyTransformer.training.deepspeed_training import initialize_distributed, set_random_seed, get_model
from SwissArmyTransformer import mpu
from SwissArmyTransformer.model import T5Model
from utils import load_pretrained, tqdm_rank_0
from configure_data import make_data_loader
from torch.utils.data import Dataset
from templates import DatasetTemplates

from datasets import load_dataset, Value


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

# T0 sub train task
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


def special_for_duplicat(dataset_config_name, raw_dataset):
    """"remove duplicate examples"""
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

    # drop duplicate
    raw_dataset = raw_dataset.filter(lambda x: x[context_key] in valid_example_set, load_from_cache_file=False)

    return raw_dataset


def get_batch(batch, args):
    datatype = torch.int64
    keys = ["input_ids", "attention_mask", "candidate_token_ids", "candidate_attention_mask", "labels"]

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, batch, datatype)

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

    return enc_input_ids, enc_attention_mask, decoder_input_ids, dec_target_ids, dec_logit_mask, labels


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)


def generate_dataset(task_name, args, tokenizer, dataset):
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

    # debug
    valid_choice_num = len(labels[keys[0]])
    for key in keys:
        if len(labels[key]) != valid_choice_num:
            print_rank_0(key)

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

    assert all_input_ids.shape[0] == feature_dict['candidate_token_ids'].shape[0], \
        f'input_ids: {all_input_ids.shape}, candidate_token_ids: {feature_dict["candidate_token_ids"].shape}'

    feature_dict["ids"] = torch.arange(len(all_choices), dtype=torch.long)
    return DictDataset(**feature_dict)


def forward_step(model, enc_input_ids, enc_attention_mask, decoder_input_ids, dec_target_ids, dec_logit_mask,
                 length_penalty=0.0, take_softmax=True):
    cur_batch_size, seq_length = enc_input_ids.size()[:2]
    encoder_outputs = model.encode(enc_input_ids, enc_attention_mask)

    hidden_size = encoder_outputs.size()[-1]
    num_choices, tgt_seq_length = decoder_input_ids.size()[1:]
    encoder_outputs = encoder_outputs.unsqueeze(1).expand(-1, num_choices, -1, -1)
    enc_attention_mask = enc_attention_mask.unsqueeze(1).expand(-1, num_choices, 1, 1, seq_length)
    encoder_outputs = encoder_outputs.reshape(cur_batch_size * num_choices, seq_length, hidden_size)
    enc_attention_mask = enc_attention_mask.reshape(cur_batch_size * num_choices, 1, 1, seq_length)
    decoder_input_ids = decoder_input_ids.reshape(-1, tgt_seq_length)
    outputs, *_ = model.decode(decoder_input_ids, None, encoder_outputs=encoder_outputs,
                               cross_attention_mask=enc_attention_mask)
    outputs = outputs.float()

    if take_softmax:
        outputs = torch.nn.functional.log_softmax(outputs, dim=-1)

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
    model_cls = T5Model
    model = get_model(args, model_cls)
    if hasattr(model, "module"):
        module = model.module
    else:
        module = model
    load_pretrained(module, args.load, args)  # load model
    torch.distributed.barrier()

    ### T0 test data Stuff
    world_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    eval_batch_size = args.batch_size * world_size
    if args.eval_batch_size is not None:
        eval_batch_size = args.eval_batch_size * world_size

    # save_dir = os.path.join("eval_t0", "eval_results", args.experiment_name)
    save_dir = os.path.join("eval_results", args.experiment_name)
    if torch.distributed.get_rank() == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
                continue   # drop invalid example
            features['is_correct'].extend(current_is_correct)

            current_idx = [[total_idx, i] for i in range(len(answer_choices_texts[batch_idx]))]
            features['idx'].extend(current_idx)
            features['inputs_pretokenized'].extend([input_texts[batch_idx]] *
                                                   len(answer_choices_texts[batch_idx]))
            features['targets_pretokenized'].extend(answer_choices_texts[batch_idx])

            features['weight'].extend([1.0] * len(answer_choices_texts[batch_idx]))

        return features

    model.eval()
    # traverse each dataset
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
                'validation': os.path.join(train_file_path, 'validation.json')
            }
            raw_datasets = load_dataset('json', data_files=data_files)['validation']

        # Optional: only eval 1000 examples for saving time
        if dataset_name != 'hellaswag':
            raw_datasets = raw_datasets.select(range(min(1000, len(raw_datasets))))

        column_names = raw_datasets.column_names

        print_rank_0(f'use template_dir: {args.template_dir}')
        prompts = DatasetTemplates(
            f"{dataset_name}" if dataset_config_name is None else f"{dataset_name}/{dataset_config_name}",
            template_dir=args.template_dir)

        template_list = prompts.templates.keys()
        print_rank_0(f'{task_name} contains templates: {template_list}')

        all_results = {}
        # traverse each task
        for template_id in template_list:
            template = prompts.templates[template_id]
            template_name = template.name

            uniq_task_name = f'{dataset_name}_{dataset_config_name}' if dataset_config_name else dataset_name
            norm_template_name = template_name.replace('/', '_')
            norm_template_name = norm_template_name.replace(' ', '_')
            norm_template_name = norm_template_name.replace('-', '_')
            norm_template_name = norm_template_name.replace('?', '_')

            uniq_task_name = f'{uniq_task_name}_{norm_template_name}'

            if template.metadata.original_task is not True:
                print_rank_0(f'skip template {template_name}, not original template.')
                continue

            filtered_dataset = None
            # some template can only apply to a part of example
            if dataset_config_name == 'copa':
                if template_name in ["\u2026What could happen next, C1 or C2?", "\u2026As a result, C1 or C2?"]:
                    filtered_dataset = raw_datasets.filter(lambda example: example['question'] == 'effect', load_from_cache_file=False)
                if template_name in ["\u2026which may be caused by", "\u2026why? C1 or C2"]:
                    filtered_dataset = raw_datasets.filter(lambda example: example['question'] == 'cause', load_from_cache_file=False)

            if not filtered_dataset:
                filtered_dataset = raw_datasets

            processed_dataset = filtered_dataset.map(
                preprocess_function, batched=True, with_indices=True, remove_columns=column_names,
                load_from_cache_file=False)

            # dedup
            if dataset_config_name == 'wsc.fixed':
                print_rank_0(f'before process wsc.fixed  {template_name}: {raw_datasets}')
                processed_dataset = special_for_duplicat(dataset_config_name, processed_dataset)
                print_rank_0(f'after process wsc.fixed {template_name}: {raw_datasets}')

            test_dataset = generate_dataset(uniq_task_name, args, tokenizer, processed_dataset)

            dataloader = make_data_loader(test_dataset, tokenizer, batch_size=eval_batch_size, args=args,
                                          num_iters=None, shuffle=False, collator=None, keep_last=True)
            test_iterator = iter(dataloader)
            preds, truths = [], []
            with torch.no_grad():
                for batch in tqdm_rank_0(test_iterator, desc="Evaluating"):
                    enc_input_ids, enc_attention_mask, decoder_input_ids, dec_target_ids, dec_logit_mask, \
                    labels = get_batch(batch, args)
                    logits = forward_step(model, enc_input_ids, enc_attention_mask, decoder_input_ids, dec_target_ids,
                                          dec_logit_mask)

                    pred = torch.argmax(logits, dim=1, keepdim=False).view(-1).detach().cpu().numpy().tolist()
                    truth = torch.argmax(labels, dim=1, keepdim=False).view(-1).detach().cpu().numpy().tolist()
                    preds += pred
                    truths += truth

            assert len(preds) == len(truths)
            ret = sum((np.array(preds) == np.array(truths)))
            all = len(preds)

            predict_for_current_task = torch.tensor(ret, dtype=torch.float).cuda()
            all_for_current_task = torch.tensor(all, dtype=torch.float).cuda()

            torch.distributed.barrier()

            torch.distributed.all_reduce(tensor=predict_for_current_task)
            torch.distributed.all_reduce(tensor=all_for_current_task)
            print_rank_0(all_for_current_task)
            all_results[uniq_task_name] = float(predict_for_current_task / all_for_current_task)

        if torch.distributed.get_rank() == 0:
            with open(os.path.join(save_dir, task_name + "-results.txt"), "w") as f:
                out_str = ""
                out_str += (f"number of promtps: {len(all_results.keys())}" + "\n")
                print_rank_0(f"number of promtps: {len(all_results.keys())}")
                for subname, value in all_results.items():
                    print_rank_0(subname + ":" + str(value))
                    out_str += (subname + ":" + str(value) + "\n")

                out_str += f"task_name: {task_name} | num_subset: {len(all_results.keys())}.\n"
                results = np.array(list(all_results.values()))
                assert len(results) == len(all_results.keys())
                print(results)
                q3, q1 = np.percentile(results, [75, 25])
                median = np.median(results)
                mean = np.mean(results)
                out_str += f"q3: {q3}." + "\n"
                out_str += f"q1: {q1}." + "\n"
                out_str += f"inter-quartile range: {q3 - q1}." + "\n"
                out_str += f"median: {median}." + "\n"
                out_str += f"mean: {mean}." + "\n"
                out_str += "***************** end ****************" + "\n\n"
                f.write(out_str)


if __name__ == "__main__":
    # Arguments.
    py_parser = argparse.ArgumentParser(add_help=False)
    T5Model.add_model_specific_args(py_parser)

    # eval train tasks or test tasks
    py_parser.add_argument("--eval_task_set", type=str, default='test', choices=['train', 'test'])

    py_parser.add_argument("--template_dir", type=str, default='./templates_test')

    known, args_list = py_parser.parse_known_args()
    args, _ = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    assert (not args.finetune)

    main(args)