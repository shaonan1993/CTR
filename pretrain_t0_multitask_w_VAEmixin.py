# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain T5 Model"""

# Flag to use Pytorch ddp which uses overlapping communication and computation.
import json
from datetime import datetime
import os
import random
import math
import argparse

import deepspeed
import torch.distributed
from filelock import FileLock
import torch

from arguments import get_args
from configure_data import make_tokenizer, read_internal_dataset, build_taskid_dataset
import pathlib

from utils import Timers
from utils import save_checkpoint, load_checkpoint
from utils import print_and_save_args, print_rank_0, get_sample_writer, get_log_dir
from SwissArmyTransformer.training.deepspeed_training import initialize_distributed, \
    set_random_seed, setup_model_and_optimizer, get_model, get_optimizer_param_groups
from SwissArmyTransformer import mpu
from model.T5_with_VAEmixin import T5VAEMixIn
from learning_rates import get_learning_rate_scheduler
from train_utils import evaluate_and_print_results, train

# for reproduce
torch.use_deterministic_algorithms(True)


def decoder_shift_right(input_ids, args):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = args.decoder_start_token_id
    return shifted_input_ids


def get_batch(data, args):
    keys = ['text', 'loss_mask', 'target', 'attention_mask', 'task_id']
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['text'].long()
    task_id = data_b['task_id'].long()
    labels = data_b['target'].long()
    decoder_tokens = decoder_shift_right(labels, args)
    attention_mask = data_b['attention_mask'].long()
    loss_mask = data_b['loss_mask'].float()

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
    elif args.bf16:
        attention_mask = attention_mask.bfloat16()
    return tokens, decoder_tokens, labels, loss_mask, attention_mask, task_id


def save_codebook(iteration, args):
    if torch.distributed.get_rank() == 0:
        task_id_to_name = args.task_id_to_name
        task_id_to_code = args.task_code_memory
        task_name_to_code = {task_id_to_name[task_id]: str(code) for task_id, code in
                             sorted(task_id_to_code.items(), key=lambda item: item[0])}
        json.dump(task_name_to_code, open(os.path.join(args.save, f'codebok_step{iteration}.json'), 'w'), indent=4)


def forward_step(data_iterator, model, args, timers, **kwargs):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    timers('data loader').start()
    """
    rand = random.Random(args.iteration * mpu.get_data_parallel_world_size() + mpu.get_data_parallel_rank())
    if data_iterator[1] and rand.random() < args.multi_task_ratio:
        data = next(data_iterator[1]) if data_iterator[1] else None
        data["mode"] = "multi-task"
    else:
        data = next(data_iterator[0]) if data_iterator[0] else None
    """

    data = next(data_iterator[1]) if data_iterator[1] else None
    data["mode"] = "multi-task"

    timers('data loader').stop()
    tokens, decoder_tokens, labels, loss_mask, attention_mask, task_id = get_batch(data, args)
    timers('batch generator').stop()

    if data is not None and "mode" in data:
        mode = data['mode']
    else:
        mode = 'bert'

    _, logits, vae_loss, *_ = model(enc_input_ids=tokens, dec_input_ids=decoder_tokens,
                                    enc_attention_mask=attention_mask, task_id=task_id)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    print_rank_0(f'debug: vae_loss {vae_loss.item()}')
    loss = torch.sum(losses.view(-1) * loss_mask)
    if loss_mask.sum().item() > 0:
        loss = loss / loss_mask.sum()
    loss = loss + vae_loss
    metrics = {name: torch.cuda.FloatTensor([1]) if name == mode else torch.cuda.FloatTensor([0]) for name in
               ['bert', 'sentence', 'gpt', 'multi-task']}
    return loss, metrics


def main(args):
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False
    # Timer.
    timers = Timers()

    if args.load and not args.new_save_directory:
        args.experiment_name = os.path.basename(os.path.normpath(args.load))
    else:
        args.experiment_name = args.experiment_name
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)
        os.makedirs(args.save, exist_ok=True)

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    tokenizer = make_tokenizer(args)
    args.decoder_start_token_id = tokenizer.get_command('sop').Id
    args.do_train = True
    args.do_valid = True

    # To ensure T0 config
    assert args.multi_task_ratio == 1.0
    assert (not args.reweight)
    assert args.shuffle
    assert args.t5_model

    args.reload_combined_raw_data = False

    # reload data
    t0_combined_raw_data_dir = args.t0_combined_data
    if not args.reload_combined_raw_data:
        print_rank_0(f'We will load exist data from {t0_combined_raw_data_dir}')
        source_train = json.load(open(os.path.join(t0_combined_raw_data_dir,
                                                   f't0_train_src_5w_rank_{torch.distributed.get_rank()}.json')))
        target_train = json.load(open(os.path.join(t0_combined_raw_data_dir,
                                                   f't0_train_tgt_5w_rank_{torch.distributed.get_rank()}.json')))
        source_valid = json.load(open(os.path.join(t0_combined_raw_data_dir,
                                                   f't0_valid_src_5w_rank_{torch.distributed.get_rank()}.json')))
        target_valid = json.load(open(os.path.join(t0_combined_raw_data_dir,
                                                   f't0_valid_tgt_5w_rank_{torch.distributed.get_rank()}.json')))
        t0_task_names = json.load(open(os.path.join(t0_combined_raw_data_dir, 't0_task_names.json')))
    else:
        print_rank_0(f'We will reload all data and save to {t0_combined_raw_data_dir}')
        os.makedirs(t0_combined_raw_data_dir, exist_ok=True)
        source_train, target_train, source_valid, target_valid, t0_task_names = read_internal_dataset(args, tokenizer)
        # save data
        json.dump(source_train, open(os.path.join(t0_combined_raw_data_dir,
                                                  f't0_train_src_5w_rank_{torch.distributed.get_rank()}.json'), 'w'),
                  indent=4)
        json.dump(target_train, open(os.path.join(t0_combined_raw_data_dir,
                                                  f't0_train_tgt_5w_rank_{torch.distributed.get_rank()}.json'), 'w'),
                  indent=4)
        json.dump(source_valid, open(os.path.join(t0_combined_raw_data_dir,
                                                  f't0_valid_src_5w_rank_{torch.distributed.get_rank()}.json'), 'w'),
                  indent=4)
        json.dump(target_valid, open(os.path.join(t0_combined_raw_data_dir,
                                                  f't0_valid_tgt_5w_rank_{torch.distributed.get_rank()}.json'), 'w'),
                  indent=4)
        json.dump(t0_task_names, open(os.path.join(t0_combined_raw_data_dir,
                                                   f't0_task_names.json'), 'w'), indent=4)

    task_name_to_id = dict()
    task_id_to_name = dict()
    for i, task_name in enumerate(t0_task_names):
        task_name_to_id[task_name] = i
        task_id_to_name[i] = task_name  # task_id -> task_name
    args.task_id_to_name = task_id_to_name

    task_train_loader, task_valid_loader = build_taskid_dataset(
        args, tokenizer, t0_task_names,
        source_train, target_train, source_valid, target_valid, task_name_to_id)
    torch.distributed.barrier()
    print_rank_0("Data Loading Finished!!!")

    # Model, optimizer, and learning rate.
    model_cls = T5VAEMixIn
    model, optimizer = setup_model_and_optimizer(args, model_cls=model_cls)

    lr_scheduler = get_learning_rate_scheduler(optimizer, args) if optimizer is not None else None
    print_rank_0(f'debug: optimizer: {optimizer}, lr_scheduler: {lr_scheduler}')
    if args.load is not None:
        # with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
        #     args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args, no_deepspeed=args.no_deepspeed_load)
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args, no_deepspeed=args.no_deepspeed_load)
        if args.no_load_lr_scheduler:
            lr_scheduler.num_iters = args.iteration
    else:
        args.iteration = 0
    torch.distributed.barrier()
    if args.switch_linear:
        lr_scheduler.switch_linear(args)

    summary_writer = None
    if torch.distributed.get_rank() == 0:
        if args.train_iters > 0:
            args.log_dir = get_log_dir(base=args.summary_dir, name=args.experiment_name)
            summary_writer = get_sample_writer(log_dir=args.log_dir, iteration=args.iteration)
        print_and_save_args(args, verbose=True, log_dir=args.log_dir)

    multi_train_iterator = iter(task_train_loader) if task_train_loader is not None else None
    multi_val_iterator = iter(task_valid_loader) if task_valid_loader is not None else None

    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            iteration, _ = train(model, optimizer, lr_scheduler,
                                 (None, multi_train_iterator),
                                 (None, multi_val_iterator),
                                 timers, args, summary_writer=summary_writer,
                                 hooks={"forward_step": forward_step,
                                        "save_codebook": save_codebook})

        if args.do_valid:
            prefix = 'the end of training for val data'
            evaluate_and_print_results(prefix, (None, multi_val_iterator),
                                       model, args, timers, verbose=False, forward_step_func=forward_step)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args, no_save_optim=args.no_save_optim)


if __name__ == "__main__":
    # Arguments.
    py_parser = argparse.ArgumentParser(add_help=False)

    T5VAEMixIn.add_vae_prompt_specific_args(py_parser)
    py_parser.add_argument('--t0_combined_data', type=str,
                           default=f'./data/t0_combined_raw_data_8node')

    known, args_list = py_parser.parse_known_args()
    args, config_params = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    # print_rank_0(f'debug: args: {args}')
    main(args)
