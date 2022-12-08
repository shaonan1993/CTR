#!/bin/bash

task_set="t0"

tok_load_dir='./pretrained_models/t5-large-lm-adapt'
load_ckpt='./exp_dir/t0_pretrain_baseline_large_384_32_step10000/t0-large/10000'

experiment_name='t0-large'
save_dir='./exp_dir/t0_pretrain_baseline_large_384_32_step10000'

root_save_dir=save_dir
max_training_steps=10000
total_decay_iters=8000
multi_data_dir='./data/my_P3'

config_json="config/config_t5_lm_large.json"

export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch  --nproc_per_node=1  ds_evaluate_t0.py \
    --experiment-name ${experiment_name} \
    --eval_task_set test \
    --model-parallel-size 1 \
    --t5-model \
    --vocab-size 32128 \
    --num-layers 24 \
    --hidden-size 1024 \
    --inner-hidden-size 2816 \
    --num-attention-heads 16 \
    --hidden-size-per-attention-head 64 \
    --relative-attention-num-buckets 32 \
    --no-share-embeddings \
    --gated-gelu-mlp \
    --layernorm-epsilon 1e-6 \
    --init-method-std 1.0 \
    --seq-length 384 \
    --shuffle \
    --loader-scatter 1 \
    --save ${save_dir} \
    --load ${load_ckpt} \
    --summary-dir ${root_save_dir} \
    --log-interval 50 \
    --eval-interval 1000 \
    --save-interval 1000 \
    --train-iters ${max_training_steps} \
    --multi-task-ratio 1.0 \
    --multi-src-seq-length 384 \
    --multi-tgt-seq-length 32 \
    --multi-cache-dir ${multi_data_dir} \
    --tlm-data-cache-dir ${multi_data_dir} \
    --multi-task-set ${task_set} \
    --train-data wikibook \
    --tokenizer-type hf_T5Tokenizer \
    --tokenizer-model-type ${tok_load_dir} \
    --distributed-backend nccl \
    --lr-decay-style cosine \
    --lr-decay-iters ${total_decay_iters} \
    --lr-decay-ratio 0.1 \
    --warmup 0.04 \
    --checkpoint-activations \
    --no-deepspeed-load \
    --no-load-optim \
    --no-load-lr-scheduler \
    --no-load-rng \
    --no-load-iteration \
    --max-task-dataset-size 50000 \
    --new-save-directory \
    --deepspeed \
    --deepspeed_config ${config_json}