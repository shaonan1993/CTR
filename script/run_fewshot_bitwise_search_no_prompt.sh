#!/bin/bash

FEWSHOT_SEED=19735

shot_num=32
task_set="t0"

tok_load_dir='./pretrained_models/t5-large-lm-adapt'

experiment_name=t0-large-vae-bitwise-search_loss_no_prompt_${shot_num}_${FEWSHOT_SEED}
commitment_cost=0.1
save_dir='./exp_dir/t0_pretrain_vae_large_384_32_step20000_des_no_prompt_128_10_0.1_pipe'
root_save_dir=save_dir
max_training_steps=10000
total_decay_iters=8000
multi_data_dir='./data/my_P3_no_prompt'
config_json="config/config_t5_lm_large.json"

t0_combined_data="./data/t0_combined_raw_data_no_prompt/t0_combined_raw_data_8node"
load_ckpt=./exp_dir/t0_pretrain_vae_large_384_32_step20000_des_no_prompt_128_10_0.1_pipe/t0-large-vae-pipe/10000

export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch  --nproc_per_node=1 --master_port=56420  fewshot_t0_VAEmixin_bitwise_search_loss.py \
    --experiment-name ${experiment_name} \
    --eval_task_set test \
    --t0_combined_data ${t0_combined_data} \
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
    --fewshot_shuffle_seed ${FEWSHOT_SEED}  \
    --layernorm-epsilon 1e-6 \
    --init-method-std 1.0 \
    --prompt_num_per_task 10 \
    --init_taskcode ./eval_results/t0-large-vae-discrete-search_loss_no_prompt_32_${FEWSHOT_SEED}/best_code.txt \
    --codebook_dir ./exp_dir/t0_pretrain_vae_large_384_32_step20000_des_no_prompt_128_10_0.1_pipe/t0-large-vae-pipe/codebok_step10000.json \
    --seq-length 384 \
    --shuffle \
    --use_no_prompt \
    --loader-scatter 1 \
    --epoch_num 20 \
    --shot_num ${shot_num} \
    --template_dir ./templates_no_prompt \
    --task_embedding_init ran_init \
    --commitment_cost ${commitment_cost} \
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
