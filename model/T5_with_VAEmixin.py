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

"""VQVAE implementation"""

import torch

from SwissArmyTransformer.model import T5Model
from SwissArmyTransformer.model.mixins import BaseMixin
from utils import print_rank_0

from torch.nn.parameter import Parameter

from torch import nn
import math
from model.VQVAE import VectorQuantizer


class VAEMixin(BaseMixin):
    def __init__(self, args, prompt_weight_decay=None, ):
        super().__init__()
        """VQVAE"""
        self.prompt_weight_decay = prompt_weight_decay
        self.prompt_num_per_task = args.prompt_num_per_task
        self.vq_num_embeddings = args.vq_num_embeddings
        self.task_num = args.task_num

        encoder_weight = Parameter(torch.empty((args.prompt_num_per_task,  # [20, 1024, 1024]
                                                args.task_embed_dim, args.vq_embedding_dim)))

        self.vae_encoder = encoder_weight

        if args.task_embedding_init == "des_init":
            print_rank_0(f'init task embeddings from description')
            init_emb = torch.load('./data/TaskEmbedding/cls_tensor.pt', map_location='cpu')
            self.task_embeddings = torch.nn.Embedding.from_pretrained(init_emb)
        elif args.task_embedding_init == "ran_init":
            # Note: embedding layer default init method: N(0, 1)
            self.task_embeddings = torch.nn.Embedding(args.task_num, args.task_embed_dim)
        else:
            raise ValueError(f'{self.task_embedding} has not bee implemented')

        self.vae_decoder = nn.Linear(args.vq_embedding_dim, args.hidden_size, bias=False)
        self.quantizer = VectorQuantizer(args.vq_num_embeddings, args.vq_embedding_dim, commitment_cost=args.commitment_cost)

        self.current_vae_loss = None
        self.args = args
        self.args.task_code_memory = dict()   # task_id -> selected_code

        self.init_vae_param()

    def get_params_for_weight_decay_optimization(self):
        # set weight decay
        if self.prompt_weight_decay is not None:
            print_rank_0(f"set weight decay {self.prompt_weight_decay} for prefix prompt")
            weight_decay_params = {'params': [], 'weight_decay': self.prompt_weight_decay}
            weight_decay_params['params'].extend(
                [p for n, p in list(self.named_parameters())
                 if p is not None and n != 'bias' and p.requires_grad])
            return weight_decay_params
        else:
            return None

    def init_vae_param(self):
        for n, p in self.named_parameters():
            if n == 'vae_encoder':
                # Note: liner default init method: uniform(-1/sqrt(in_features), 1/sqrt(in_features))
                torch.nn.init.normal_(p, mean=0, std=0.1)
            elif n == 'task_embeddings.weight':
                pass
            elif n == 'vae_decoder.weight':
                pass
            elif n == 'quantizer._embedding.weight':
                # *100 for numerical stability
                torch.nn.init.uniform_(p, -1/self.vq_num_embeddings * 100, 1/self.vq_num_embeddings * 100)
            else:
                raise NotImplementedError(n)

    def word_embedding_forward(self, input_ids, **kwargs):
        if 'code' in kwargs:
            quantized = self.quantizer.get_encoding(kwargs['code'])
            loss = 0

        else:
            if 'task_emb' in kwargs:
                task_embs = kwargs['task_emb']
            else:
                assert 'task_id' in kwargs, f'kw_args: {kwargs}'
                task_id = kwargs['task_id']
                task_embs = self.task_embeddings(task_id).to(input_ids.device)

            batch_size = input_ids.shape[0]

            # task_embs: [batch_size, emb_dim] * [prompt_num_per_task, embed_dim, embed_dim]
            task_encoding = torch.matmul(task_embs, self.vae_encoder).permute(1, 0, 2)  # [batch_size, prompt_num_per_task, emb_dim]

            # print_rank_0(f'debug task_encoding: {task_encoding.shape}')
            loss, quantized, perplexity, encodings, encoding_indices = self.quantizer(task_encoding)
            self.current_vae_loss = loss

        vae_prompt_embs = self.vae_decoder(quantized)

        word_embs = self.transformer.word_embeddings(input_ids)

        hidden_stats = torch.cat((vae_prompt_embs, word_embs), dim=1)

        if 'task_id' in kwargs:
            for cur_task, cur_encoding_idx in zip(task_id, encoding_indices.view(batch_size, -1)):
                self.args.task_code_memory[cur_task.item()] = cur_encoding_idx.tolist()

        return hidden_stats


class T5VAEMixIn(T5Model):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.args = args
        self.prompt_num_per_task = args.prompt_num_per_task
        self.task_num = args.task_num

        self.freeze_transformer = args.freeze_transformer
        self.freeze_vae = args.freeze_vae
        self.freeze_task_emb = args.freeze_task_emb

        # self.freeze_strategy = args.freeze_strategy
        if args.prompt_num_per_task > 0:
            # Note: only input of encoder block contains CTR
            self.encoder.add_mixin('vae_prompt', VAEMixin(args))

        self.disable_untrainable_params()

    def forward(self, enc_input_ids, dec_input_ids, *, enc_attention_mask=None, dec_attention_mask=None,
                cross_attention_mask=None, **kw_args):
        batch_size, seq_length = enc_input_ids.size()[:2]

        if enc_attention_mask is None:
            enc_attention_mask = torch.ones(1, 1, 1, seq_length + self.prompt_num_per_task,
                                            dtype=self.encoder.transformer.word_embeddings.weight.dtype,
                                            device=enc_input_ids.device)
        else:
            more_attention_mask = torch.ones(batch_size, 1, 1, self.prompt_num_per_task,
                                             device=enc_input_ids.device).type_as(enc_input_ids)
            enc_attention_mask = torch.cat((more_attention_mask, enc_attention_mask), dim=-1)

        if cross_attention_mask is None:
            cross_attention_mask = enc_attention_mask
        encoder_outputs = self.encode(enc_input_ids, enc_attention_mask, **kw_args)

        decoder_outputs, *mems = self.decode(dec_input_ids, dec_attention_mask,
                                             encoder_outputs=encoder_outputs,
                                             cross_attention_mask=cross_attention_mask,
                                             **kw_args)

        # debug
        assert self.encoder.mixins["vae_prompt"].current_vae_loss is not None or 'code' in kw_args

        vae_loss = self.encoder.mixins["vae_prompt"].current_vae_loss
        self.encoder.mixins["vae_prompt"].current_vae_loss = None

        return (encoder_outputs, decoder_outputs, vae_loss, *mems)

    def disable_untrainable_params(self):
        if self.freeze_transformer:
            print_rank_0("Freeze LM parameters")
            self.requires_grad_(False)
            if "vae_prompt" in self.encoder.mixins:
                self.encoder.mixins["vae_prompt"].requires_grad_(True)
            else:
                raise AssertionError(self.encoder.mixins)
        if self.freeze_vae:
            print_rank_0("Freeze VAE parameters")
            self.requires_grad_(True)
            if "vae_prompt" in self.encoder.mixins:
                self.encoder.mixins["vae_prompt"].requires_grad_(False)
            else:
                raise AssertionError(self.encoder.mixins)

        if self.freeze_task_emb:
            print_rank_0("Freeze task embedding parameters")
            if "vae_prompt" in self.encoder.mixins:
                self.encoder.mixins["vae_prompt"].task_embeddings.weight.requires_grad_(False)
            else:
                raise AssertionError(self.encoder.mixins)

    @classmethod
    def add_vae_prompt_specific_args(cls, parser):
        super().add_model_specific_args(parser)

        parser.add_argument('--task_embedding_init', type=str, default="des_init")
        parser.add_argument("--task_num", type=int, default=318)
        parser.add_argument("--task_embed_dim", type=int, default=1024)
        parser.add_argument("--prompt_num_per_task", type=int, default=10)
        parser.add_argument("--vq_num_embeddings", type=int, default=128)
        parser.add_argument("--vq_embedding_dim", type=int, default=1024)
        parser.add_argument("--commitment_cost", type=float, default=0.1, help='commitment_cost')

        parser.add_argument('--freeze_vae', action='store_true', default=False)
        parser.add_argument('--freeze_task_emb', action='store_true', default=False)

