import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:

            attention = attention.masked_fill_(attn_mask, float("-inf"))

        attention = self.softmax(attention)

        attention = self.dropout(attention)

        context = torch.bmm(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
        query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention
    
def residual(sublayer_fn,x):
    return sublayer_fn(x)+x

class LayerNorm(nn.Module):

    def __init__(self, features, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        # alpha
        self.gamma = nn.Parameter(torch.ones(features))
        # beta
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta
    
def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1) # L_q
    # `PAD` is 0
    pad_mask = seq_k.eq(0) # batch_size * L_k
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_len):

        super(PositionalEncoding, self).__init__()
        
        position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)] for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model]).double()
        position_encoding = torch.cat((pad_row, torch.tensor(position_encoding)))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)
    def forward(self, input_len, max_len):

        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        input_pos = tensor(
          [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)

class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

class EncoderLayer(nn.Module):

    def __init__(self, model_dim=768, num_heads=8, ffn_dim=3072, dropout=0.0, whether_PositionalWiseFeedForward=True):
        super(EncoderLayer, self).__init__()
        self.whether_PositionalWiseFeedForward = whether_PositionalWiseFeedForward

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        output, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        if self.whether_PositionalWiseFeedForward:
            output = self.feed_forward(output)

        return output, attention

class transformerEncoder(nn.Module):

    def __init__(self,
                 max_seq_len,
                 num_layers=6,
                 model_dim=768,
                 num_heads=8,
                 ffn_dim=3072,
                 dropout=0.0,
                 whether_PositionalEncoding=True,
                 whether_PositionalWiseFeedForward=True
                ):
        super().__init__()
        self.whether_PositionalEncoding = whether_PositionalEncoding
        self.whether_PositionalWiseFeedForward = whether_PositionalWiseFeedForward

        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, ffn_dim, dropout, self.whether_PositionalWiseFeedForward) for _ in range(num_layers)])

        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, attention_mask):
        inputs_len = torch.cuda.LongTensor([sum(row).item() for row in attention_mask])
        output = inputs # batch_size * seq_len * hidden_state
        if self.whether_PositionalEncoding:
            output += self.pos_embedding(inputs_len, inputs.size(1)) # batch_size * seq_len * hidden_state

        self_attention_mask = padding_mask(attention_mask, attention_mask) # batch_size * seq_len_q * seq_len_k

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions
