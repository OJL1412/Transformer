import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def SelfAttention(q, k, v, mask=None, dropout=None):
    """
    :param q: [batch_size, n_head, seq_len_q, d_k]
    :param k: [batch_size, n_head, seq_len_k, d_k]
    :param v: [batch_size, n_head, seq_len_v(=seq_len_k), d_v]
    :param mask: [batch_size, n_heads, seq_len, seq_len]
    :param dropout: 置0概率
    :return:
    """
    d_k = q.shape[-1]

    score = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)  # (batch_size, 8, -1, -1) / math.sqrt(d_k)

    if mask is not None:
        score = score.masked_fill(mask, -1e9)

    prob_attn = F.softmax(score, dim=-1)

    if dropout is not None:
        prob_attn = dropout(prob_attn)

    a_q_k_v = torch.matmul(prob_attn, v)  # (batch_size, 8, -1, -1) * (batch_size, 8, -1, 64) => (batch_size, 8, -1, 64)

    return a_q_k_v


