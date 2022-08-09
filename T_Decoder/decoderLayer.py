import torch
import torch.nn as nn
import numpy as np

from T.T_IayerHandle.feedForward import FeedForward
from T.T_IayerHandle.multiHeadAttention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout):
        super(DecoderLayer, self).__init__()

        # Masked Multi-Head Attention
        self.masked_attn = MultiHeadAttention(d_model, n_head, dropout)

        # Multi-Head Attention
        self.attn = MultiHeadAttention(d_model, n_head, dropout)

        # Feed Forward
        self.FF = FeedForward(d_model, d_ff, dropout)

        # Norm
        self.norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, dc_inp, ec_out, src_mask, tgt_mask):
        """
        :param dc_inp: [batch_size, tgt_len, d_model]
        :param ec_out: [batch_size, src_len, d_model]
        :param src_mask: [batch_size, tgt_len, src_len]
        :param tgt_mask: [batch_size, tgt_len, tgt_len]
        """
        # Multi-Head Attention Masked 和 Add & Norm
        # [batch_size, tgt_seq_len, src_seq_len] [batch_size, 8, tgt_seq_len, tgt_seq_len]
        x = dc_inp
        mask_attn_out = self.masked_attn(x, x, x, tgt_mask)
        mask_attn_out = self.norm(mask_attn_out + x)
        # print("解码层——带mask——多头注意力机制结果:")
        # print(x.size())
        # print(x.size())
        # print(x.size())
        # print("**" * 30)

        # Multi-Head Attention 和 Add & Norm
        # dc_out: [batch_size, tgt_seq__len, d_model], ec_res: [batch_size, h_heads, tgt_seq_len, src_seq_len]
        attn_out = self.attn(mask_attn_out, ec_out, ec_out, src_mask)
        attn_out = self.norm(attn_out + mask_attn_out)
        # print("解码层——多头注意力机制结果:")
        # print(x.size())
        # print(ec_res.size())
        # print(ec_res.size())
        # print("**" * 30)

        # Feed Forward 和 Add & Norm
        ff_out = self.FF(attn_out)
        ff_out = self.norm(ff_out + attn_out)

        return self.dropout(ff_out)

