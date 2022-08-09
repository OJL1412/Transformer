import torch
import torch.nn as nn

from T.T_Decoder.decoderLayer import DecoderLayer
from T.T_Handle.clonelayer import clone_layers
from T.T_Input.embedding import *
from T.T_Input.positionalEncoding import *
from T.T_Mask.paddingMask import *
from T.T_Mask.subsequenceMask import *


class Decoder(nn.Module):
    def __init__(self, tgt_v_size, d_model, d_ff, N, n_head, dropout):
        """
        解码器，核心为6层解码子层
        :param tgt_v_size: 目标词表大小
        :param d_model: 默认为255
        :param d_ff: 可设为2048
        :param N: 默认为6
        :param n_head: 头数，默认为8
        :param dropout: 置0比率
        """
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)

        # input处理
        self.tgt_emb = Embedding(tgt_v_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout)

        # n层解码子层
        self.layers = clone_layers(DecoderLayer(d_model, d_ff, n_head, dropout), N)

    def forward(self, dc_inp, ec_inp, ec_out):
        """
        :param dc_inp: [batch_size, tgt_seq_len]
        :param ec_inp: [batch_size, tgt_seq_len]
        :param ec_out: [batch_size, src_seq_len, d_model]   # 用在Encoder-Decoder Attention层
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # dc_inp经过embedding和positionalEncoding处理
        dc_out = self.tgt_emb(dc_inp)  # [batch_size, tgt_seq_len, d_model]
        dc_out = self.pe(dc_out).to(device)  # [batch_size, tgt_seq_len, d_model]
        dc_out = self.dropout(dc_out)

        # mask生成，由于是解码器，既需要padding_mask，又需要subsequence_mask
        dc_pad_mask = get_pad_mask(dc_inp, dc_inp).to(device)  # [batch_size, tgt_seq_len, tgt_seq_len]
        dc_sbsq_mask = get_subsequence_mask(dc_inp).to(device)  # [batch_size, tgt_seq_len, tgt_seq_len]

        # 把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息），torch.gt比较两个矩阵的元素，大于则返回1，否则返回0
        tgt_mask = torch.gt((dc_pad_mask + dc_sbsq_mask), 0).to(device)  # [batch_size, tgt_seq_len, tgt_seq_len]

        # 屏蔽pad的信息，主要用于ec_inp
        src_mask = get_pad_mask(dc_inp, ec_inp).to(device)  # [batch_size, tgt_seq_len, src_seq_len]

        for layer in self.layers:
            dc_out = layer(dc_out, ec_out, src_mask, tgt_mask)

        return dc_out
