import torch
import torch.nn as nn

from T.T_Encoder.encoderLayer import EncoderLayer
from T.T_Handle.clonelayer import clone_layers
from T.T_Input.embedding import *
from T.T_Input.positionalEncoding import *
from T.T_Mask.paddingMask import *


class Encoder(nn.Module):
    def __init__(self, src_v_size, d_model, d_ff, N, n_head, dropout):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)

        # input处理
        self.src_emb = Embedding(src_v_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout)

        # n层编码子层
        self.layers = clone_layers(EncoderLayer(d_model, d_ff, n_head, dropout), N)

    def forward(self, ec_inp):
        """
        :param ec_inp: (batch_size, seq_len)
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ec_out = self.src_emb(ec_inp).to(device)
        ec_out = self.pe(ec_out).to(device)
        ec_out = self.dropout(ec_out)

        mask = get_pad_mask(ec_inp, ec_inp)  # [batch_size, src_seq_len, src_seq_len]

        for layer in self.layers:
            ec_out = layer(ec_out, mask)

        return ec_out
