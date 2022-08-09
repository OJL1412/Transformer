import torch
import torch.nn as nn
import torch.nn.functional as F

from T.T_Decoder.decoder import Decoder
from T.T_Encoder.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, src_v_size, tgt_v_size, d_model=512, d_ff=2048, N=6, n_head=8, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_v_size, d_model, d_ff, N, n_head, dropout)
        self.decoder = Decoder(tgt_v_size, d_model, d_ff, N, n_head, dropout)

        self.classifier = nn.Linear(d_model, tgt_v_size)

    def forward(self, src, tgt):
        """
        :param src: [batch_size, src_len]
        :param tgt: [batch_size, tgt_len]
        :return:
        """

        ec_out = self.encoder(src)
        dc_out = self.decoder(tgt, src, ec_out)

        output = self.classifier(dc_out)
        output = output.view(-1, output.size(-1))
        # output = F.log_softmax(output, dim=-1)

        return output


