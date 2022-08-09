import torch
import torch.nn as nn


def lrUp(d_model, step_num, warmup_step):
    lrate = d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_step ** (-1.5))

    return lrate
