import torch
import torch.nn as nn
import copy


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def create_padding_mask(lens, max_len):

    return torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
