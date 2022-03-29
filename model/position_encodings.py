import math
import torch
from torch import nn

import pdb

class PositionEmbeddingSine(nn.Module):
    def __init__(self,  temperature=10000, scale=None):
        super().__init__()
        #self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    def forward(self,shape,device):
        b, n, c = shape
        #assert self.num_pos_feats == c
        dim_t = torch.arange(c,dtype=torch.float32,device=device)
        dim_t = 1/10000 ** (2 * (dim_t // 2) /c)
        #pdb.set_trace()
        pos = torch.arange(n,device=device)/n*self.scale
        pos = pos.unsqueeze(0).repeat(b,1)[:,:,None]*dim_t

        pos [:, :, 0::2] = pos [:, :, 0::2].sin()
        pos [:, :, 1::2] = pos [:, :, 1::2].cos()

        return pos



