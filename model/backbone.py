from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from util.misc import is_main_process
import config
import pdb
#from util.misc import  is_main_process

#from .position_encodings import build_position_encoding
class FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        inds = config.INDS
        self.center_left = [config.AU_CENTERS_LEFT[i] for i in inds]
        self.center_right = [config.AU_CENTERS_RIGHT[i] for i in inds]
        self.location_scale = [config.LOCATION_SCALE[i] for i in inds]
        
    def forward(self, tensor, landmarks):
        
        batch,channels,h,w=tensor.shape
        left,right = self.au_centers(landmarks) #batch X 2 X n_centers  
        
        spatial_scale = min(h,w)/config.IMG_SIZE
        left = (left*spatial_scale).round().long()
        right = (right*spatial_scale).round().long()
        left[:,0].clamp_(0,w-1);left[:,1].clamp_(0,h-1)
        right[:,0].clamp_(0,w-1);right[:,1].clamp_(0,h-1)

        n_centers=left.shape[-1]
        to_encoder = tensor[torch.arange(batch).repeat_interleave(n_centers),:,left[:,0,:].flatten(),left[:,1,:].flatten()].view(batch,n_centers,channels)
        to_decoder = tensor[torch.arange(batch).repeat_interleave(n_centers),:,right[:,0,:].flatten(),right[:,1,:].flatten()].view(batch,n_centers,channels)        
        return to_encoder, to_decoder


    def au_centers(self,landmarks):
        #pdb.set_trace()
        ruler = abs(landmarks[:,0, 22] - landmarks[:,0, 25]) #batch
        scales = ruler[:,None] * torch.tensor(self.location_scale,device = landmarks.device) #batch X n_loc
        centers_left_x = landmarks[:,0,self.center_left]; centers_right_x = landmarks[:,0,self.center_right] #batchXn_cent
        centers_left_y = landmarks[:,1,self.center_left] + scales; centers_right_y = landmarks[:,1,self.center_right] + scales 
        return torch.stack((centers_left_x,centers_left_y),dim=1).round(),torch.stack((centers_right_x,centers_right_y),dim=1).round()
            
class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x      

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool,num_channels: int, feat_extractor: FeatureExtraction, interpolator: Interpolate,
                  embedding_dim: int):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
 
        return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.interpolator = interpolator
        self.feat_extractor = feat_extractor
        self.linear = nn.Linear(num_channels,embedding_dim)

    def forward(self, tensors,landmarks):
        xs = self.body(tensors)['0']
        if self.interpolator !=None:
            xs = self.interpolator(xs)
        #xs.register_hook(lambda x: torch.save(x,'grad.pt'))
        to_enc,to_dec = self.feat_extractor(xs,landmarks)
        to_enc = self.linear(to_enc); to_dec=self.linear(to_dec)
        return to_enc,to_dec


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    
    def __init__(self, name: str,
                 train_backbone: bool,
                 feat_extractor: FeatureExtraction,
                 interpolate: Interpolate = None,
                 embedding_dim: int =1024,
                 dilation: bool=False):
        #TODO:check main process
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d)
        backbone.conv1 = nn.Conv2d(3,64,kernel_size=(3,3),padding=(1,1))
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone,num_channels,feat_extractor, interpolate,embedding_dim=embedding_dim)




        
