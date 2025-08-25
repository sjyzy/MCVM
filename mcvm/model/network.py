import sys
import math
import numpy as np
import einops
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal
from .vcdcdensenet_encode import VCDCDenseNet_encode
from .decode import MLP_decoder, SpatialTransformer_block, Multiview_decoder
from .Double_Densenet import Double_Densenet
from .Double_Densenet_BPDC import Double_Densenet_BPDC,Double_Densenet_BPDC264
from torchinfo import summary
import torchvision.models as models


class densenet_fpran(nn.Module):
    
    def __init__(self, 
                 in_channels: int = 1, 
                 enc_channels: int = 128,
                 dec_channels: int = 8,
                 use_checkpoint: bool = False):
        super().__init__()
        
        # self.Encoder = VCDCDenseNet_encode(spatial_dims=3, in_channels=1, out_channels=2)
        self.Encoder = Double_Densenet(spatial_dims=3, in_channels=in_channels, out_channels=enc_channels)
        self.Decoder = Multiview_decoder(in_channels=enc_channels,
                                   channel_num=dec_channels,
                                   use_checkpoint=use_checkpoint)
        
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

        

    def forward(self, x):
        x_class, x_sag, x_tra = self.Encoder(x)
        flow = self.Decoder(x_sag, x_tra)
        warped = self.SpatialTransformer(x[:,0,:,:,:].unsqueeze(1), flow)
        return warped, flow, x_class
    

class bpdc_densenet_fpran(nn.Module):
    
    def __init__(self, 
                 in_channels: int = 1, 
                 enc_channels: int = 128,
                 dec_channels: int = 8,
                 use_checkpoint: bool = False):
        super().__init__()
        
        # self.Encoder = VCDCDenseNet_encode(spatial_dims=3, in_channels=1, out_channels=2)
        self.Encoder = Double_Densenet_BPDC(spatial_dims=3, in_channels=in_channels, out_channels=enc_channels)
        self.Decoder = Multiview_decoder(in_channels=enc_channels,
                                   channel_num=dec_channels,
                                   use_checkpoint=use_checkpoint)
        
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

        

    def forward(self, x):
        x_class, x_sag, x_tra,  = self.Encoder(x)
        flow = self.Decoder(x_sag, x_tra)
        warped = self.SpatialTransformer(x[:,0,:,:,:].unsqueeze(1), flow)
        return warped, flow, x_class
    
class bpdc_densenet_fpran264(nn.Module):
    
    def __init__(self, 
                 in_channels: int = 1, 
                 enc_channels: int = 128,
                 dec_channels: int = 8,
                 use_checkpoint: bool = False):
        super().__init__()
        
        # self.Encoder = VCDCDenseNet_encode(spatial_dims=3, in_channels=1, out_channels=2)
        self.Encoder = Double_Densenet_BPDC264(spatial_dims=3, in_channels=in_channels, out_channels=enc_channels)
        self.Decoder = Multiview_decoder(in_channels=enc_channels,
                                   channel_num=dec_channels,
                                   use_checkpoint=use_checkpoint)
        
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

        

    def forward(self, x):
        x_class, x_sag, x_tra,  = self.Encoder(x)
        flow = self.Decoder(x_sag, x_tra)
        warped = self.SpatialTransformer(x[:,0,:,:,:].unsqueeze(1), flow)
        return warped, flow, x_class
# class summary_model:
#     # model = vcdcdensenet_mlp() 
#     # model = models.resnet50()
#     model = VCDCDenseNet_encode(spatial_dims=3, in_channels=1, out_channels=2)
#     model1 = MLP_decoder(in_channels=128, channel_num=8, use_checkpoint=True)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"模型总参数量: {total_params}")

#     total_params = sum(p.numel() for p in model1.parameters())
#     print(f"模型总参数量: {total_params}")

#     summary(model, input_size=(1, 3, 320, 320, 32))
