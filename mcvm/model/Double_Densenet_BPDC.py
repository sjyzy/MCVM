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
# from .vcdcdensenet_encode import VCDCDenseNet_encode
# from .decode import MLP_decoder, SpatialTransformer_block
from .densenet_encode import DenseNet_encode
from .bpdc_densenet_encode import BPDC_DenseNet_encode,BPDC_DenseNet_encode264
from .adbpdc_densenet_encode import ADBPDC_DenseNet_encode

from torchinfo import summary
import torchvision.models as models
import monai
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from collections import OrderedDict
import torch.nn.functional as F

class Double_Densenet_BPDC(nn.Module):
    
    def __init__(self, 
                 spatial_dims: int = 3,
                 in_channels: int = 1, 
                 out_channels: int = 2,
                 mid_channels: int = 512,
                 ):
        super().__init__()
        # self.Encoder = monai.networks.nets.DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=mid_channels)
        self.Encoder1 = BPDC_DenseNet_encode(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=mid_channels)
        self.Encoder2 = BPDC_DenseNet_encode(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=mid_channels)
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Linear(2*mid_channels, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels)
        )

        self.fusion_module = FusionNet(in_channels=mid_channels, num_classes=out_channels, kernel_size=7)
        self.mfb_fusion = MFBFusion(in_channels=mid_channels, out_dim=512, k=5, dropout_rate=0.1)

    
        

    def forward(self, x):
        x1 = x[:,0,:,:,:].unsqueeze(1)
        x2 = x[:,1,:,:,:].unsqueeze(1)
        feature_1, x1_4 = self.Encoder1(x1) # torch.Size([2, 128, 40, 40, 8]) torch.Size([2, 256, 20, 20, 4]) torch.Size([2, 1024, 5, 5, 1]) torch.Size([2, 512, 10, 10, 2])
        feature_2, x2_4 = self.Encoder2(x2)
      
        # 线性层
        x = torch.cat((x1_4, x2_4), 1)
  
        x = self.classifier1(x)
        # 空间注意力
        # x = self.fusion_module(x1_4, x2_4)
        # MFB融合
        # x = self.mfb_fusion(x1_4, x2_4)
        return (x,feature_1, feature_2)

class Double_Densenet_ADBPDC(nn.Module):
    
    def __init__(self, 
                 spatial_dims: int = 3,
                 in_channels: int = 1, 
                 out_channels: int = 2,
                 mid_channels: int = 512,
                 ):
        super().__init__()
        # self.Encoder = monai.networks.nets.DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=mid_channels)
        self.Encoder1 = ADBPDC_DenseNet_encode(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=mid_channels)
        self.Encoder2 = ADBPDC_DenseNet_encode(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=mid_channels)
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Linear(2*mid_channels, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels)
        )

        self.fusion_module = FusionNet(in_channels=mid_channels, num_classes=out_channels, kernel_size=7)
        self.mfb_fusion = MFBFusion(in_channels=mid_channels, out_dim=512, k=5, dropout_rate=0.1)

    def forward(self, x):
        x1 = x[:,0,:,:,:].unsqueeze(1)
        x2 = x[:,1,:,:,:].unsqueeze(1)
        feature_1, x1_4 = self.Encoder1(x1) # torch.Size([2, 128, 40, 40, 8]) torch.Size([2, 256, 20, 20, 4]) torch.Size([2, 1024, 5, 5, 1]) torch.Size([2, 512, 10, 10, 2])
        feature_2, x2_4 = self.Encoder2(x2)
        # 线性层
        x = torch.cat((x1_4, x2_4), 1)
        x = self.classifier1(x)
        # 空间注意力
        # x = self.fusion_module(x1_4, x2_4)
        # MFB融合
        # x = self.mfb_fusion(x1_4, x2_4)
        return (x,feature_1, feature_2)

class Double_Densenet_BPDC264(nn.Module):
    
    def __init__(self, 
                 spatial_dims: int = 3,
                 in_channels: int = 1, 
                 out_channels: int = 2,
                 mid_channels: int = 512,
                 ):
        super().__init__()
        # self.Encoder = monai.networks.nets.DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=mid_channels)
        self.Encoder = BPDC_DenseNet_encode264(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=mid_channels)
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Linear(2*mid_channels, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels)
        )

        self.fusion_module = FusionNet(in_channels=mid_channels, num_classes=out_channels, kernel_size=7)
        self.mfb_fusion = MFBFusion(in_channels=mid_channels, out_dim=512, k=5, dropout_rate=0.1)

    
        

    def forward(self, x):
        x1 = x[:,0,:,:,:].unsqueeze(1)
        x2 = x[:,1,:,:,:].unsqueeze(1)
        feature_1, x1_4 = self.Encoder(x1) # torch.Size([2, 128, 40, 40, 8]) torch.Size([2, 256, 20, 20, 4]) torch.Size([2, 1024, 5, 5, 1]) torch.Size([2, 512, 10, 10, 2])
        feature_2, x2_4 = self.Encoder(x2)
      
        # 线性层
        x = torch.cat((x1_4, x2_4), 1)
  
        x = self.classifier1(x)
        # 空间注意力
        # x = self.fusion_module(x1_4, x2_4)
        # MFB融合
        # x = self.mfb_fusion(x1_4, x2_4)
        return (x,feature_1, feature_2)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        padding = kernel_size // 2
        # 输入2通道（平均池化和最大池化结果），输出1通道的空间注意力图
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
    
    def forward(self, x):
        # x: [B, C, W, H, D]
        avg_out = torch.mean(x, dim=1, keepdim=True)   # [B, 1, W, H, D]
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # [B, 1, W, H, D]
        x_cat = torch.cat([avg_out, max_out], dim=1)       # [B, 2, W, H, D]
        attention = torch.sigmoid(self.conv(x_cat))        # [B, 1, W, H, D]
        return x * attention

class FusionNet(nn.Module):
    def __init__(self, in_channels, num_classes=2, kernel_size=7):
        super(FusionNet, self).__init__()
        # 将两个骨干特征在通道维度拼接后，通过1×1×1卷积将通道数降回in_channels
        self.fuse_conv = nn.Conv3d(2 * in_channels, in_channels, kernel_size=1)
        # 空间注意力模块
        self.spatial_attention = SpatialAttentionModule(kernel_size=kernel_size)
        # 分类头：全局平均池化 + 全连接层输出2个类别
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, feat1, feat2):
        # feat1, feat2: [B, C, W, H, D]
        fused = torch.cat([feat1, feat2], dim=1)  # [B, 2C, W, H, D]
        fused = self.fuse_conv(fused)             # [B, C, W, H, D]
        attended = self.spatial_attention(fused)  # [B, C, W, H, D]
        pooled = self.global_avg_pool(attended).view(attended.size(0), -1)  # [B, C]
        out = self.fc(pooled)                     # [B, 2]
        return out
    
class MFBFusion(nn.Module):
    def __init__(self, in_channels, out_dim, k=5, dropout_rate=0.1):
        """
        Args:
            in_channels: 输入特征的通道数（两个模态均为 C）
            out_dim: MFB 融合后输出的低维特征数
            k: 分解因子，决定投影后通道数为 k*out_dim
            dropout_rate: 元素乘法后的 dropout 率
        """
        super(MFBFusion, self).__init__()
        self.out_dim = out_dim
        self.k = k
        
        # 对两个模态分别进行1x1x1卷积，将通道数提升到 k*out_dim
        self.conv1 = nn.Conv3d(in_channels, k * out_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv3d(in_channels, k * out_dim, kernel_size=1, bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)
        # 全局平均池化，将 (W,H,D) 尺寸压缩到 1
        self.pool = nn.AdaptiveAvgPool3d(1)
        # 最后二分类全连接层（输出2个类别的logits）
        self.fc = nn.Linear(out_dim, 2)
        
    def forward(self, x1, x2):
        """
        Args:
            x1, x2: 分别来自两个模态的特征，形状为 (B, C, W, H, D)
        Returns:
            二分类的 logits，形状为 (B, 2)
        """
        # 分别对两个模态做 1x1x1 卷积变换
        x1_proj = self.conv1(x1)  # shape: (B, k*out_dim, W, H, D)
        x2_proj = self.conv2(x2)  # shape: (B, k*out_dim, W, H, D)
        
        # 元素级相乘（对应位置相乘）
        joint_feature = x1_proj * x2_proj  # (B, k*out_dim, W, H, D)
        joint_feature = self.dropout(joint_feature)
        
        # 将通道数分解成 (out_dim, k)
        B, total_channels, W, H, D = joint_feature.size()
        # reshape 成 (B, out_dim, k, W, H, D)
        joint_feature = joint_feature.view(B, self.out_dim, self.k, W, H, D)
        # 在 k 维上做求和池化
        joint_feature = torch.sum(joint_feature, dim=2)  # (B, out_dim, W, H, D)
        
        # power normalization：符号开方，保证梯度连续
        joint_feature = torch.sign(joint_feature) * torch.sqrt(torch.abs(joint_feature) + 1e-10)
        # L2 归一化（在通道维度上归一化）
        joint_feature = F.normalize(joint_feature, p=2, dim=1)
        
        # 全局平均池化：将 (W,H,D) 维度池化到 1
        pooled_feature = self.pool(joint_feature)  # (B, out_dim, 1, 1, 1)
        pooled_feature = pooled_feature.view(B, self.out_dim)  # (B, out_dim)
        
        # 二分类全连接层输出 logits
        out = self.fc(pooled_feature)  # (B, 2)
        return out

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
if __name__ == '__main__':
    model = Double_Densenet()
    x = torch.randn(2, 2, 320, 320, 32)
    y = model(x)
    # print(y.shape)
    # print(model)
    # summary(model, input_size=(2, 32, 320, 320))