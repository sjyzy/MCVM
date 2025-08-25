import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np

class bpdc_3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(bpdc_3d, self).__init__() 
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        
        self.mask = torch.tensor([[
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]],
            [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]],
            [[0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]]).to('cuda')
    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight * self.mask
            kernel_diff = kernel_diff.sum(2).sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None, None]
            out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias,stride=self.conv.stride, padding=0, groups=self.conv.groups)
            return out_normal - self.theta * out_diff


class adbpdc_3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(adbpdc_3d, self).__init__() 
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        
        mask = torch.tensor([[
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]],
            [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]],
            [[0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]], dtype=torch.float32)
        self.register_buffer('mask', mask)
        self.fusion3d = LocalDiffAdaptiveFusion3D(kernel_size=3)
    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight * self.mask
            kernel_diff = kernel_diff.sum(2).sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None, None]
            out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias,stride=self.conv.stride, padding=0, groups=self.conv.groups)
            output, alpha_map = self.fusion3d(x, out_normal, out_diff)
            return output
            
class LocalDiffAdaptiveFusion3D(nn.Module):
    def __init__(self, kernel_size=3):
        super(LocalDiffAdaptiveFusion3D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        # 1x1x1卷积用于增强表达能力
        self.alpha_conv = nn.Conv3d(1, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, conv_out, diff_out):
        # x: 原始输入，[B, C, D, H, W]
        # conv_out, diff_out: 普通/差分卷积输出
        # Step 1：计算局部梯度（采用Sobel 3D核或简单的局部差分）
        grad_x = F.conv3d(x, weight=self.get_sobel_kernel3d(x.device, x.shape[1]), padding=1, groups=x.shape[1]) # sobel核大小为
        grad_mag = torch.abs(grad_x)  # 梯度绝对值
        grad_mean = torch.mean(grad_mag, dim=1, keepdim=True)  # [B, 1, D, H, W]
        
        # Step 2：归一化梯度，1x1x1卷积后sigmoid
        alpha = self.alpha_conv(grad_mean)
        alpha = self.sigmoid(alpha)  # [B, 1, D, H, W]

        # Step 3：自适应融合
        out = conv_out + (1 - alpha) * diff_out
        print(alpha)
        return out, alpha

    @staticmethod
    def get_sobel_kernel3d(device, channels):
        # 这里只对z轴做梯度，可根据需要自行拓展
        sobel_z = torch.tensor([[[1, 2, 1],
                                 [2, 4, 2],
                                 [1, 2, 1]],
                                [[0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0]],
                                [[-1, -2, -1],
                                 [-2, -4, -2],
                                 [-1, -2, -1]]], dtype=torch.float32, device=device)
        sobel_z = sobel_z.unsqueeze(0).unsqueeze(0)  # [1,1,3,3,3]
        sobel_kernel = sobel_z.repeat(channels, 1, 1, 1, 1)  # [C,1,3,3,3]
        return sobel_kernel