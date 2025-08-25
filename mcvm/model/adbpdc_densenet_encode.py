# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils.module import look_up_option
from .bpdc import adbpdc_3d,bpdc_3d
import random



__all__ = [
    "DenseNet",
    "Densenet",
    "DenseNet121",
    "densenet121",
    "Densenet121",
    "DenseNet169",
    "densenet169",
    "Densenet169",
    "DenseNet201",
    "densenet201",
    "Densenet201",
    "DenseNet264",
    "densenet264",
    "Densenet264",
]


class _DenseLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_prob: float,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            growth_rate: how many filters to add each layer (k in paper).
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.layers.add_module("relu1", get_act_layer(name=act))
        self.layers.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))



        self.layers.add_module("norm2", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels))
        self.layers.add_module("relu2", get_act_layer(name=act))
        self.layers.add_module("conv2", adbpdc_3d(out_channels, growth_rate, kernel_size=3, stride=1,padding=1, dilation=1, groups=1, bias=False, theta=0.7))

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            layers: number of layers in the block.
            in_channels: number of the input channel.
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            growth_rate: how many filters to add each layer (k in paper).
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob, act=act, norm=norm)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            out_channels: number of the output classes.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.add_module("relu", get_act_layer(name=act))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))


class ADBPDC_DenseNet_encode(nn.Module):
    """
    Densenet based on: `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.
    This network is non-deterministic When `spatial_dims` is 3 and CUDA is enabled. Please check the link below
    for more details:
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        avg_pool_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[Pool.ADAPTIVEAVG, spatial_dims]
        self.upsample = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False)
        self.features0 = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=(2,2,1), padding=1)),
                ]
            )
        )
        self.features1 = nn.Sequential()
        self.features2 = nn.Sequential()
        self.features3 = nn.Sequential()
        self.features4 = nn.Sequential()


        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            if i == 0:
                self.features1.add_module(f"denseblock{i + 1}", block)
                in_channels += num_layers * growth_rate
                _out_channels = in_channels // 2
                trans = _Transition(spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm)
                self.features1.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels
            if i == 1:
                self.features2.add_module(f"denseblock{i + 1}", block)
                in_channels += num_layers * growth_rate
                _out_channels = in_channels // 2
                trans = _Transition(spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm)
                self.features2.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels
            if i == 2:
                self.features3.add_module(f"denseblock{i + 1}", block)
                in_channels += num_layers * growth_rate
                _out_channels = in_channels // 2
                trans = _Transition(spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm)
                self.features3.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels
            if i == 3:
                self.features4.add_module(f"denseblock{i + 1}", block)
                in_channels += num_layers * growth_rate
                _out_channels = in_channels // 2
                trans = _Transition(spatial_dims, in_channels=in_channels, out_channels=out_channels, act=act, norm=norm)
                self.features4.add_module(f"transition{i + 1}", trans)
                # self.features4.add_module("norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
           

        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)),
                    ("out", nn.Linear(in_channels, out_channels)),
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mri = torch.rot90(mri.transpose(2,4), k=1, dims=[3,4]).detach() # tra -> sag
        
        
        x_0 = self.features0(x)
        x_1 = self.features1(x_0)
        x_2 = self.features2(x_1)
        x_3 = self.features3(x_2)
        x_4 = self.features4(x_3)
       
        # torch.Size([2, 128, 40, 40, 8]) torch.Size([2, 256, 20, 20, 4]) torch.Size([2, 1024, 5, 5, 1]) torch.Size([2, 512, 10, 10, 2])
        return [x_1, x_2, x_3, x_4],x_4


def _load_state_dict(model: nn.Module, arch: str, progress: bool):
    """
    This function is used to load pretrained models.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.

    """
    model_urls = {
        "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
        "densenet169": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
        "densenet201": "https://download.pytorch.org/models/densenet201-c1103571.pth",
    }
    model_url = look_up_option(arch, model_urls, None)
    if model_url is None:
        raise ValueError(
            "only 'densenet121', 'densenet169' and 'densenet201' are supported to load pretrained weights."
        )

    pattern = re.compile(
        r"^(.*denselayer\d+)(\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + ".layers" + res.group(2) + res.group(3)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    model_dict = model.state_dict()
    state_dict = {
        k: v for k, v in state_dict.items() if (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)
    }
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

# class BPDC_DenseNet_encode264(BPDC_DenseNet_encode):
#     """DenseNet264"""

#     def __init__(
#         self,
#         spatial_dims: int,
#         in_channels: int,
#         out_channels: int,
#         init_features: int = 64,
#         growth_rate: int = 32,
#         block_config: Sequence[int] = (6, 12, 64, 48),
#         pretrained: bool = False,
#         progress: bool = True,
#         **kwargs,
#     ) -> None:
#         super().__init__(
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             init_features=init_features,
#             growth_rate=growth_rate,
#             block_config=block_config,
#             **kwargs,
#         )
#         if pretrained:
#             raise NotImplementedError("Currently PyTorch Hub does not provide densenet264 pretrained models.")
# x = torch.randn(1, 1, 320, 320, 32).cuda()
# model = DenseNet_encode(spatial_dims=3, in_channels=1, out_channels=2).cuda()
# model(x)






