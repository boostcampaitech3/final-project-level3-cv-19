#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import math 
import torch.nn.functional as F

from .network_blocks import Bottleneck, SE_Block, get_activation, Focus



class DepthwiseConvBn(nn.Module):
    def __init__(self, in_channels, kernels_per_layer, kernel_size=3, stride=1, act="silu"):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer,
                                   kernel_size=kernel_size, padding=1, groups=in_channels, stride=stride)
        self.bn = nn.BatchNorm2d(in_channels * kernels_per_layer)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.depthwise(x)))


class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out  = self.bn(out)
        out = self.act(out)
        return out


# gmodule1 -> DepthwiseConvBn || SE_Block -> gmodule2
class G_bneck(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, exp, stride, SE, ratio=2):
        super(G_bneck, self).__init__()
        self.stride=stride
        self.use_se = SE
        self.gmodule1 = G_module(in_channels=in_channels, kernel_size=1, out_channels=exp, stride=1)
        self.gmodule2 = G_module(in_channels=exp, kernel_size=1, out_channels=out_channels, stride=1)
        
        #params reduce possible
        self.dconv = DepthwiseConvBn(in_channels=exp, kernels_per_layer=1, kernel_size=3, stride=self.stride)
        self.se_block = SE_Block(in_channels=exp ,internal_neurons=16)

    def forward(self, input):
        output = self.gmodule1(input)
        if self.stride == 2:
            output = self.dconv(output)
        if self.use_se:
            output = self.se_block(output)
        output = self.gmodule2(output)
        return output 


class G_module(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride, ratio=2):
        super(G_module, self).__init__()
        self.init_channels = math.ceil(out_channels//ratio)
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=self.init_channels, kernel_size=kernel_size, stride=stride)
        self.dconv = DepthwiseConvBn(self.init_channels, kernels_per_layer=1, kernel_size=3, stride=stride)

    def forward(self, input):
        output = self.conv(input)
        depth_output = self.dconv(output)
        output = torch.cat([output, depth_output], axis=1)
        return output



class _GhostNet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark2", "dark4", "dark5")):
        super(_GhostNet, self).__init__()
        self.out_features = out_features
        
        out_channels = 320
        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        self.stem = Focus(3, base_channels, ksize=3, act="silu")

        # config = in_channels, kernel_size, out_channels, exp, stride, SE
        layer1 = [ 
            [base_channels, 3, base_channels, base_channels, 1, False], 
            [base_channels, 3, 24, 48, 2, False]
        ]
        layer2 = [
            [24, 3, 24, 72, 1, False], [24, 3, 40, 72, 2, True]
        ]
        layer3 = [ 
            #[40, 3, 40, 120, 1, True], [40, 3, 80, 240, 4, False] #245
            [40, 3, 40, 120, 1, True], [40, 3, 40, 120, 1, True],[40, 3, 40, 120, 1, True],[40, 3, 80, 240, 2, False] #234
        ]
        layer4 = [ 
            [80, 3, 80, 200, 1, False], [80, 3, 80, 184, 1, False],
            [80, 3, 80, 184, 1, False], [80, 3, 112, 480, 1, True],
            [112, 3, 112, 672, 1, True], [112, 3, 160, 672, 2, True]
        ]
        layer5 = [ 
            [160, 3, 160, 960, 1, False], [160, 3, 160, 960, 1, True],
            [160, 3, 160, 960, 1, False], [160, 3, 160, 960, 2, True]
        ]

        self.layer1 = self.make_layer(layer1)
        self.layer2 = self.make_layer(layer2)
        self.layer3 = self.make_layer(layer3)
        self.layer4 = self.make_layer(layer4)
        self.layer5 = self.make_layer(layer5)
        

    def forward(self, input):
        outputs = {}
        stem_out = self.stem(input)
        s1 = self.layer1(stem_out) #stage인데 기존 것과 같이쓰기위해 기존 형식을 이용
        outputs["stem"] = s1
        s2 = self.layer2(s1)
        outputs["dark2"] = s2
        s3 = self.layer3(s2)
        outputs["dark3"] = s3
        s4 = self.layer4(s3)
        outputs["dark4"] = s4
        s5 = self.layer5(s4)
        outputs["dark5"] = s5
        return {k: v for k, v in outputs.items() if k in self.out_features}

    def make_layer(self, layers_configs):
        layers = []
        for i, k, o, e, s, se in layers_configs:
            layers.append(G_bneck(
                in_channels=i, kernel_size=k, out_channels=o, exp=e, stride=s, SE=se
            ))
        return nn.Sequential(*layers)
