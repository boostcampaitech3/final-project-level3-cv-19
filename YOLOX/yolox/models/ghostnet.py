#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import math 
import torch.nn.functional as F

from .network_blocks import Bottleneck, SE_Block, get_activation, Focus


__all__ = ['ghost_net']

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se): 
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostNet(nn.Module):
    def __init__(self, cfgs, out_features, width_mult=1., ):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.width_mult = width_mult
        self.out_features = out_features

        # building first layer
        self.output_channel = _make_divisible(16 * self.width_mult, 4)
        self.layers = [nn.Sequential(
            nn.Conv2d(3, self.output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.output_channel),
            nn.ReLU(inplace=True)
        )]

        self.input_channel = self.output_channel
        
        self.layer1 = self.make_layer(self.cfgs[0])
        self.layers = []
        self.layer2 = self.make_layer(self.cfgs[1])
        self.layers = []
        self.layer3 = self.make_layer(self.cfgs[2])   
        self.layers = []

        self._initialize_weights()

    def forward(self, x):
        outputs = {}
        x1 = self.layer1(x) 
        outputs["layer1"] = x1
        x2 = self.layer2(x1) 
        outputs["layer2"] = x2
        x3 = self.layer3(x2)
        outputs["layer3"] = x3

        return {k: v for k, v in outputs.items() if k in self.out_features}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, layers_configs):
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in layers_configs: #kernel, exp_size, SEblock, stride
            self.output_channel = _make_divisible(c * self.width_mult, 4)
            hidden_channel = _make_divisible(exp_size * self.width_mult, 4)
            self.layers.append(block(self.input_channel, hidden_channel, self.output_channel, k, s, use_se))
            self.input_channel = self.output_channel
        return nn.Sequential(*self.layers)


def ghost_net(**kwargs):
    """
    Constructs a GhostNet model
    """
    layer1 = [
        # k, t, c, SE, s 
        [3,  16,  16, 0, 1],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [3,  72,  40, 1, 2],
        [3,  120,  40, 1, 1],
        [3,  160,  80, 1, 1]
    ]
    layer2 = [
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480,  112, 1, 1], 
        [3, 672,  112, 1, 1], 
        [3, 672,  160, 1, 2], 
    ]
    layer3 = [
        [3, 960, 160, 0, 1],
        [3, 960, 160, 1, 1],
        [3, 960, 160, 0, 1],
        [3, 960, 160, 1, 2], 
        
        # # L3_rm4 case
        # [5, 480, 112, 1, 1],
        # [5, 672, 112, 1, 1],
        # [5, 672, 160, 1, 2], # ('Params: 1.17M, Gflops: 1.08', 1.167871, 1.076669594)
        
        # # L3_rm2 case
        # [5, 480, 112, 1, 1],
        # [5, 672, 112, 1, 1],
        # [5, 672, 160, 1, 2],
        # [5, 960, 160, 0, 1],
        # [5, 960, 160, 1, 1], # ('Params: 1.95M, Gflops: 1.22', 1.951631, 1.224767674)
        
        # # L3_full case
        # [3, 480, 112, 1, 1],
        # [3, 672, 112, 1, 1],
        # [5, 672, 160, 1, 2],
        # [5, 960, 160, 0, 1],
        # [5, 960, 160, 1, 1],
        # [5, 960, 160, 0, 1], 
        # [5, 960, 160, 1, 1]  # ('Params: 2.74M, Gflops: 1.37', 2.735391, 1.372865754)
    ]
    cfgs = [
        layer1, layer2, layer3
    ]
    return GhostNet(cfgs, **kwargs)