#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import math 
import torch.nn.functional as F
import numpy as np

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "gelu":
        module = nn.GELU()
    elif name == "hardswish":
        module = nn.Hardswish(inplace=inplace)
    elif name == "mish":
        module = nn.Mish(inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, dilation=1, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2 + (dilation - 1)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Seperable Convolution"""
    """Depthwise Conv + Pointwise Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu", dilation=1):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class DWConv_2(nn.Module):
    """Depthwise Convolution"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu", dilation=1):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            out_channels,
            ksize=ksize,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
            act=act,
        )

    def forward(self, x):
        return self.dconv(x)


class GroupConv(nn.Module):
    """Grouped Convolution"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu", dilation=1, bias=False):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2 + (dilation - 1)
        self.gconv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            dilation=dilation,
            # groups=in_channels,
            groups=8,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.gconv(x)))

    def fuseforward(self, x):
        return self.act(self.gconv(x))


""" Mixed Depthwise Convolution Starts """

def split_layer(total_channels, num_groups):
    split = [int(np.ceil(total_channels / num_groups)) for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split


class DepthwiseConv2D(nn.Module):
    """Depthwise Conv for Mixed Depthwise Convolution"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, bias=False):
        super().__init__()
        padding = (ksize - 1) // 2

        self.depthwise_conv = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size=ksize,
            padding=padding,
            stride=stride,
            groups=in_channels,
            bias=bias
        )

    def forward(self, x):
        out = self.depthwise_conv(x)
        return out


class GroupConv2D(nn.Module):
    """Group Conv for Mixed Depthwise Convolution"""

    def __init__(self, in_channels, out_channels, ksize=1, n_chunks=1, bias=False):
        super(GroupConv2D, self).__init__()
        self.n_chunks = n_chunks
        self.split_in_channels = split_layer(in_channels, n_chunks)
        split_out_channels = split_layer(out_channels, n_chunks)

        if n_chunks == 1:
            self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, bias=bias)
        else:
            self.group_layers = nn.ModuleList()
            for idx in range(n_chunks):
                self.group_layers.append(nn.Conv2d(
                    self.split_in_channels[idx], 
                    split_out_channels[idx], 
                    kernel_size=ksize,
                    bias=bias
                ))

    def forward(self, x):
        if self.n_chunks == 1:
            return self.group_conv(x)
        else:
            split = torch.split(x, self.split_in_channels, dim=1)
            out = torch.cat([layer(s) for layer, s in zip(self.group_layers, split)], dim=1)
            return out


class MDConv(nn.Module):
    """Mixed Depthwise Convolution"""

    def __init__(self, in_channels, out_channels, n_chunks, stride=1, bias=False):
        super(MDConv, self).__init__()
        self.n_chunks = n_chunks
        self.split_in_channels = split_layer(in_channels, n_chunks)
        self.split_out_channels = split_layer(out_channels, n_chunks)

        self.layers = nn.ModuleList()
        for idx in range(self.n_chunks):
            ksize = 2 * idx + 3
            self.layers.append(
                DepthwiseConv2D(
                    self.split_in_channels[idx],
                    self.split_out_channels[idx],
                    ksize=ksize,
                    stride=stride,
                    bias=bias
                )
            )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation("silu", inplace=True)

    def forward(self, x):
        split = torch.split(x, self.split_in_channels, dim=1)
        out = torch.cat([layer(s) for layer, s in zip(self.layers, split)], dim=1)
        return self.act(self.bn(out))
        
""" Mixed Depthwise Convolution Ends """


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        n_chunks=1,
        mdconv=False
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        # Conv = DWConv_2 if depthwise else BaseConv
        # Conv = GroupConv if depthwise else BaseConv
        self.modified_ksize = 3
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        if mdconv:
            self.conv2 = MDConv(hidden_channels, out_channels, n_chunks, 1)
        else:
            self.conv2 = Conv(hidden_channels, out_channels, self.modified_ksize, stride=1, act=act)

        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        n_chunks=1,
        mdconv=False
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act, n_chunks=n_chunks, mdconv=mdconv
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x) 
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, dilation=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, dilation, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

class SE_Block(nn.Module):
    def __init__(self, in_channels, internal_neurons):
        super(SE_Block, self).__init__()
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=in_channels, kernel_size=1, stride=1, bias=True)
        self.in_channels = in_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.in_channels, 1, 1)
        return inputs * x