# !/usr/bin/python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as torch

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class encodeBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(encodeBlock, self).__init__()
        self.doubleConv = conv_block(ch_in, ch_out)
        self.MaxPooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        rConv = self.doubleConv(x)
        rPool = self.MaxPooling(rConv)
        return rConv, rPool

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(2, 2), stride=(2, 2))
        )
        self.BnRelu = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        result = self.up(x[0])
        result = torch.cat([result, x[1]], dim=1)
        result = self.BnRelu(result)
        return result

class DilaConv(nn.Module):
    def __init__(self, ch_in, ch_out, rate):
        super(DilaConv, self).__init__()
        self.dilaConv = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3), stride=(1, 1), padding=rate,
                      dilation=rate),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        rConv = self.dilaConv(x)
        return rConv


class center_block(nn.Module):
    def __init__(self, ch_in, ch_out, rate):
        super(center_block, self).__init__()
        self.dilateConv0 = DilaConv(ch_in, ch_out, rate[0])
        self.dilateConv1 = DilaConv(ch_out, ch_out, rate[1])
        self.dilateConv2 = DilaConv(ch_out, ch_out, rate[2])
        self.dilateConv3 = DilaConv(ch_out, ch_out, rate[3])
        self.dilateConv4 = DilaConv(ch_out, ch_out, rate[4])
        self.dilateConv5 = DilaConv(ch_out, ch_out, rate[5])

    def forward(self, x):
        cent0 = self.dilateConv0(x)
        cent1 = self.dilateConv1(cent0)
        cent2 = self.dilateConv2(cent1)
        cent3 = self.dilateConv3(cent2)
        cent4 = self.dilateConv4(cent3)
        cent5 = self.dilateConv5(cent4)
        result = cent0 + cent1 + cent2 + cent3 + cent4 + cent5
        return result

class rrm_ours(nn.Module):
    def __init__(self, ch_in, ch_out, rate):
        super(rrm_ours, self).__init__()
        self.dilateConv0 = DilaConv(ch_in, 64, rate[0])
        self.dilateConv1 = DilaConv(64, 64, rate[1])
        self.dilateConv2 = DilaConv(64, 64, rate[2])
        self.dilateConv3 = DilaConv(64, 64, rate[3])
        self.dilateConv4 = DilaConv(64, 64, rate[4])
        self.dilateConv5 = DilaConv(64, 64, rate[5])
        self.Conv = nn.Conv2d(64, ch_out, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        # self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        cent0 = self.dilateConv0(x)
        cent1 = self.dilateConv1(cent0)
        cent2 = self.dilateConv2(cent1)
        cent3 = self.dilateConv3(cent2)
        cent4 = self.dilateConv4(cent3)
        cent5 = self.dilateConv5(cent4)
        rConv = self.Conv(cent0 + cent1 + cent2 + cent3 + cent4 + cent5)
        result = rConv + x
        # result = self.Sigmoid(result)
        return result

class PredOnly(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(PredOnly, self).__init__()
        self.pred = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(1, 1), stride=(1, 1), padding=0),
            # nn.Sigmoid()
        )

    def forward(self, x):
        pred = self.pred(x)
        return pred
