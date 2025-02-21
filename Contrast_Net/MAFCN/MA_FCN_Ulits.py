# -*- coding: utf-8 -*-

import torch.nn as nn

class Down_Double_Conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.Conv(x)
        return out


class Down_Three_Conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.Conv(x)
        return out


class Up_Three_Conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.ConvFirst = nn.Sequential(
            nn.Conv2d(ch_in, ch_out * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out * 2, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.ConvSecond = nn.Sequential(
            nn.Conv2d(ch_out, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        First_Out = self.ConvFirst(x)
        Second_Out = self.ConvSecond(First_Out)
        return First_Out, Second_Out


class Up_Double_Conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_in/2), kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(int(ch_in/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(ch_in/2), int(ch_in/4), kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(int(ch_in/4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(ch_in/4), ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        Out = self.Conv(x)
        return Out
