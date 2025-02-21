 # -*- coding: utf-8 -*-

import torch.nn as nn
import torch as torch

from Contrast_Net.MAFCN.MA_FCN_Ulits import Down_Double_Conv, Up_Double_Conv, Up_Three_Conv, Down_Three_Conv

class MA_FCN(nn.Module):
    def __init__(self, image_channel=3, class_number=1):
        super(MA_FCN, self).__init__()
        # tool
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.UpSample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pred = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0, bias=True)

        # Encode
        self.Down_Conv_Double1 = Down_Double_Conv(ch_in=image_channel, ch_out=64)
        self.Down_Conv_Double2 = Down_Double_Conv(ch_in=64, ch_out=128)
        self.Down_Conv_Three1 = Down_Three_Conv(ch_in=128, ch_out=256)
        self.Down_Conv_Three2 = Down_Three_Conv(ch_in=256, ch_out=512)
        self.Down_Conv_Three3 = Down_Three_Conv(ch_in=512, ch_out=512)

        # Decode
        self.Up_Conv_Three1 = Up_Three_Conv(ch_in=1024, ch_out=256)
        self.Up_Conv_Three2 = Up_Three_Conv(ch_in=512, ch_out=128)
        self.Up_Conv_Three3 = Up_Three_Conv(ch_in=256, ch_out=64)
        self.Up_Conv_Tow = Up_Double_Conv(ch_in=128, ch_out=class_number)

    def forward(self, x):
        # Encode
        d1 = self.Down_Conv_Double1(x)
        m1 = self.Maxpool(d1)
        d2 = self.Down_Conv_Double2(m1)
        m2 = self.Maxpool(d2)
        d3 = self.Down_Conv_Three1(m2)
        m3 = self.Maxpool(d3)
        d4 = self.Down_Conv_Three2(m3)
        m4 = self.Maxpool(d4)
        d5 = self.Down_Conv_Three3(m4)
        # Decode
        up1 = nn.UpsamplingBilinear2d(scale_factor=2)(d5)
        e1 = torch.cat((d4, up1), dim=1)
        f1, s1 = self.Up_Conv_Three1(e1)  # 一个传递下一层，一个用于输出
        s1 = nn.UpsamplingBilinear2d(scale_factor=8)(s1)

        up2 = nn.UpsamplingBilinear2d(scale_factor=2)(f1)
        e2 = torch.cat((d3, up2), dim=1)
        f2, s2 = self.Up_Conv_Three2(e2)
        s2 = nn.UpsamplingBilinear2d(scale_factor=4)(s2)

        up3 = nn.UpsamplingBilinear2d(scale_factor=2)(f2)
        e3 = torch.cat((d2, up3), dim=1)
        f3, s3 = self.Up_Conv_Three3(e3)
        s3 = nn.UpsamplingBilinear2d(scale_factor=2)(s3)

        up4 = nn.UpsamplingBilinear2d(scale_factor=2)(f3)
        e4 = torch.cat((d1, up4), dim=1)
        s4 = self.Up_Conv_Tow(e4)

        total = torch.cat((s1, s2, s3, s4), dim=1)
        total = self.pred(total)

        # return [s1, s2, s3, s4, total]
        return total

if __name__ == '__main__':
    from torchstat import stat
    num_class = 1
    model = MA_FCN(3, num_class)
    stat(model, (3, 512, 512))
