# -*- coding: utf-8 -*-
import torch as torch
import torch.nn as nn

class Residual_Connected_Unit(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Residual_Connected_Unit, self).__init__()
        self.Increase = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.BR = nn.Sequential(
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        InOut = self.Increase(x)
        ConOut = self.Conv(InOut)
        AddOut = torch.add(InOut, ConOut)
        out = self.BR(AddOut)
        return out


class Pyramid_Aggregation_Unit(nn.Module):
    def __init__(self, ch_in=4, ch_out=1):
        super(Pyramid_Aggregation_Unit, self).__init__()
        self.Pred = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        ConOut = torch.cat((x[0], x[1], x[2], x[3]), dim=1)
        out = self.Pred(ConOut)
        return out


class Dilated_Perception_Unit(nn.Module):
    def __init__(self, ch_in, ch_out, dilations):
        super(Dilated_Perception_Unit, self).__init__()
        self.AtrousConv0 = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, dilation=dilations[0],
                      padding=dilations[0],
                      bias=True)
        )
        self.AtrousConv1 = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, dilation=dilations[1],
                      padding=dilations[1],
                      bias=True)
        )
        self.AtrousConv2 = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, dilation=dilations[2],
                      padding=dilations[2],
                      bias=True)
        )
        self.AtrousConv3 = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, dilation=dilations[3],
                      padding=dilations[3],
                      bias=True)
        )
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        AtOut0 = self.AtrousConv0(x)
        AtOut1 = self.AtrousConv1(x)
        AtOut2 = self.AtrousConv2(x)
        AtOut3 = self.AtrousConv3(x)
        ConvOut = self.Conv(x)
        Out = torch.cat((AtOut0, AtOut1, AtOut2, AtOut3, ConvOut), dim=1)
        return Out


class Up_Pred(nn.Module):
    def __init__(self, ch_in, scale):
        super(Up_Pred, self).__init__()
        self.up_sigmodi = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.UpsamplingBilinear2d(scale_factor=scale),
            # nn.Sigmoid()
        )

    def forward(self, x):
        out = self.up_sigmodi(x)
        return out


class Concate(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Concate, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        concate = torch.cat((x[0], x[1]), dim=1)
        out = self.Conv(concate)
        return out


class MFCNNModel(nn.Module):
    def __init__(self, image_channel=3, class_number=1):
        super(MFCNNModel, self).__init__()
        # Tool
        self.MaxPooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.DropOut = nn.Dropout(0.5)
        # Encode
        self.RCU1 = Residual_Connected_Unit(ch_in=image_channel, ch_out=64)
        self.RCU2 = Residual_Connected_Unit(ch_in=64, ch_out=64)
        self.RCU3 = Residual_Connected_Unit(ch_in=64, ch_out=128)
        self.RCU4 = Residual_Connected_Unit(ch_in=128, ch_out=128)
        self.RCU5 = Residual_Connected_Unit(ch_in=128, ch_out=256)
        self.RCU6 = Residual_Connected_Unit(ch_in=256, ch_out=256)
        self.RCU7 = Residual_Connected_Unit(ch_in=256, ch_out=512)
        self.RCU8 = Residual_Connected_Unit(ch_in=512, ch_out=512)
        self.RCU9 = Residual_Connected_Unit(ch_in=512, ch_out=1024)
        self.RCU10 = Residual_Connected_Unit(ch_in=1024, ch_out=1024)
        self.DPU1 = Dilated_Perception_Unit(ch_in=256, ch_out=256, dilations=[1, 6, 12, 18])
        self.DPU2 = Dilated_Perception_Unit(ch_in=512, ch_out=512, dilations=[1, 4, 8, 12])
        # Decode
        self.PAU = Pyramid_Aggregation_Unit()
        self.UpandPred8 = Up_Pred(ch_in=512, scale=8)
        self.UpandPred4 = Up_Pred(ch_in=256, scale=4)
        self.UpandPred2 = Up_Pred(ch_in=128, scale=2)
        self.UpandPred1 = Up_Pred(ch_in=64, scale=1)
        self.Concate512 = Concate(ch_in=3584, ch_out=512)
        self.Concate256 = Concate(ch_in=1792, ch_out=256)
        self.Concate128 = Concate(ch_in=384, ch_out=128)
        self.Concate64 = Concate(ch_in=192, ch_out=64)
    def forward(self, x):
        # Encode
        o1 = self.RCU1(x)  # 512x512x64
        o2 = self.RCU2(o1)  # 512x512x64
        m1 = self.MaxPooling(o2)  # 256x256x64
        o3 = self.RCU3(m1)  # 256x256x128
        o4 = self.RCU4(o3)  # 128x128x128
        m2 = self.MaxPooling(o4)  # 64x64x128
        o5 = self.RCU5(m2)  # 64x64x256
        o6 = self.RCU6(o5)  # 64x64x256
        m3 = self.MaxPooling(o6)  # 32x32x256
        o7 = self.RCU7(m3)  # 32x32x512
        o8 = self.RCU8(o7)  # 32x32x512
        m4 = self.MaxPooling(o8)  # 16x16x512
        o9 = self.RCU9(m4)  # 16x16x1024
        o10 = self.RCU10(o9)  # 16x16x1024
        drop1 = self.DropOut(o10)  # 16x16x1024

        # Decode
        Up1 = nn.UpsamplingBilinear2d(scale_factor=2)(drop1)
        D2 = self.DPU2(o8)
        cat1 = self.Concate512([D2, Up1])
        pred1 = self.UpandPred8(cat1)

        Up2 = nn.UpsamplingBilinear2d(scale_factor=2)(cat1)
        D1 = self.DPU1(o6)
        cat2 = self.Concate256([D1, Up2])
        pred2 = self.UpandPred4(cat2)

        Up3 = nn.UpsamplingBilinear2d(scale_factor=2)(cat2)
        cat3 = self.Concate128([o4, Up3])
        pred3 = self.UpandPred2(cat3)

        Up4 = nn.UpsamplingBilinear2d(scale_factor=2)(cat3)
        cat4 = self.Concate64([o2, Up4])
        pred4 = self.UpandPred1(cat4)

        out = self.PAU([pred1, pred2, pred3, pred4])

        return [pred1, pred2, pred3, pred4, out]

if __name__ == '__main__':
    model = MFCNNModel()
    from torchstat import stat
    stat(model, (3, 512, 512))


