import torch
import torch.nn as nn
from GBSCM import GBSCM
from PKSRM import PKSRM

class ConvBR(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, mode=1, groups=1):
        super(ConvBR, self).__init__()
        self.mode = mode
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel, kernel), stride=(stride, stride),
                              padding=(kernel//2, kernel//2), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.mode == 1:
            out = self.relu(self.bn(self.conv(x)))
        elif self.mode == 2:
            out = self.bn(self.conv(x))
        elif self.mode == 3:
            out = self.relu(self.bn(x))
        else:
            out = self.conv(x)

        return out

class MSCFA(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(MSCFA, self).__init__()
        self.conv3 = ConvBR(in_planes, out_planes // 2, mode=2)
        self.conv3_relu = nn.ReLU(inplace=True)
        self.conv5 = ConvBR(out_planes // 2, out_planes // 2, mode=2)
        self.res = ConvBR(in_planes, out_planes // 2, kernel=1, mode=2)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        if self.stride == 2:
            self.conv7 = ConvBR(out_planes // 2, out_planes // 4, stride=2)
            self.conv9 = ConvBR(out_planes // 4, out_planes // 4)
            self.skip = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        else:
            self.conv7 = ConvBR(out_planes // 2, out_planes // 4)
            self.conv9 = ConvBR(out_planes // 4, out_planes // 4)

    def forward(self, x):
        conv3 = self.conv3(x)
        conv3_relu = self.conv3_relu(conv3)
        conv5 = self.conv5(conv3_relu)
        if x.size() != conv5.size():
            x = self.res(x)
        if self.stride == 1:
            convs = self.relu(x + conv3 + conv5)
            conv7 = self.conv7(convs)
            conv9 = self.conv9(conv7)
        else:
            convs = self.relu(x + conv3 + conv5)
            conv7 = self.conv7(convs)
            conv9 = self.conv9(conv7)
            convs = self.skip(convs)

        out = torch.cat([convs, conv7, conv9], dim=1)

        return out

class SERNet(nn.Module):
    def __init__(self, inch=3, classes=1):
        super(SERNet, self).__init__()
        self.Encode1 = nn.Sequential(MSCFA(inch, 64), MSCFA(64, 128, stride=2))
        self.Encode2 = nn.Sequential(MSCFA(128, 128), MSCFA(128, 256, stride=2))
        # 空间细节分支
        self.Spatial_branch = nn.Sequential(MSCFA(256, 512, stride=2), MSCFA(512, 512), MSCFA(512, 512))
        # 上下文语义分支
        self.Context_branch = nn.Sequential(MSCFA(256, 256, stride=2), MSCFA(256, 256, stride=2), MSCFA(256, 512, stride=2))

        self.PKSRM256 = PKSRM(256, 4)
        self.PKSRM128 = PKSRM(128, 2)

        self.GBSCM512 = GBSCM(512, 8)
        self.GBSCM256 = GBSCM(256, 4)
        self.GBSCM128 = GBSCM(128, 2)

        self.Decode3 = nn.Sequential(MSCFA(512, 256))
        self.Decode2 = nn.Sequential(MSCFA(256, 128))
        self.Decode1 = nn.Sequential(MSCFA(128, 64))

        self.seghead = ConvBR(64, classes, kernel=1, mode=4)

    def forward(self, x):
        # 浅层特征
        Encode1 = self.Encode1(x)  # out 1/2 128
        Encode2 = self.Encode2(Encode1)  # out 1/4 256
        Spatial_branch = self.Spatial_branch(Encode2)  # out 1/8 512
        Context_branch = self.Context_branch(Encode2)  # out 1/8 256

        fusions8 = self.GBSCM512(Spatial_branch, Context_branch)
        Decode3 = self.Decode3(fusions8)  # out 1/8 256

        PKSRM256 = self.PKSRM256(Encode2)
        fusions4 = self.GBSCM256(PKSRM256, Decode3)
        Decode2 = self.Decode2(fusions4)  # out 1/4 128

        PKSRM128 = self.PKSRM128(Encode1)
        fusions2 = self.GBSCM128(PKSRM128, Decode2)
        Decode1 = self.Decode1(fusions2)  # out 1/2 64

        feature1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)(Decode1)  # out 1/1 64
        segout = self.seghead(feature1)

        return segout

if __name__ == '__main__':
    model = SERNet()
    from torchstat import stat
    stat(model, (3, 512, 512))