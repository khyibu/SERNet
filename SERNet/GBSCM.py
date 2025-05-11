import torch.nn as nn
from torch.nn import functional as F
import torch

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

class GBSCM(nn.Module):
    def __init__(self,inch, group):
        super(GBSCM, self).__init__()
        self.groups = group
        self.cp = ConvBR(inch, inch//2, kernel=1)
        self.sp = ConvBR(inch, inch//2, kernel=1)
        self.redim = nn.Sequential(nn.Conv2d(inch,  inch//2, 1, 1, 0))

        self.direction = nn.Conv2d(group, 4*group, 3, 1, 1)
        self.distance = nn.Conv2d(inch//2, 4 * self.groups, 3, 1, 1)

        self.conv_offset = nn.Conv2d(int(inch//2 + 8 * self.groups), 4 * self.groups, 1, 1, 0)

    def forward(self, cp, sp):
        n, _, out_h, out_w = cp.size()
        sp = F.interpolate(sp, cp.size()[2:], mode='bilinear', align_corners=True)
        cp1x1 = self.cp(cp)
        sp1x1 = self.sp(sp)
        redim = self.redim(torch.cat([cp1x1, sp1x1], dim=1))
        # -------------------边缘差异偏量预测-------------------
        cp1x1_reshape = cp1x1.reshape(n * self.groups, -1, out_h, out_w)
        sp1x1_reshape = sp1x1.reshape(n * self.groups, -1, out_h, out_w)
        # 每组相识度结果，为高低级特征及横纵两个方向，预测偏移方向。
        similarity = F.cosine_similarity(cp1x1_reshape, sp1x1_reshape, dim=1).reshape(n, -1, out_h, out_w)
        fea_direction = self.direction(similarity)
        # 计算特征之间的差异性，预测偏移距离
        fea_dif = torch.abs(cp1x1_reshape-sp1x1_reshape).reshape(n, -1, out_h, out_w)
        fea_distance = self.distance(fea_dif)
        # 结合偏移的方向与距离预测校准偏量
        offset_join = torch.cat([redim, fea_direction, fea_distance], dim=1)
        conv_results = self.conv_offset(offset_join)

        cp = cp.reshape(n*self.groups, -1, out_h, out_w)
        sp = sp.reshape(n*self.groups, -1, out_h, out_w)

        offset_l = conv_results[:, 0:self.groups*2, :, :].reshape(n*self.groups,-1,out_h,out_w)
        offset_h = conv_results[:, self.groups*2:self.groups*4, :, :].reshape(n*self.groups,-1,out_h,out_w)

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(sp).to(sp.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n*self.groups, 1, 1, 1).type_as(sp).to(sp.device)

        grid_l = grid + offset_l.permute(0, 2, 3, 1) / norm
        grid_h = grid + offset_h.permute(0, 2, 3, 1) / norm

        cp = F.grid_sample(cp, grid_l.type_as(cp), align_corners=True)
        sp = F.grid_sample(sp, grid_h.type_as(sp), align_corners=True)

        cp = cp.reshape(n, -1, out_h, out_w)
        sp = sp.reshape(n, -1, out_h, out_w)

        fusion = cp + sp

        return fusion
