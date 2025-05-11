import torch.nn as nn
from torch.nn import functional as F
import math
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

class Similarity_coding(nn.Module):
    def __init__(self, group, kernel):
        super(Similarity_coding, self).__init__()
        self.group = group
        self.kernel = kernel
        self.pool_conv_q = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.kernel, self.kernel, 1, bias=False))
        self.pool_conv_k = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.kernel, self.kernel, 1, bias=False))
        self.conv = ConvBR(self.kernel, self.kernel, kernel=1)

    def forward(self, x):
        n, mask_c, h, w = x.size()  # n k*k*g h w
        x = x.reshape(n * self.group, -1, h, w)  # n*g k*k h w
        mask_q = self.pool_conv_q(x).view(n * self.group, -1, 1)  # 1*G k*k 1
        mask_k = self.pool_conv_k(x).view(n * self.group, 1, -1)  # 1*G 1 k*k
        similar = mask_q @ mask_k  # 1*G k*k k*k
        similar = F.softmax(similar, dim=-1)  # 1*G k*k
        mask_v = self.conv(x).view(n * self.group, -1, h*w)  # 1*G k*k h*w
        similar_mask = (similar @ mask_v).view(n * self.group, -1, h, w)  # 1 G k*k h w
        similar_mask = similar_mask.view(n, self.group, -1, h, w)

        norm_mask = F.softmax(similar_mask, dim=2, dtype=x.dtype).contiguous()  # 1 G k*k h w

        return norm_mask

class PKSRM(nn.Module):
    def __init__(self, in_ch, group):
        super(PKSRM, self).__init__()
        self.group = group
        self.special_ker = 3
        self.special_compres = nn.Sequential(ConvBR(in_ch, in_ch//2, kernel=1),
            ConvBR(in_ch//2, self.group*self.special_ker * self.special_ker, mode=4))
        self.Similarity_coding = Similarity_coding(self.group, self.special_ker * self.special_ker)

    def high_edge(self, x, norm_mask, k):
        b, c, h, w = x.shape  # n c h w
        _, _, m_c, m_h, m_w = norm_mask.shape  # 1 G k*k h w

        unfold_x = F.unfold(x, kernel_size=(k, k), stride=1, padding=k//2)  # n k*k*c h*w
        unfold_x = unfold_x.reshape(b, c * k * k, h, w)  # n k*k*c h w
        unfold_x = unfold_x.reshape(b, c, k * k, h, w)  # n c k*k h w
        unfold_x = unfold_x.reshape(b, self.group, c//self.group, k * k, m_h, m_w)  # n g c//g k*k h w

        normed_mask = norm_mask.reshape(b, self.group, 1, k * k, m_h, m_w)  # n g 1 k*k h w
        res = unfold_x * normed_mask  # n g c//g k*k h w
        res = res.reshape(b, c, k * k, m_h, m_w)  # n c k*k h w
        res = res.sum(dim=2).reshape(b, c, m_h, m_w)  # n c h w

        return res

    def forward(self, special):
        special_compres = self.special_compres(special)  # n k*k*g h w
        norm_special = self.Similarity_coding(special_compres)  # n k*k*g h w
        special_out = self.high_edge(special, norm_special, k=self.special_ker)  # n c h w

        return special_out
