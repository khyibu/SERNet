# !/usr/bin/python
# -*- coding: utf-8 -*-
import torch.nn as nn
from BRRNet_Utils import conv_block, encodeBlock, up_conv, center_block, rrm_ours,PredOnly

class BRRNet(nn.Module):
    def __init__(self, img_channel=3, classnumber=1):
        super(BRRNet, self).__init__()
        self.encodeBlock1 = encodeBlock(ch_in=img_channel, ch_out=64)
        self.encodeBlock2 = encodeBlock(ch_in=64, ch_out=128)
        self.encodeBlock3 = encodeBlock(ch_in=128, ch_out=256)
        self.centBolock = center_block(ch_in=256, ch_out=512, rate=[1, 2, 4, 8, 16, 32])

        self.upConv1 = up_conv(ch_in=512, ch_out=256)
        self.up_conv_block1 = conv_block(ch_in=512, ch_out=256)
        self.upConv2 = up_conv(ch_in=256, ch_out=128)
        self.up_conv_block2 = conv_block(ch_in=256, ch_out=128)
        self.upConv3 = up_conv(ch_in=128, ch_out=64)
        self.up_conv_block3 = conv_block(ch_in=128, ch_out=64)
        self.rrm_block = rrm_ours(ch_in=1, ch_out=classnumber, rate=[1, 2, 4, 8, 16, 32])
        self.pred = PredOnly(ch_in=64, ch_out=classnumber)

    def forward(self, x):
        rEncode1, pooling1 = self.encodeBlock1(x)
        rEncode2, pooling2 = self.encodeBlock2(pooling1)
        rEncode3, pooling3 = self.encodeBlock3(pooling2)
        rCent = self.centBolock(pooling3)

        rUp_Conv1 = self.upConv1([rCent, rEncode3])
        rDecode_Conv1 = self.up_conv_block1(rUp_Conv1)
        rUp_Conv2 = self.upConv2([rDecode_Conv1, rEncode2])
        rDecode_Conv2 = self.up_conv_block2(rUp_Conv2)
        rUp_Conv3 = self.upConv3([rDecode_Conv2, rEncode1])
        rDecode_Conv3 = self.up_conv_block3(rUp_Conv3)
        rPred = self.pred(rDecode_Conv3)
        result = self.rrm_block(rPred)
        return result


if __name__ == '__main__':
    from torchstat import stat
    num_class = 1
    model = BRRNet()
    stat(model, (3, 512, 512))
