

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


import torch
import math
EPSILON = 1e-5
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class SANet(nn.Module):
    def __init__(self, in_channel):
        super(SANet, self).__init__()
        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.cbr2 = nn.Sequential(
            nn.Conv2d( 3, 1, kernel_size=7, stride=1, padding=3 ),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    def forward(self, x):

        x1 = self.cbr1(x)
        x2 = torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1), x1], dim=1)
        scale = self.cbr2(x2)

        return scale*x+x

class CANet(nn.Module):
    def __init__(self,  inchannel):
        super(CANet, self).__init__()

        reduction = inchannel // 2
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(inchannel, inchannel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel // reduction, inchannel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        scale = self.sigmoid(max_out + avg_out)


        return scale*x+x


class IVF(nn.Module):

    def __init__(self,in_channel):
        super(IVF, self).__init__()
        self.channel_attention_ir = CANet(in_channel)
        self.channel_attention_vis = CANet(in_channel)

        self.spatial_attention_ir = SANet(in_channel)
        self.spatial_attention_vis = SANet(in_channel)

        self.w = torch.nn.Parameter(torch.ones(2), requires_grad=True)


    def forward(self,tensor_ir, tensor_vis):

        ir_weight  = self.channel_attention_ir(tensor_ir)
        vis_weight = self.channel_attention_vis(tensor_vis)
        f_channel = (1.0+ir_weight)*tensor_ir+(1.0+vis_weight)*tensor_vis

        ir_weight  = self.spatial_attention_ir(tensor_ir)
        vis_weight = self.spatial_attention_vis(tensor_vis)
        f_spatial = (1.0+ir_weight)*tensor_ir+(1.0+vis_weight)*tensor_vis

        # 归一化权重
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        out = w1 * f_channel + w2 * f_spatial

        return out














