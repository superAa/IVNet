# 这里的loss构成 = 正则化loss(预测分割图的边界<-->真实边界、预测分割图)
from copy import deepcopy
import torch
import torch.nn as nn
from Loss.DualTaskLoss import DualTaskLoss
from Loss.Focal_loss import FocalLoss
import torch.nn.functional as F


import pdb
class Loss(nn.Module):
    def __init__(self,edge_weight=1, seg_weight=1, att_weight=1, dual_weight=1):
        super(Loss, self).__init__()
        self.seg_weight = seg_weight
        self.edge_weight = edge_weight
        self.att_weight = att_weight
        self.dual_weight = dual_weight
        self.bce2d = nn.BCELoss()
        self.dual_task = DualTaskLoss()
        self.seg_loss = FocalLoss(class_num=9)
        pass

    def edge_attention(self, input, target, edge):
        n, c, h, w = input.size()
        filler = torch.ones_like(target) * 255
        return F.cross_entropy(input, torch.where(edge.max(1)[0] > 0.7, target, filler),ignore_index=255)

    def forward(self,inputs,targets):

        segin = inputs
        segmask = targets

        losses = {}

        losses['seg_loss'] = self.seg_weight * self.seg_loss(segin, segmask)

        # losses['edge_loss'] = self.edge_weight * 25 * self.bce2d(edgein, edgemask)# 二分类。

        losses['att_loss'] = self.att_weight * self.edge_attention(segin, segmask, edgein)

        # losses['dual_loss'] = self.dual_weight * self.dual_task(segin, segmask)

        return losses

        pass

