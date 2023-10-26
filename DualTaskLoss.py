

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from Loss.util.custom_functional import compute_grad_mag


def perturbate_input_(input, n_elements=200):
    N, C, H, W = input.shape
    assert N == 1
    c_ = np.random.random_integers(0, C - 1, n_elements)
    h_ = np.random.random_integers(0, H - 1, n_elements)
    w_ = np.random.random_integers(0, W - 1, n_elements)
    for c_idx in c_:
        for h_idx in h_:
            for w_idx in w_:
                input[0, c_idx, h_idx, w_idx] = 1
    return input


def _sample_gumbel(shape, eps=1e-10):

    U = torch.rand(shape).cuda()
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):

    assert logits.dim() == 3
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps)
    y = logits + gumbel_noise
    return F.softmax(y / tau, 1)


def _one_hot_embedding(labels, num_classes):


    y = torch.eye(num_classes).cuda()
    return y[labels].permute(0, 3, 1, 2)


class DualTaskLoss(nn.Module):
    def __init__(self, cuda=False):
        super(DualTaskLoss, self).__init__()
        self._cuda = cuda
        return

    def forward(self, input_logits, gts, ignore_pixel=255):
        """
        :param input_logits: NxCxHxW
        :param gt_semantic_masks: NxCxHxW
        :return: final loss
        """
        N, C, H, W = input_logits.shape
        th = 1e-8  # 1e-10
        eps = 1e-10
        ignore_mask = (gts == ignore_pixel).detach()
        input_logits = torch.where(ignore_mask.view(N, 1, H, W).expand(N, 9, H, W),
                                   torch.zeros(N, C, H, W).cuda(),
                                   input_logits)
        gt_semantic_masks = gts.detach()
        gt_semantic_masks = torch.where(ignore_mask, torch.zeros(N, H, W).long().cuda(), gt_semantic_masks)
        gt_semantic_masks = _one_hot_embedding(gt_semantic_masks, 9).detach()

        g = _gumbel_softmax_sample(input_logits.view(N, C, -1), tau=0.5)
        g = g.reshape((N, C, H, W))
        g = compute_grad_mag(g, cuda=self._cuda)

        g_hat = compute_grad_mag(gt_semantic_masks, cuda=self._cuda)

        g = g.contiguous().view(N, -1)
        g_hat = g_hat.contiguous().view(N, -1)
        loss_ewise = F.l1_loss(g, g_hat, reduction='none', reduce=False)

        p_plus_g_mask = (g >= th).detach().float()
        loss_p_plus_g = torch.sum(loss_ewise * p_plus_g_mask) / (torch.sum(p_plus_g_mask) + eps)

        p_plus_g_hat_mask = (g_hat >= th).detach().float()
        loss_p_plus_g_hat = torch.sum(loss_ewise * p_plus_g_hat_mask) / (torch.sum(p_plus_g_hat_mask) + eps)

        total_loss = 0.5 * loss_p_plus_g + 0.5 * loss_p_plus_g_hat

        return total_loss
