# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class CELoss(torch.nn.Module):
    def __init__(self, base_criterion: torch.nn.Module):
        super().__init__()
        self.base_criterion = base_criterion

    def forward(self, outputs, labels):
        ce_loss = self.base_criterion(outputs, labels)
        return ce_loss


class DistillKL(torch.nn.Module):
    def __init__(self):
        super(DistillKL, self).__init__()
        self.T = 4.0

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t.detach(), reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss


class HSMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_s, y_t):
        loss = 0.
        for i in range(0, 24):
            # y_s[i] = F.normalize(y_s[i], dim=2)
            # y_t[i] = F.normalize(y_t[i], dim=2)
            loss = F.mse_loss(y_s[i], y_t[i])

        return loss
