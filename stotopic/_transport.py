import torch
from torch import nn
import torch.nn.functional as F
from ._model_utils import pairwise_euclidean_distance

class ETP(nn.Module):

    def __init__(self, sinkhorn_alpha, init_a_dist=None, init_b_dist=None, OT_max_iter=5000, stop_thr=0.005):
        super().__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.stop_thr = stop_thr
        self.init_a_dist = init_a_dist
        self.init_b_dist = init_b_dist
        if init_a_dist is not None:
            self.a_dist = init_a_dist
        if init_b_dist is not None:
            self.b_dist = init_b_dist

    def forward(self, x, y):
        M = pairwise_euclidean_distance(x, y)
        device = M.device
        if self.init_a_dist is None:
            a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
        else:
            a = F.softmax(self.a_dist, dim=0).to(device)
        if self.init_b_dist is None:
            b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)
        else:
            b = F.softmax(self.b_dist, dim=0).to(device)
        log_a = torch.log(a + 1e-30)
        log_b = torch.log(b + 1e-30)
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        log_K = -M * self.sinkhorn_alpha
        err = 1
        iteration = 0
        while err > self.stop_thr and iteration < self.OT_max_iter:
            log_Ku = log_K.T + log_u.T
            log_v = log_b - torch.logsumexp(log_Ku, dim=1).unsqueeze(1)
            log_Kv = log_K + log_v.T
            log_u = log_a - torch.logsumexp(log_Kv, dim=1).unsqueeze(1)
            iteration += 1
            if iteration % 50 == 1:
                log_K = log_K + log_u + log_v.T
                log_u = torch.zeros_like(log_a)
                log_v = torch.zeros_like(log_b)
                err = self.check_convergence(log_K, log_u, log_v, a, b)
        u = torch.exp(log_u)
        v = torch.exp(log_v)
        K = torch.exp(log_K)
        transport_plan = u * (K * v.T)
        loss_ETP = torch.sum(transport_plan * M)
        return (loss_ETP, transport_plan)

    def check_convergence(self, log_K, log_u, log_v, a, b):
        with torch.no_grad():
            log_Kv = log_K + log_v.T
            log_Kv_sum = torch.logsumexp(log_Kv, dim=1, keepdim=True)
            log_row_sums = log_u + log_Kv_sum
            row_sums = torch.exp(log_row_sums)
            log_Ku = log_K.T + log_u.T
            log_Ku_sum = torch.logsumexp(log_Ku, dim=1, keepdim=True)
            log_col_sums = log_v + log_Ku_sum
            col_sums = torch.exp(log_col_sums)
            row_err = torch.abs(row_sums - a).sum()
            col_err = torch.abs(col_sums - b).sum()
            return max(row_err.item(), col_err.item())
