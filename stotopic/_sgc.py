import numpy as np
import torch
from scipy.sparse import identity, diags

def make_sparse_tensor(adj):
    adj = adj.astype('float32').tocoo()
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adj.shape
    adj = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return adj

def precompute_SGC_scipy(x, adj, n_layers, mode='sign', add_diag=True, csr=True):
    deg = adj.sum(axis=1)
    if add_diag:
        deg = deg + 2
        adj = adj + identity(n=x.shape[0])
    deg = diags(deg.A1).power(-0.5)
    deg.data[deg.data == np.inf] = 0
    adj = deg @ adj @ deg
    adj = make_sparse_tensor(adj)
    ls = x.sum(dim=1, keepdims=True)
    ms = torch.median(x.sum(axis=1))
    xs = x / ls.clamp_min(1e-12) * ms
    sgc = [xs]
    for _ in range(n_layers):
        xs = torch.sparse.mm(adj, sgc[-1])
        sgc.append(xs)
    if mode == 'sgc':
        sgc = [sgc[-1]]
    sgc_x = torch.cat(sgc, dim=1)
    sgc_x = torch.log(sgc_x + 1)
    return sgc_x
