import numpy as np
import pickle as pkl
import warnings
warnings.filterwarnings('ignore') 
import torch 
from torch.optim import Adam

def log_sum_exp(inputs, keepdim=False, mask=None):
    """Numerically stable logsumexp on the last dim of `inputs`.
       reference: https://github.com/pytorch/pytorch/issues/2591
    Args:
        inputs: A Variable with any shape.
        keepdim: A boolean.
        mask: A mask variable of type float. It has the same shape as `inputs`.
              **ATTENTION** invalid entries are masked to **ONE**, not ZERO
    Returns:
        Equivalent of log(sum(exp(inputs), keepdim=keepdim)).
    """
    if mask is not None:
        max_offset = -1e7 * mask
    else:
        max_offset = 0.
    s, _ = torch.max(inputs + max_offset, dim=-1, keepdim=True)
    inputs_offset = inputs - s
    if mask is not None:
        inputs_offset.masked_fill_(mask.bool(), -float('inf'))
    outputs = s + inputs_offset.exp().sum(dim=-1, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(-1)
    return outputs

def convert_input(X, y_weighted):
    X_diff = [X[i][1][1:] - X[i][0][1:] for i in range(len(X))]
    x_all = torch.tensor(np.array(X_diff), dtype=torch.float64)
    y_all_weighted = torch.tensor(y_weighted, dtype=torch.float64)
    return x_all, y_all_weighted

def loss_function(params, x_all, y_all_weighted):
    # Negative Log-Likelihood
    # x_all: (N, M, d)
    # y_all: (N,)
    # msk_all: (N, M + 1)
    # utility_all: (N, M+1)
    utility_all = torch.sum(params * x_all, 1)  
    utility_all = utility_all.unsqueeze(1)
    utility_all = torch.cat((torch.zeros(utility_all.shape[0],1), utility_all), 1)
    LL = torch.sum(utility_all * y_all_weighted, 1) - log_sum_exp(utility_all) 
    return - torch.sum(LL) / x_all.shape[0] 

def fit(X, y_weighted, seed=0):
    params = torch.nn.Parameter(torch.ones(X[0].shape[1] - 1), requires_grad=True)
    torch.manual_seed(seed)
    x_all, y_all_weighted = convert_input(X, y_weighted)
    nll = loss_function(params, x_all, y_all_weighted)
    optimizer = Adam([params], lr=1e-2)
    for i in range(200):
        optimizer.zero_grad()
        nll = loss_function(params, x_all, y_all_weighted)
        nll.backward()
        optimizer.step()
        if (i + 1) % 1000 == 0 and display:
            print(nll)

    return params.detach().numpy()
