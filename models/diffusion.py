import torch

def diffusion(A0, W, lambda_diff, K):
    deg = W.sum(dim=1).clamp(min=1e-12)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    W_norm = D_inv_sqrt @ W @ D_inv_sqrt
    A = A0.clone()
    # A(t+1)=λWA(t)+(1−λ)A0​

    for i in range(K):
        influence = A@W_norm
        A = lambda_diff*influence+(1-lambda_diff)*A0
    return A    
