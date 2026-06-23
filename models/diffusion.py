import torch

def diffusion(A0, W, lambda_diff, K):
    A = A0.clone()
    # A(t+1)=λWA(t)+(1−λ)A0​

    for i in range(K):
        influence = A@W
        A = lambda_diff*influence+(1-lambda_diff)*A0
    return A    
