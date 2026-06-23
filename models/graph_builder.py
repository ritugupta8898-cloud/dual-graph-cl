#right now the adj matrix only has 0/1 so i have not normalied i will do. this when i actially use edge weights 
import torch
def build_graph(hidden_dims, graph_type="layer"):
    #N total nodes 
    N = sum(hidden_dims)

    total_layer = len(hidden_dims)
    offsets = [0]
    
    for d in hidden_dims:
        offsets.append(offsets[-1] + d)

    W = torch.zeros(N, N)
    if graph_type == "layer":
     for i in range(total_layer - 1):
        start_j, end_j = offsets[i+1], offsets[i+2]
        start_i, end_i = offsets[i], offsets[i+1]
        W[start_i:end_i,start_j:end_j] = 1
    W = W + W.T
    W.fill_diagonal_(1)
    return W
       

