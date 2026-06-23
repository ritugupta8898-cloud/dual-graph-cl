import torch.nn.functional as F
import torch.nn as nn
class MainGraph(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        in_d = input_dim
        for h_d in hidden_dims:
            self.layers.append(nn.Linear(in_d, h_d))
            in_d = h_d
        self.head = nn.Linear(in_d, output_dim)    
            

    def forward(self, x, gain):
        h = x
        offset = 0
        for i, layer in enumerate(self.layers):
            h_dim = self.hidden_dims[i]
            g_slice = gain[:, offset:offset+h_dim]
            h = F.relu(layer(h))
            h = h * g_slice
            offset += h_dim
        return self.head(h), h