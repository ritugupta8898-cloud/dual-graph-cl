import torch.nn.functional as F
import torch.nn as nn

class MainGraph(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.all_dims = [input_dim] + hidden_dims + [output_dim]  
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        in_d = input_dim
        for h_d in hidden_dims:
            self.layers.append(nn.Linear(in_d, h_d))
            in_d = h_d
        self.head = nn.Linear(in_d, output_dim)

    def forward(self, x, gain):
        offset = 0
        h = x  # no gating here anymore

        for i, layer in enumerate(self.layers):
            h_dim = self.hidden_dims[i]
            g_slice = gain[:, offset:offset+h_dim]
            h = F.relu(layer(h))
            h = h * g_slice
            offset += h_dim

        out = self.head(h)
        output_dim = self.all_dims[-1]
        g_out = gain[:, offset:offset+output_dim]
        out = out * g_out
        offset += output_dim

        return out, h