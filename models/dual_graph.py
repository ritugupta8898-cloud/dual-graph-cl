from models.control_graph import ControlGraph
from models.main_graph import MainGraph
import torch.nn as nn

class DualGraphNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, context_dim, W, lambda_diff=0.5, K=5):
        super().__init__()
        self.num_control_nodes = sum(hidden_dims)
        self.g_mod = ControlGraph(input_dim, context_dim, self.num_control_nodes, W, lambda_diff, K)
        self.g_main = MainGraph(input_dim, hidden_dims, output_dim)

    def forward(self, x, z=None,supression =None):
        gain, AK, A0 = self.g_mod(x, z,supression)
        out, final_h = self.g_main(x, gain)
        return out, AK, final_h