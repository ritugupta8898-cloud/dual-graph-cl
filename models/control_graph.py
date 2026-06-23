import torch
import torch.nn as nn

from models.diffusion import diffusion


class ControlGraph(nn.Module):
    def __init__(
        self,
        input_dim,
        context_dim,
        num_control_nodes,
        W,
        lambda_diff,
        K,
       
    ):
        super().__init__()

        self.action_generator = nn.Linear(
            input_dim + context_dim,
            num_control_nodes,
        )
        

        self.register_buffer('W', W)
        self.lambda_diff = lambda_diff
        self.K = K

    def forward(self, x, z=None,supression=None):

        if z is not None:
            inp = torch.cat([x, z], dim=-1)
        else:
            inp = x

        
        A0 = self.action_generator(inp)

       
        AK = diffusion(
            A0,
            self.W,
            self.lambda_diff,
            self.K,
        )
        if supression is not None:
            AK = AK*supression
         
        gain = 1.0 + torch.tanh(AK)

        return gain, AK, A0