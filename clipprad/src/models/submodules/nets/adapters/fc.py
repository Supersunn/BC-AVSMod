import torch.nn as nn

class AdapterLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=8, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        self.dim_out = dim // mlp_ratio
        
        self.fc_down = nn.Sequential(
            nn.Linear(dim, self.dim_out),
            nn.GELU(),
        )
        self.fc_up = nn.Sequential(
            nn.Linear(self.dim_out, dim)
        )
        
    def forward(self, x):
        # Shape of x: (B * T, HW + 1, D)
        xs = self.fc_down(x)
        xs = self.fc_up(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    