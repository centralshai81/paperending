import numpy as np
import torch
import torch.nn as nn

class PINNMultiClass(nn.Module):
    
    def __init__(self, in_dim=156, n_bus=39, n_classes=5,
                 hidden=128, depth=5, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh(), nn.Dropout(dropout)]
            d = hidden
        self.backbone   = nn.Sequential(*layers)
        self.cls_head   = nn.Linear(hidden, n_classes)
        self.state_head = nn.Linear(hidden, 2 * n_bus)
        self.n_bus      = n_bus
        self.n_classes  = n_classes

    def forward(self, x):
        h      = self.backbone(x)
        logits = self.cls_head(h)
        state  = self.state_head(h)
        #  Vm to [0.8, 1.2] pu 
        Vm_hat = 0.2 * torch.tanh(state[:, :self.n_bus]) + 1.0
        #  Va to [-10°, +10°] 
        Va_hat = (10.0 * np.pi / 180.0) * torch.tanh(state[:, self.n_bus:])
        return logits, Vm_hat, Va_hat


class MLPBaseline(nn.Module):
    """Plain MLP baseline for ablation study comparison."""
    def __init__(self, in_dim=156, n_classes=5, hidden=128, depth=3, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.cls_head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.cls_head(self.backbone(x))
