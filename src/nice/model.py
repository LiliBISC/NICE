"""
Implementation of models from paper.
"""
import torch
import torch.nn as nn
from .layers import AdditiveCouplingLayer
    
class NICEModel(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(NICEModel, self).__init__()

        self.input_dim = input_dim
        half_dim = int(input_dim // 2)

        self.layer1 = AdditiveCouplingLayer('even', half_dim, hidden_dim)
        self.layer2 = AdditiveCouplingLayer('odd', half_dim, hidden_dim)
        self.layer3 = AdditiveCouplingLayer('even', half_dim, hidden_dim)
        self.layer4 = AdditiveCouplingLayer('odd', half_dim, hidden_dim)
        
        self.scaling_diag = Scaling(input_dim)

    def log_prob(self, x):
        z, log_det_J = self.forward(x)
        def prior():
            return -(torch.nn.functional.softplus(z) + torch.nn.functional.softplus(-z))
        
        log_ll = torch.sum(prior(), dim=1)
        return log_ll + log_det_J

    def forward(self, xs):
        """
        Forward pass through all invertible coupling layers.
        
        Args:
        * xs: float tensor of shape (B,dim).
        Returns:
        * ys: float tensor of shape (B,dim).
        """
        ys = self.layer1(xs)
        ys = self.layer2(ys)
        ys = self.layer3(ys)
        ys = self.layer4(ys)
        output = self.scaling_diag(ys)
        return output

    def inverse(self, ys):
        """Invert a set of draws from gaussians"""

        xs, _ = self.scaling(ys, reverse=True)
        xs = self.layer4.inverse(xs)
        xs = self.layer3.inverse(xs)
        xs = self.layer2.inverse(xs)
        xs = self.layer1.inverse(xs)

        return xs
    
class Scaling(nn.Module):
    def __init__(self, dim):
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(torch.zeros((1, dim)), requires_grad=True)

    def forward(self, x, reverse=False):
        log_det_J = torch.sum(self.scale)
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)
        return x, log_det_J
