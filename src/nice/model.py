"""
Implementation of models from paper.
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from .layers import AdditiveCouplingLayer
    
class NICEModel(nn.Module):
    """
    Replication of model from the paper:
      "Nonlinear Independent Components Estimation",
      Laurent Dinh, David Krueger, Yoshua Bengio (2014)
      https://arxiv.org/abs/1410.8516
    Contains the following components:
    * four additive coupling layers with nonlinearity functions consisting of
      five-layer RELUs (2 in our case)
    * a diagonal scaling matrix output layer
    """
    def __init__(self, input_dim, hidden_dim):
        super(NICEModel, self).__init__()

        self.input_dim = input_dim
        half_dim = int(input_dim / 2)

        self.layer1 = AdditiveCouplingLayer('odd', half_dim, hidden_dim)
        self.layer2 = AdditiveCouplingLayer('even', half_dim, hidden_dim)
        self.layer3 = AdditiveCouplingLayer('odd', half_dim, hidden_dim)
        self.layer4 = AdditiveCouplingLayer('even', half_dim, hidden_dim)
        
        self.scaling_diag = nn.Parameter(torch.ones(input_dim))

        # randomly initialize weights:
        for p in self.layer1.parameters():
            if len(p.shape) > 1:
                init.kaiming_uniform_(p, nonlinearity='relu')
            else:
                init.normal_(p, mean=0., std=0.001)
        for p in self.layer2.parameters():
            if len(p.shape) > 1:
                init.kaiming_uniform_(p, nonlinearity='relu')
            else:
                init.normal_(p, mean=0., std=0.001)
        for p in self.layer3.parameters():
            if len(p.shape) > 1:
                init.kaiming_uniform_(p, nonlinearity='relu')
            else:
                init.normal_(p, mean=0., std=0.001)
        for p in self.layer4.parameters():
            if len(p.shape) > 1:
                init.kaiming_uniform_(p, nonlinearity='relu')
            else:
                init.normal_(p, mean=0., std=0.001)   

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
        ys = torch.matmul(ys, torch.diag(torch.exp(self.scaling_diag)))
        return ys


    def inverse(self, ys):
        """Invert a set of draws from gaussians"""
        with torch.no_grad():
            xs = torch.matmul(ys, torch.diag(torch.reciprocal(torch.exp(self.scaling_diag))))
            xs = self.layer4.inverse(xs)
            xs = self.layer3.inverse(xs)
            xs = self.layer2.inverse(xs)
            xs = self.layer1.inverse(xs)
        return xs