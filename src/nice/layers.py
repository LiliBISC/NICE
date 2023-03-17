
import torch
import torch.nn as nn
import numpy as np

even = lambda xs: xs[:,0::2]
odd = lambda xs: xs[:,1::2]

class AdditiveCouplingLayer(nn.Module):
    """Layer with coupling law g(a;b) := a + b."""
    def __init__(self, partition, latent_dim, hidden_dim):
        super(AdditiveCouplingLayer, self).__init__()

        # partition choice : article page 6. 5/5.1
        #split x in two blocks x1 and x2
        self.partition = partition
        if (partition == 'even'):
            self.first_block = even
            self.second_block = odd
        else:
            self.first_block = odd
            self.second_block = even

        #non linear function : article page 2 "Relu MLP in our example"
        #five hidden layer 
        self.layer = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x):
        x1 = self.first_block(x)
        y1  = self.coupling_law(self.second_block(x), self.layer(self.first_block(x)))
        cols = []
        if self.partition == 'even':
            for k in range(y1.shape[1]):
                cols.append(x1[:,k])
                cols.append(y1[:,k])
            if x1.shape[1] > y1.shape[1]:
                cols.append(x1[:,-1])
        else:
            for k in range(x1.shape[1]):
                cols.append(y1[:,k])
                cols.append(x1[:,k])
            if y1.shape[1] > x1.shape[1]:
                cols.append(y1[:,-1])
        return torch.stack(cols, dim=1)

    def inverse(self, y):
        y1 = self.first_block(y)
        x2 = self.anticoupling_law(self.second_block(y), self.layer(self.first_block(y)))

        cols = []
        if self.partition == 'even':
            for k in range(x2.shape[1]):
                cols.append(y1[:,k])
                cols.append(x2[:,k])
            if y1.shape[1] > x2.shape[1]:
                cols.append(y1[:,-1])
        else:
            for k in range(y1.shape[1]):
                cols.append(x2[:,k])
                cols.append(y1[:,k])
            if x2.shape[1] > y1.shape[1]:
                cols.append(x2[:,-1])
        return torch.stack(cols, dim=1)
        
    def coupling_law(self, a, b):
        return (a + b)
    def anticoupling_law(self, a, b):
        return (a - b)
