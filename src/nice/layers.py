
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.add_module('NonLinearLayer',self.layer)
        
    def forward(self, x):
        y1 = self.first_block(x)
        y2  = self.coupling_law(self.second_block(x), self.layer(self.first_block(x)))
        
        y = torch.zeros((y1.shape[0], int(y1.shape[1] * 2)))
        if self.partition == 'even':
            j=0
            for k in range(y2.shape[1]):
                y[:,j] = y1[:,k]
                y[:,j+1] = y2[:,k]
                j+=2
        else:
            j=0
            for k in range(y1.shape[1]):
                y[:,j] = y2[:,k]
                y[:,j+1] = y1[:,k]
                j+=2
        return y

    def inverse(self, y):
        x1 = self.first_block(y)
        x2 = self.anticoupling_law(self.second_block(y), self.layer(self.first_block(y)))

        x = torch.zeros((x1.shape[0], int(x1.shape[1] * 2)))
        if self.partition == 'even':
            j=0
            for k in range(x2.shape[1]):
                x[:,j] = x1[:,k]
                x[:,j+1] = x2[:,k]
                j+=2
        else:
            j=0
            for k in range(x1.shape[1]):
                x[:,j] = x2[:,k]
                x[:,j+1] = x1[:,k]
                j+=2
        return x
        
    def coupling_law(self, a, b):
        return (a + b)
    def anticoupling_law(self, a, b):
        return (a - b)
