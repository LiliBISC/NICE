import torch
import torchvision
import torch.optim as optim
from nice.model import NICEModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def rescale(x, lo, hi):
    """Rescale a tensor to [lo,hi]."""
    assert(lo < hi), "[rescale] lo={0} must be smaller than hi={1}".format(lo,hi)
    old_width = torch.max(x)-torch.min(x)
    old_center = torch.min(x) + (old_width / 2.)
    new_width = float(hi-lo)
    new_center = lo + (new_width / 2.)
    # shift everything back to zero:
    x = x - old_center
    # rescale to correct width:
    x = x * (new_width / old_width)
    # shift everything to the new center:
    x = x + new_center
    # return:
    return x

def load_mnist(train=True, batch_size=400):
    """Rescale and preprocess MNIST dataset."""
    mnist_transform = torchvision.transforms.Compose([
        # convert PIL image to tensor:
        torchvision.transforms.ToTensor(),
        # flatten:
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
        # add uniform noise:
        torchvision.transforms.Lambda(lambda x: (x + torch.rand_like(x).div_(256.))),
        # rescale to [0,1]:
        torchvision.transforms.Lambda(lambda x: rescale(x, 0., 1.))
    ])

    dataloader = DataLoader(
                            dataset = torchvision.datasets.MNIST(root="./datasets/mnist", train=train, transform=mnist_transform, download=True),
                            batch_size=batch_size,
                            drop_last=True
                            )
    return dataloader

def Nice_criterion(diag, outputs):
    """
    Definition of log-likelihood function with a Logistic prior.
    """
    l = (torch.sum(diag) - (torch.sum(torch.log1p(torch.exp(outputs)) + torch.log1p(torch.exp(-outputs)), dim=1)))
    return torch.mean(l) #size_average = True comme dans la NLLLoss de pytorch

def save_model(model, name):
    path = os.getcwd()+name+'.pt'
    torch.save(model.state_dict(), path)

dataloader = load_mnist()

def train(num_epochs, dataloader, model):
    """Construct a NICE model and train over a number of epochs."""
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.01), eps=1e-4)
    # === train over a number of epochs
    total_loss = []
    for t in range(num_epochs):
        print("* Epoch {0}:".format(t))
        stock_loss = 0.0
        for inputs, _ in tqdm(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = Nice_criterion(model.scaling_diag, outputs)
            loss.backward()
            optimizer.step()

            stock_loss+=stock_loss+loss.item()
        
        total_loss.append(stock_loss)
        # perform validation loop:
        print(loss)
    
    save_model(model, 'nice_test')

model = NICEModel(28*28, 1000)
train(10, dataloader, model)