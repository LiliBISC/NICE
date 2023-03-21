import torch
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
from nice.model import NICEModel
from torch.utils.data import DataLoader
from sklearn.datasets import make_moons
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def rescale(x, lo, hi):
    """Rescale a tensor to [lo,hi]."""
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

def load_mnist(train=True, batch_size=200):
    """Rescale and preprocess MNIST dataset."""
    mnist_transform = torchvision.transforms.Compose([
        # convert PIL image to tensor:
        torchvision.transforms.ToTensor(),
        # flatten:
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
        torchvision.transforms.Lambda(lambda x: (x + torch.rand_like(x).div_(256.))),
        torchvision.transforms.Lambda(lambda x: rescale(x, 0., 1.))

    ])

    dataloader = DataLoader(
                            dataset = torchvision.datasets.MNIST(root="./datasets/mnist", train=train, transform=mnist_transform, download=True),
                            batch_size=batch_size,
                            drop_last=True
                            )
    return dataloader

def save_model(model, name):
    path = os.getcwd()+"/src/saved_models/"+ str(name) + ".pt"
    torch.save(model.state_dict(), path)

dataloader = load_mnist()

def train(num_epochs, dataloader, model):
    """Construct a NICE model and train over a number of epochs."""
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-4)
    # === train over a number of epochs
    total_loss = []
    for t in range(num_epochs):
        print("* Epoch {0}:".format(t))
        stock_loss = 0.0
        model.train()
        for inputs, _ in tqdm(dataloader):

            optimizer.zero_grad()
            output, _ = model(inputs)
            loss = - model.log_prob(inputs).mean()
            loss.backward()
            optimizer.step()
            
            stock_loss-=loss.item()
        total_loss.append(stock_loss/len(dataloader))
        print(stock_loss/len(dataloader))

        '''
        model.eval()        # set to inference mode
        with torch.no_grad():
            z, _ = model.forward(inputs)
            reconst = model.inverse(z)
            reconst = reconst.reshape((200, 1, 28, 28))
            samples = model.sample(64)
            samples = samples.reshape((200, 1, 28, 28))
            torchvision.utils.save_image(torchvision.utils.make_grid(reconst),
                './reconstruction/mnist' + str(t * 300) +'iter.png')
            torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                './samples/mnist' + str(t * 300) +'iter.png')
        '''

    save_model(model, 'nice_test')
    return total_loss

model = NICEModel(28*28, 28*28)
#model = NICE(4, 784, 1000, 5, 1)
loss_ls = train(100, dataloader, model)

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('NLLLoss')
plt.plot(loss_ls)
plt.grid(True)
plt.show()
