import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# params
batch_size = 16
epochs = 1

# dataset load
train = torchvision.datasets.MNIST(root='./data', train=True, 
                                   download=True, transform=transforms.ToTensor())

train_set, val_set = torch.utils.data.random_split(train, [50000, 10000])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

test = torchvision.datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                         shuffle=True, num_workers=0)

# train loop
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):        
        # use GPU
        inputs, labels = data[0].to(device), data[1].to(device)
        
        print(i)
        print(type(data))