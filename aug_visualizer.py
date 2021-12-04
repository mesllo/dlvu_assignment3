import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.autoaugment import AutoAugmentPolicy

policy = transforms.AutoAugmentPolicy.CIFAR10

aug_transform = transforms.Compose(
    [#transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
     transforms.RandomRotation(degrees=(-30, 30)), # best augmentation performed equally as well as no augmentation,
     #transforms.RandomAffine(degrees=(0, 30), translate=(0.1, 0.3), scale=(1, 1.5)),
     #transforms.AutoAugment(policy),
     transforms.ToTensor()])

# dataset load
train = torchvision.datasets.MNIST(root='./data', train=True, 
                                   download=True, transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(train, batch_size=10,
                                          shuffle=False, num_workers=0)

aug_train = torchvision.datasets.MNIST(root='./data', train=True, 
                                   download=True, transform=aug_transform)
    
aug_trainloader = torch.utils.data.DataLoader(aug_train, batch_size=10,
                                          shuffle=False, num_workers=0)

imgs = enumerate(trainloader)
aug_imgs = enumerate(aug_trainloader)

batch_idx, (img, label) = next(imgs)
aug_batch_idx, (aug_img, aug_label) = next(aug_imgs)

# visualize augmentation example
plt.imshow(img[4][0], cmap="gray")
plt.show()
plt.imshow(aug_img[4][0], cmap="gray")
plt.show()