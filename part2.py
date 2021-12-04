import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.autoaugment import AutoAugmentPolicy

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# params
batch_size = 16
epochs = 1
learning_rate = 0.001
augment = True
# no auto-augment policy autoperformed default or random rotation
policy = transforms.AutoAugmentPolicy.CIFAR10

# dataset load
train = torchvision.datasets.MNIST(root='./data', train=True, 
                                   download=True, transform=transforms.ToTensor())

if augment:
    aug_transform = transforms.Compose(
    [#transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
     transforms.RandomRotation(degrees=(-30, 30)), # best augmentation performed equally as well as no augmentation,
     #transforms.RandomAffine(degrees=(0, 30), translate=(0.1, 0.3), scale=(1, 1.5)),
     #transforms.AutoAugment(policy),
     transforms.ToTensor()])
    train = torchvision.datasets.MNIST(root='./data', train=True, 
                                   download=True, transform=aug_transform)

train_set, val_set = torch.utils.data.random_split(train, [50000, 10000])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

test = torchvision.datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                         shuffle=True, num_workers=0)    

# define CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x
    
net = Net()
net.to(device)

# define loss function
#criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

# define optimizer
optimizer = optim.AdamW(net.parameters(), lr=0.001)

# define loss lists to collect errors for plotting
running_losses = []
epoch_running_losses = []

# train loop
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    
    for i, data in enumerate(trainloader):        
        # use GPU
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # set num_classes in case batch size doesn't contain all labels
        one_hot_labels = F.one_hot(labels, num_classes=10)
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, one_hot_labels.float())
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        running_losses.append(running_loss/200)
        if i % 200 == 0:    # print every 200 batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    
    epoch_running_losses.append(np.mean(running_losses))

print('Finished Training')

# plot running loss to show training behaviour changes
plt.ylabel('Runinng loss')
plt.xlabel('Epoch #')
plt.xticks(range(len(epoch_running_losses)))
plt.plot(epoch_running_losses)
plt.title('Running loss of MNIST CNN for batch_size = ' + str(batch_size) + ', epochs = ' + str(epochs) + ', and lr = ' + str(learning_rate))
plt.show()

#plt.imshow(trainset.data[0], cmap='gray')
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        # use GPU
        images, labels = data[0].to(device), data[1].to(device)        
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))