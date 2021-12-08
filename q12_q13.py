import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# params
batch_size = 16
epochs = 3
learning_rate = 0.001
augment = False
# no auto-augment policy autoperformed default or random rotation
policy = transforms.AutoAugmentPolicy.CIFAR10

# dataset load
data_path = './data/mnist-varres'

im_height = 28
im_width = 28
transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                 transforms.Resize((im_height, im_width)),
                                 transforms.ToTensor()])

# datasets
train = datasets.ImageFolder(root=data_path+'/train', transform=transforms)

train_set, val_set = torch.utils.data.random_split(train, [50000, 10000])

test = datasets.ImageFolder(root=data_path+'/test', transform=transforms)

# loaders
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                        shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                         shuffle=True, num_workers=0)


# define CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * math.floor(im_height/8) * math.floor(im_width/8), 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x


net = Net()
net.to(device)

no_of_params = 0
for param in net.parameters():
    no_of_params += np.prod(np.array(list(param.shape)))

print('Number of parameters: ' + str(no_of_params))

# define loss function
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

# define optimizer
optimizer = optim.AdamW(net.parameters(), lr=0.001)

# define loss lists to collect errors for plotting
running_losses = []
epoch_running_losses = []

# iter validation loader
vals = iter(valloader)

# define validation accuracy list for plotting
val_accs = []

# checkpoint size of data to plot
d = 100

# train loop
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, data in enumerate(trainloader):
        # use GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # keep loading validation sets throughout training
        try:
            val_data = next(vals)
        except StopIteration:
            vals = iter(valloader)
            val_data = next(vals)
        val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)

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

        if i % d == 0 and i is not 0:  # print every d batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / d))
            running_losses.append(running_loss / d)
            running_loss = 0.0

            ## Compute validation accuracy
            o = net(val_inputs)
            _, predicted = torch.max(o.data, 1)  # outputs max and max_indices

            num_correct = (predicted == val_labels).sum()
            acc = num_correct / val_labels.shape[0]
            val_accs.append(float(acc))

            print(f'       accuracy: {acc:.4}')

    epoch_running_losses.append(np.mean(running_losses))

print('Finished Training')

# plot running loss to show training behaviour changes
plt.ylabel('Running loss')
plt.xlabel('Batches')
# range(0, int(50000/batch_size), d)) is not working for the ticks
#plt.xticks(range(len(running_losses)))
plt.plot(running_losses)
plt.title(
    'Running loss of MNIST CNN for batch_size = ' + str(batch_size) + ', epochs = ' + str(epochs) + ', and lr = ' + str(
        learning_rate))
plt.show()

# plot validation accuracy to show training behaviour changes
plt.ylabel('Validation accuracy')
plt.xlabel('Batches')
# range(0, int(50000/batch_size), d)) is not working for the ticks
#plt.xticks(range(len(val_accs)))
plt.plot(val_accs)
plt.title('Validation accuracy of MNIST CNN for batch_size = ' + str(batch_size) + ', epochs = ' + str(
    epochs) + ', and lr = ' + str(learning_rate))
plt.show()

# plt.imshow(trainset.data[0], cmap='gray')
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