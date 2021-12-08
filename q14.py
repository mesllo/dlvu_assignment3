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
variable = True

# no auto-augment policy autoperformed default or random rotation
policy = transforms.AutoAugmentPolicy.CIFAR10

train_loaders = {}; val_loaders = {}; test_loaders = {}
if variable:
    data_path = './data/mnist-varres'
    if augment:
        transforms.Compose([transforms.Grayscale(num_output_channels=1),
                            # transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
                            transforms.RandomRotation(degrees=(-30, 30)),
                            # best augmentation performed equally as well as no augmentation,
                            # transforms.RandomAffine(degrees=(0, 30), translate=(0.1, 0.3), scale=(1, 1.5)),
                            # transforms.AutoAugment(policy),
                            transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.ToTensor()])
    train = datasets.ImageFolder(root=data_path + '/train', transform=transform)
    test = datasets.ImageFolder(root=data_path + '/test', transform=transform)

    resolutions = {}; train_sets = {}; val_sets = {}; test_sets = {}; testres = {}
    for index, image in enumerate(train):
        if image[0].shape in resolutions:
            resolutions[image[0].shape].append(index)
        else:
            resolutions[image[0].shape] = [index]

    for index, image in enumerate(test):
        if image[0].shape in testres:
            testres[image[0].shape].append(index)
        else:
            testres[image[0].shape] = [index]

    for key in resolutions:
        train_sets[key] = torch.utils.data.Subset(train, resolutions[key])
        train_sets[key], val_sets[key] = torch.utils.data.random_split(train_sets[key], [round(len(train_sets[key])*0.80), len(train_sets[key]) - round(len(train_sets[key])*0.80)])

        train_loaders[key] = torch.utils.data.DataLoader(train_sets[key], batch_size=batch_size,
                                                                   shuffle=True, num_workers=0)
        val_loaders[key] = torch.utils.data.DataLoader(val_sets[key], batch_size=batch_size,
                                                               shuffle=True, num_workers=0)

        test_sets[key] = torch.utils.data.Subset(test, testres[key])
        test_loaders[key] = torch.utils.data.DataLoader(test_sets[key], batch_size=batch_size,
                                               shuffle=True, num_workers=0)
else:
    if augment:
        transform = transforms.Compose(
            [  # transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
                transforms.RandomRotation(degrees=(-30, 30)),
                # best augmentation performed equally as well as no augmentation,
                # transforms.RandomAffine(degrees=(0, 30), translate=(0.1, 0.3), scale=(1, 1.5)),
                # transforms.AutoAugment(policy),
                transforms.ToTensor()])
    else:
        transform = transforms.ToTensor()

    train = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
    test = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transforms.ToTensor())
    train_set, val_set = torch.utils.data.random_split(train, [50000, 10000])

    # dataloaders
    train_loaders['vanilla'] = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                         shuffle=True, num_workers=0)
    val_loaders['vanilla'] = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                     shuffle=True, num_workers=0)
    test_loaders['vanilla'] = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                               shuffle=True, num_workers=0)


# define CNN
class Net(nn.Module):
    def __init__(self, N=64, global_mean_pool=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=N, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(N, 10)

        # other attributes
        self.N = N
        self.has_global_mean_pool = global_mean_pool

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # get dim of square matrix and apply global pool
        n = (x.size()[-1])
        gpool = nn.MaxPool2d(n, n)
        if self.has_global_mean_pool:
            gpool = nn.AvgPool2d(n, n)
        x = gpool(x)

        # squeeze 1-dimensions
        x = torch.squeeze(x)

        x = self.fc1(x)
        return x


net = Net(N=81, global_mean_pool=True)
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


# define validation accuracy list for plotting
val_accs = []

# checkpoint size of data to plot
d = 100

# train loop
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0

    for var_resolution in train_loaders:
        # iter validation loader
        trainloader = train_loaders[var_resolution]
        vals = iter(val_loaders[var_resolution])

        for i, data in enumerate(trainloader):
            # use GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # keep loading validation sets throughout training
            try:
                val_data = next(vals)
            except StopIteration:
                vals = iter(val_loaders[var_resolution])
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

            if i % d == 0 and i != 0:  # print every 100 batches
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
plt.xlabel('Batch')
# range(0, int(50000/batch_size), d)) is not working for the ticks
#plt.xticks(range(len(running_losses)))
plt.plot(running_losses)
plt.title(
    'Running loss for batch_size = ' + str(batch_size) + ', epochs = ' + str(epochs) + ', and lr = ' + str(
        learning_rate))
plt.show()

# plot validation accuracy to show training behaviour changes
plt.ylabel('Validation accuracy')
plt.xlabel('Batches')
# range(0, int(50000/batch_size), d)) is not working for the ticks
#plt.xticks(range(len(val_accs)))
plt.plot(val_accs)
plt.title('Validation accuracy for batch_size = ' + str(batch_size) + ', epochs = ' + str(
    epochs) + ', and lr = ' + str(learning_rate))
plt.show()

# plt.imshow(trainset.data[0], cmap='gray')
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for res in test_loaders:
        testloader = test_loaders[res]

        for i, data in enumerate(testloader):
            # use GPU
            inputs, labels = data[0].to(device), data[1].to(device)
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
