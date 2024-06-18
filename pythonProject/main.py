import torch
import torch.nn as nn
import torchvision.datasets as dsets
from matplotlib import pyplot as plt
from skimage import transform
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 1
batch_size = 100
learning_rate = 0.001

class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28)
        self.Y = np.array(data.iloc[:, 0])
        del data
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = self.X[idx]
        label = self.Y[idx]

        if self.transform:
            item = self.transform(item)

        return (item, label)

train_dataset = FashionMNISTDataset(csv_file='C://Users//adamk//Desktop//BIAIproject//fashion-mnist_train.csv')
test_dataset = FashionMNISTDataset(csv_file='C://Users//adamk//Desktop//BIAIproject//fashion-mnist_test.csv')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

labels_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
              7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}

class CNNConfig1(nn.Module):
    def __init__(self):
        super(CNNConfig1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class CNNConfig2(nn.Module):
    def __init__(self):
        super(CNNConfig2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(3 * 3 * 128, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class CNNConfig3(nn.Module):
    def __init__(self):
        super(CNNConfig3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25))
        self.fc1 = nn.Linear(3 * 3 * 64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

cnn = CNNConfig1().to(device)
# cnn = CNNConfig2().to(device)
# cnn = CNNConfig3().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

losses = []
accuracies = []

start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    cnn.train()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.float()).to(device)
        labels = Variable(labels).to(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())  # Use item() to get the scalar value

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))

    accuracy = 100 * correct / total
    accuracies.append(accuracy)
    epoch_end_time = time.time()
    print('Epoch [%d/%d] Train Accuracy: %.4f %% Time Taken: %.2f seconds' % (epoch + 1, num_epochs, accuracy, epoch_end_time - epoch_start_time))

cnn.eval()
correct = 0
total = 0
all_labels = []
all_predicted = []

test_start_time = time.time()

for images, labels in test_loader:
    images = Variable(images.float()).to(device)
    labels = labels.to(device)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    all_labels.extend(labels.cpu().numpy())
    all_predicted.extend(predicted.cpu().numpy())

test_end_time = time.time()
test_accuracy = 100 * correct / total
print('Test Accuracy of the model on the 10000 test images: %.4f %% Time Taken: %.2f seconds' % (test_accuracy, test_end_time - test_start_time))

end_time = time.time()
total_time = end_time - start_time
print('Total Time Taken: %.2f seconds' % total_time)

# Plotting the loss and accuracy
# Plot loss
losses_in_epochs = losses[0::len(train_loader)]
plt.style.use('ggplot')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.title(f'Loss plot (Batch size: {batch_size}, Epochs: {num_epochs})')
plt.plot(losses_in_epochs)

plt.show()

# Plot accuracy
plt.style.use('ggplot')
plt.xlabel('Epoch #')
plt.ylabel('Accuracy')
plt.title(f'Accuracy plot (Batch size: {batch_size}, Epochs: {num_epochs})')
plt.plot(accuracies)

plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(labels_map.values()))
disp.plot(cmap='viridis')
plt.xticks(rotation=45)
plt.show()

def plot_kernels(tensor, num_cols=6):
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(num_kernels):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.imshow(tensor[i][0, :, :], cmap='gray')
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
