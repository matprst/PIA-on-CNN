import PIL
import os, sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader

import pandas as pd

import datasets


attribute = 'Pale_Skin'
dataset = datasets.get_dataset(attribute)
dataloader = datasets.get_dataloader(dataset, attribute, 2000)


# sum_0 = 0
# sum_1 = 0
# for batch_idx, (data, target) in enumerate(dataloader):
#     total_0 = (target == 0).sum()
#     total_1 = (target == 1).sum()
#     sum_0 += total_0
#     sum_1 += total_1
#     print("batch {}, 0/1: {}/{}".format(
#         batch_idx, total_0, total_1))
#
# print(sum_0.item()/(sum_0.item()+sum_1.item()), sum_1.item()/(sum_0.item()+sum_1.item()))
# print((sum_0+sum_1).item())
#
# sys.exit()


dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=3, pin_memory=True)

# real_batch = next(iter(dataloader))
#
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()
#
# sys.exit()

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)

import torch.optim as optim

criterion = nn.MSELoss(reduction='sum')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('Training...')
for epoch in range(30):
    # break
    running_loss = 0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs).to(device)
        # print(outputs)
        # print(labels)
        loss = criterion(outputs, labels.float())
        # print(loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch+1}, {i+1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0

PATH = './test_dataloader.pth'
torch.save(net.state_dict(), PATH)
