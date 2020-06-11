import time
import PIL
import os, sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
import torch.optim as optim

import pandas as pd

import datasets
import sys
import os


class ShadowDataset(Dataset):
    map_split = {'train':0, 'test':1, 'valid':2}

    def __init__(self, csv_file, root_dir, split='train', architecture='a1'):
        df = pd.read_csv(csv_file, index_col=0)
        mask_split = (df['split'] == ShadowDataset.map_split[split])
        mask_architecture = (df['architecture'] == architecture)
        df = df[mask_split & mask_architecture]

        self.proportions_frame = df.set_index(pd.Index(range(0, len(df))))

        print(self.proportions_frame)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.proportions_frame)

    def __getitem__(self, index):
        model_name = f'{self.proportions_frame.iloc[index].model}'
        model_path = os.path.join(self.root_dir, model_name)

        flattened_weights = ShadowDataset.get_weights(model_path)

        label = torch.Tensor([1.0]) if self.proportions_frame.iloc[index].male_dist > 0.7 else torch.Tensor([0.0])

        return flattened_weights, label

    def get_weights(path, device='cpu'):
        weights = torch.Tensor().to(device)
        for k, v in torch.load(path, map_location=torch.device(device)).items():
            weights = torch.cat((weights, v.view(-1).to(device)))
        return weights


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 13**2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # print(x.size())
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.size())
        x = x.view(-1, 16 * 13**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __str__(self):
        return 'a1'

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 30**2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # print(x.size())
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        # x = self.pool2(F.relu(self.conv2(x)))
        # print(x.size())
        x = x.view(-1, 6 * 30**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __str__(self):
        return 'a2'

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 4**2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # print(x.size())
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(-1, 32 * 4**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __str__(self):
        return 'a3'

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 4**2, 84)
        # self.fc2 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 1)

    def forward(self, x):
        # print(x.size())
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(-1, 32 * 4**2)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

    def __str__(self):
        return 'a4'

class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 4**2, 1)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc2 = nn.Linear(84, 1)

    def forward(self, x):
        # print(x.size())
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(-1, 32 * 4**2)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc1(x)
        return x

    def __str__(self):
        return 'a5'

class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 13**2, 84)
        # self.fc2 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 1)

    def forward(self, x):
        # print(x.size())
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.size())
        x = x.view(-1, 16 * 13**2)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

    def __str__(self):
        return 'a6'

class Net7(nn.Module):
    def __init__(self):
        super(Net7, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 13**2, 1)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc2 = nn.Linear(84, 1)

    def forward(self, x):
        # print(x.size())
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.size())
        x = x.view(-1, 16 * 13**2)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc1(x)
        return x

    def __str__(self):
        return 'a7'

class Net8(nn.Module):
    def __init__(self):
        super(Net8, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 30**2, 84)
        # self.fc2 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 1)

    def forward(self, x):
        # print(x.size())
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        # x = self.pool2(F.relu(self.conv2(x)))
        # print(x.size())
        x = x.view(-1, 6 * 30**2)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

    def __str__(self):
        return 'a8'

class AttackNet(nn.Module):
    def __init__(self, in_dim=337721, out_dim=1):
        super(AttackNet, self).__init__()
        print(in_dim)
        self.fc1 = nn.Linear(in_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 5)
        self.fc4 = nn.Linear(5, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def number_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# path = './models/shadow_0_0.pth'
# t = torch.Tensor()
# for k, v in torch.load(path).items():
#     # print(k, v)
#     t = torch.cat((t, v.view(-1).to('cpu')))
# print(t)
# print(t.size())
#
# net = Net()
# net.load_state_dict(torch.load(path))
# # print(net)
# # pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# # print(pytorch_total_params)
# # total = 0
# weights = torch.Tensor()
# for param in net.parameters():
# #     # print(param)
# #     # print(param.data.size())
# #     # print(param.data.view(-1))
#     weights = torch.cat((weights, param.data.view(-1)))
# #     # total += weights[0]
# #     # print(weights)
# print(weights)
# print(weights.size())
# assert torch.all(t.eq(weights))
# # print(list(net.parameters())
# # print(net.)



def train_attack(model, epoch=10, filename='test'):
    print(device)
    tic = time.time()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
    # print(f'{rep} - lr={lr} - Training...')
    losses = []
    for epoch in range(epoch):
        # break
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            # print(inputs.size())
            # print(labels.size())
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs).to(device)
            # print(outputs.size())

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'[{epoch+1}] loss: {running_loss/(i+1):.3f}')
        # losses.append(running_loss/(i+1))
    tac = time.time()
    print(tac - tic)
    PATH = f'./models/{filename}.pth'
    torch.save(model.state_dict(), PATH)


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)
# attack_model = AttackNet(in_dim=number_param(Net())).to(device)
#
# train_attack(attack_model, 'new_attack')

EPOCHS = 20
for target_model in [Net7()]:
    for rep in range(5, 10):
        shadow_dataset = ShadowDataset('./models/models.csv', './models/', split='train', architecture=str(target_model))
        dataloader = DataLoader(shadow_dataset, batch_size=32, shuffle=True)

        params = number_param(target_model)
        print(params)
        attack_model = AttackNet(in_dim=params).to(device)
        filename = f'{target_model}-{rep}'
        print(filename)
        train_attack(attack_model, EPOCHS, filename)
