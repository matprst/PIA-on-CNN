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
        for k, v in torch.load(path).items():
            weights = torch.cat((weights, v.view(-1).to(device)))
        return weights

shadow_dataset = ShadowDataset('./models/models.csv', './models/', architecture='a2')
dataloader = DataLoader(shadow_dataset, batch_size=32, shuffle=True)

class AttackNet(nn.Module):
    def __init__(self, in_dim=337721, out_dim=1):
        super(AttackNet, self).__init__()
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


device = 'cuda:0'
attack_model = AttackNet(in_dim=658825).to(device)

# for i_batch, batch in enumerate(dataloader):
#     data, labels = batch
#     print(data.size())
#     print(labels.size())
#     outputs = attack_model(data)
#     print(outputs.size())
#     print(outputs)
#     break
#
# net = Net().to(device)

tic = time.time()
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
# print(f'{rep} - lr={lr} - Training...')
losses = []
for epoch in range(10):
    # break
    running_loss = 0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        # print(inputs.size())
        # print(labels.size())
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = attack_model(inputs).to(device)
        # print(outputs.size())

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'[{epoch+1}] loss: {running_loss/(i+1):.3f}')
    # losses.append(running_loss/(i+1))
tac = time.time()
print(tac - tic)
PATH = f'./models/attack_model2.pth'
torch.save(attack_model.state_dict(), PATH)
