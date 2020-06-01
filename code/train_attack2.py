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


shadow_dataset_train = ShadowDataset('./models/models.csv', './models/', split='train', architecture='a2')
dataloader_train = DataLoader(shadow_dataset_train, batch_size=32, shuffle=True)

shadow_dataset_valid = ShadowDataset('./models/models.csv', './models/', split='valid', architecture='a2')
dataloader_valid = DataLoader(shadow_dataset_valid, batch_size=32, shuffle=True)

def get_eval_metrics(predicted, actual):
    '''
    Return the true positive, true negative, false positive and false negative
    counts given the predicted and actual values.
    '''
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for p, a in zip(predicted, actual):
        # print(p, a)
        if p == 1:
            if a == 1:
                tp += 1
            else:
                fp += 1
        elif p == 0:
            if a == 1:
                fn += 1
            elif a == 0:
                tn += 1
    # print(tp, tn, fp, fn)
    return tp, tn, fp, fn

def evaluate(model, dataloader):
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.view(-1)

            outputs = model(inputs)
            predicted = (outputs.view(-1)>0.5).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            tp, tn, fp, fn = get_eval_metrics(predicted, labels)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    return accuracy



device = 'cuda:0'
# attack_model = AttackNet().to(device)
attack_model = AttackNet(in_dim=658825).to(device)

tic = time.time()
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

losses = []
for epoch in range(10):
    running_loss = 0
    for i, data in enumerate(dataloader_train, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = attack_model(inputs).to(device)

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'[{epoch+1}] - loss: {running_loss/(i+1):.3f} - valid acc: {evaluate(attack_model, dataloader_valid)}')
    # losses.append(running_loss/(i+1))
tac = time.time()
print(tac - tic)
PATH = f'./models/attack_model2.pth'
torch.save(attack_model.state_dict(), PATH)
