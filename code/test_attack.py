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


# class ShadowDataset(Dataset):
#     def __init__(self, csv_file, root_dir):
#         self.proportions_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#
#     def __len__(self):
#         return len(self.proportions_frame)
#
#     def __getitem__(self, index):
#         model_name = f'{self.proportions_frame.iloc[index].model}.pth'
#         model_path = os.path.join(self.root_dir, model_name)
#
#         flattened_weights = ShadowDataset.get_weights(model_path)
#
#         label = torch.Tensor([1.0]) if self.proportions_frame.iloc[index].male_dist > 0.7 else torch.Tensor([0.0])
#
#         return flattened_weights, label
#
#     def get_weights(path, device='cpu'):
#         weights = torch.Tensor().to(device)
#         for k, v in torch.load(path).items():
#             weights = torch.cat((weights, v.view(-1).to(device)))
#         return weights

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

def evaluate(path):
    net = AttackNet(in_dim=658825)
    net.load_state_dict(torch.load(path))

    # outputs = net(images)
    # # print(outputs)
    # print('yo')
    # predicted = (outputs.view(-1)>0.5).float()
    # print(predicted)
    # print('actual')
    # print(labels)

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            # break
            images, labels = data
            # print(images.size())
            # print(labels.size())
            outputs = net(images)
            # print(outputs.size())
            labels = labels.view(-1)
            predicted = (outputs.view(-1)>0.5).float()
            # print(labels)
            # print(predicted)
            print(f'total={total}, labels.size(0)={labels.size(0)}')
            total += labels.size(0)
            # print(predicted == labels)
            correct += (predicted == labels).sum().item()

            print(predicted==labels)

            # print(predicted)
            # print(labels)
            # print(predicted == labels)
            tp, tn, fp, fn = get_eval_metrics(predicted, labels)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

    if total_tp + total_fp != 0:
        precision = total_tp / (total_tp + total_fp)
    else: precision = '//'
    recall = total_tp / (total_tp + total_fn)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    print(total)
    # print(f'rep {rep} - lr {lr}')
    print(f'\tAccuracy: {accuracy} %')
    print(f'\tPrecision: {precision} %')
    print(f'\tRecall: {recall} %')

# for lr in [0.001, 0.001, 0.0001]:
#     for rep in range(3):
#         # PATH = f'./64_Mouth_Slightly_Open_model_{rep}_{lr}.pth'
#         path = 'shadow_Mouth_Slightly_Open_model_0.001.pth'
#         evaluate(path)

# shadow_dataset = ShadowDataset('./models/backup/valid_attr.csv', './models/backup/')
shadow_dataset = ShadowDataset('./models/models.csv', './models/', split='valid', architecture='a2')
testloader = DataLoader(shadow_dataset, batch_size=32, shuffle=True)

device = 'cuda:0'
# attack_model = AttackNet().to(device)

path = f'./models/attack_model2.pth'
print(path)
evaluate(path)


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

# tic = time.time()
# criterion = nn.MSELoss(reduction='sum')
# optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
# # print(f'{rep} - lr={lr} - Training...')
# losses = []
# for epoch in range(10):
#     # break
#     running_loss = 0
#     for i, data in enumerate(dataloader, 0):
#         inputs, labels = data
#         print(inputs.size())
#         print(labels.size())
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#
#         optimizer.zero_grad()
#
#         outputs = attack_model(inputs).to(device)
#
#         loss = criterion(outputs, labels.float())
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#     print(f'[{epoch+1}] loss: {running_loss/(i+1):.3f}')
#     # losses.append(running_loss/(i+1))
# tac = time.time()
# print(tac - tic)
# PATH = f'./models/attack_model.pth'
# torch.save(attack_model.state_dict(), PATH)
