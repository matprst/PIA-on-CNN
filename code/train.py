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


def plot_conf_interval(ax, x, evaluations, label, color, style):
    '''This function takes a 2D list of values (evaluations), computes and plots
     the averages of each list and display the standard deviation as a shaded
     area around the mean. label is a string displayed in the legend of the plot.
    '''
    mean_evals = np.mean(evaluations, axis=0)
    std_evals = np.std(evaluations, axis=0)
    err_low = mean_evals - std_evals
    err_high = mean_evals + std_evals
    ax.plot(x, mean_evals, linewidth=2, label=label, color=color, linestyle=style)
    ax.fill_between(x, err_low, err_high, color=color, alpha=0.2)

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


# n = Net()
# pytorch_total_params = sum(p.numel() for p in n.parameters() if p.requires_grad)
# print(pytorch_total_params)
# sys.exit()

def lr_tuning_experiments():

    EPOCHS = 50
    ax = plt.subplot()

    losses_reps = []

    for lr, color in zip([0.01, 0.001, 0.0001], ['blue', 'red', 'green']):
        for rep in range(3):
            net = Net().to(device)


            tic = time.time()
            criterion = nn.MSELoss(reduction='sum')
            # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            optimizer = optim.Adam(net.parameters(), lr=lr)
            losses = []
            for epoch in range(EPOCHS):
                # break
                running_loss = 0
                for i, data in enumerate(dataloader, 0):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    outputs = net(inputs).to(device)

                    loss = criterion(outputs, labels.float())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                print(f'[{epoch+1}] loss: {running_loss/(i+1):.3f}')
                losses.append(running_loss/(i+1))

            PATH = f'./64_{target_attribute}_model_{rep}_{lr}.pth'
            torch.save(net.state_dict(), PATH)

            losses_reps.append(losses)

            tac = time.time()
            print(tac - tic)

        plot_conf_interval(ax, range(EPOCHS), losses_reps, f'lr={lr}', color, 'solid')
    plt.title('LR tuning')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid('--')
    plt.show()



device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=3, pin_memory=True)
target_attribute = 'Mouth_Slightly_Open'
hidden_attribute = 'Male'
dataset = datasets.get_dataset(target_attribute)
dataloader = datasets.get_dataloader(dataset, hidden_attribute, 2000, np.array([0.5, 0.5]))




EPOCHS = 50
MODELS = 1500

def train_shadow_model(dataset, lr, hidden_attribute, class_distribution, device, size=2000, filename='test'):
    dataloader = datasets.get_dataloader(dataset, hidden_attribute, size, class_distribution)
    net = Net7().to(device)

    tic = time.time()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=lr)

    losses = []
    sum_0 = 0
    sum_1 = 0
    for epoch in range(EPOCHS):
        # break
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            total_0 = (labels == 0).sum()
            total_1 = (labels == 1).sum()
            sum_0 += total_0
            sum_1 += total_1

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs).to(device)

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        losses.append(running_loss/(i+1))

    tac = time.time()
    print(f'[{epoch+1}] loss: {running_loss/(i+1):.3f} - {int(tac-tic)} sec')

    PATH = f'./models/{filename}.pth'
    torch.save(net.state_dict(), PATH)


def train_shadow_models(dataset, device, lr=0.001, hidden_attribute='Male'):
    print(f'\nStarting training of {MODELS} shadow models on {device} - lr={lr}')
    threshold = 70
    data = {'model': [], 'male_dist': [], 'split': [], 'architecture': []}
    df = pd.read_csv('./models/models.csv')
    print(len(df))
    for model in range(len(df), len(df) + MODELS, 2):
        # if model > 4302: break
        # target property p is whether dataset is composed of more than 70% of male images
        p_true = np.random.randint(threshold, 101)
        p_false = np.random.randint(0, threshold)
        p_true_distribution = np.array([100-p_true, p_true])/100
        p_false_distribution = np.array([100-p_false, p_false])/100

        try:
            # Train model with property p
            print(f'{model} - p=True - d={p_true_distribution} - Training...')
            filename = f'{model}'
            train_shadow_model(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_true_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_true/100)
            data['split'].append(0)
            data['architecture'].append('a7')
        except:
            break

        try:
            # Train model without property p
            print(f'{model+1} - p=False - d={p_false_distribution} - Training...')
            filename = f'{model+1}'
            train_shadow_model(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_false_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_false/100)
            data['split'].append(0)
            data['architecture'].append('a7')
        except:
            break

    new_df = pd.DataFrame(data).set_index(pd.Index(range(len(df), len(df)+len(data['model']))))
    new_df.to_csv('./models/models.csv', mode='a', header=False)

def train_shadow_models_test(dataset, device, lr=0.001, hidden_attribute='Male'):
    print(f'\nStarting training of {198} shadow models test on {device} - lr={lr}')
    threshold = 70
    data = {'model': [], 'male_dist': [], 'split': [], 'architecture': []}
    df = pd.read_csv('./models/models.csv')
    print(len(df))
    for model in range(len(df), len(df) + 198, 2):
        # if model > 4302: break
        # target property p is whether dataset is composed of more than 70% of male images
        p_true = np.random.randint(threshold, 101)
        p_false = np.random.randint(0, threshold)
        p_true_distribution = np.array([100-p_true, p_true])/100
        p_false_distribution = np.array([100-p_false, p_false])/100

        try:
            # Train model with property p
            print(f'{model} - p=True - d={p_true_distribution} - Training...')
            filename = f'{model}'
            train_shadow_model(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_true_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_true/100)
            data['split'].append(1)
            data['architecture'].append('a7')
        except:
            break

        try:
            # Train model without property p
            print(f'{model+1} - p=False - d={p_false_distribution} - Training...')
            filename = f'{model+1}'
            train_shadow_model(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_false_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_false/100)
            data['split'].append(1)
            data['architecture'].append('a7')
        except:
            break

    new_df = pd.DataFrame(data).set_index(pd.Index(range(len(df), len(df)+len(data['model']))))
    new_df.to_csv('./models/models.csv', mode='a', header=False)

def train_shadow_models_valid(dataset, device, lr=0.001, hidden_attribute='Male'):
    print(f'\nStarting training of {100} shadow models valid on {device} - lr={lr}')
    threshold = 70
    data = {'model': [], 'male_dist': [], 'split': [], 'architecture': []}
    df = pd.read_csv('./models/models.csv')
    print(len(df))
    for model in range(len(df), len(df) + 100, 2):
        # if model > 4302: break
        # target property p is whether dataset is composed of more than 70% of male images
        p_true = np.random.randint(threshold, 101)
        p_false = np.random.randint(0, threshold)
        p_true_distribution = np.array([100-p_true, p_true])/100
        p_false_distribution = np.array([100-p_false, p_false])/100

        try:
            # Train model with property p
            print(f'{model} - p=True - d={p_true_distribution} - Training...')
            filename = f'{model}'
            train_shadow_model(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_true_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_true/100)
            data['split'].append(2)
            data['architecture'].append('a7')
        except:
            break

        try:
            # Train model without property p
            print(f'{model+1} - p=False - d={p_false_distribution} - Training...')
            filename = f'{model+1}'
            train_shadow_model(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_false_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_false/100)
            data['split'].append(2)
            data['architecture'].append('a7')
        except:
            break

    new_df = pd.DataFrame(data).set_index(pd.Index(range(len(df), len(df)+len(data['model']))))
    new_df.to_csv('./models/models.csv', mode='a', header=False)

# train_shadow_models(dataset, device, lr=0.001, hidden_attribute='Male')
train_shadow_models_test(dataset, device, lr=0.001, hidden_attribute='Male')
# train_shadow_models_valid(dataset, device, lr=0.001, hidden_attribute='Male')


MODELS = 1500

def train_shadow_model2(dataset, lr, hidden_attribute, class_distribution, device, size=2000, filename='test'):
    dataloader = datasets.get_dataloader(dataset, hidden_attribute, size, class_distribution)
    net = Net8().to(device)

    tic = time.time()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=lr)

    losses = []
    sum_0 = 0
    sum_1 = 0
    for epoch in range(EPOCHS):
        # break
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            total_0 = (labels == 0).sum()
            total_1 = (labels == 1).sum()
            sum_0 += total_0
            sum_1 += total_1

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs).to(device)

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        losses.append(running_loss/(i+1))

    tac = time.time()
    print(f'[{epoch+1}] loss: {running_loss/(i+1):.3f} - {int(tac-tic)} sec')

    PATH = f'./models/{filename}.pth'
    torch.save(net.state_dict(), PATH)

def train_shadow_models2(dataset, device, lr=0.001, hidden_attribute='Male'):
    print(f'\nStarting training of {MODELS} shadow models on {device} - lr={lr}')
    threshold = 70
    data = {'model': [], 'male_dist': [], 'split': [], 'architecture': []}
    df = pd.read_csv('./models/models.csv')
    print(len(df))
    for model in range(len(df), len(df) + MODELS, 2):
        # if model > 4302: break
        # target property p is whether dataset is composed of more than 70% of male images
        p_true = np.random.randint(threshold, 101)
        p_false = np.random.randint(0, threshold)
        p_true_distribution = np.array([100-p_true, p_true])/100
        p_false_distribution = np.array([100-p_false, p_false])/100

        try:
            # Train model with property p
            print(f'{model} - p=True - d={p_true_distribution} - Training...')
            filename = f'{model}'
            train_shadow_model2(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_true_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_true/100)
            data['split'].append(0)
            data['architecture'].append('a8')
        except:
            break

        try:
            # Train model without property p
            print(f'{model+1} - p=False - d={p_false_distribution} - Training...')
            filename = f'{model+1}'
            train_shadow_model2(dataset=dataset,
                               lr=lr,
                               hidden_attribute=hidden_attribute,
                               class_distribution=p_false_distribution,
                               device=device,
                               size=2000,
                               filename=filename)
            data['model'].append(f'{filename}.pth')
            data['male_dist'].append(p_false/100)
            data['split'].append(0)
            data['architecture'].append('a8')
        except:
            break

    new_df = pd.DataFrame(data).set_index(pd.Index(range(len(df), len(df)+len(data['model']))))
    new_df.to_csv('./models/models.csv', mode='a', header=False)

train_shadow_models2(dataset, device, lr=0.001, hidden_attribute='Male')
