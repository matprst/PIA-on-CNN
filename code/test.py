import sys

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

class SelectAttr(object):
    '''
    '''

    list_attr = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young'.split()

    def __init__(self, attr_name):
        # print(self.list_attr)
        assert attr_name in self.list_attr
        self.attr_name = attr_name

    def __call__(self, sample):
        # print(sample[self.list_attr.index(self.attr_name)])
        # print(self.list_attr.index(self.attr_name))
        # print(sample[self.list_attr.index(self.attr_name)])
        # print('call')
        # print(sample)
        # print('return call')
        # print(sample[self.list_attr.index(self.attr_name)])

        return sample[self.list_attr.index(self.attr_name)]

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

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.CIFAR10(
#     root='./data',
#     train=True,
#     download=True,
#     transform=transform)
#
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(
#     root='./data',
#     train=False,
#     download=True,
#     transform=transform)
#
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

target_transform = transforms.Compose([
    SelectAttr('Mouth_Slightly_Open')
])

testset = torchvision.datasets.CelebA('../datasets/celeba/celeba-dataset/', split='test', transform=transform, target_transform=target_transform)
print(len(testset))
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

classes = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young'.split()

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    # print(img)
    # img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(testloader)
images, labels = dataiter.next()

# sys.exit()

# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# imshow(torchvision.utils.make_grid(images))

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
    net = Net7()
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
            # print(labels.size())
            outputs = net(images)
            # print(outputs.size())
            predicted = (outputs.view(-1)>0.5).float()
            # print(predicted.size())
            # print(labels)
            # print(predicted)
            total += labels.size(0)
            # print(predicted == labels)
            correct += (predicted == labels).sum().item()
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
    # print(f'rep {rep} - lr {lr}')
    print(f'\tAccuracy: {accuracy} %')
    print(f'\tPrecision: {precision} %')
    print(f'\tRecall: {recall} %')

# for lr in [0.001, 0.001, 0.0001]:
#     for rep in range(3):
#         # PATH = f'./64_Mouth_Slightly_Open_model_{rep}_{lr}.pth'
#         path = 'shadow_Mouth_Slightly_Open_model_0.001.pth'
#         evaluate(path)


path = f'./models/10130.pth'
print(path)
evaluate(path)
# path = f'./models/9001.pth'
# print(path)
# evaluate(path)
