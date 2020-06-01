import PIL
import os, sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import pandas as pd

from collections import Counter

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
        # print(sample[self.list_attr.index(self.attr_name):])
        # print(sample[self.list_attr.index(self.attr_name)])
        # assert sample[self.list_attr.index(self.attr_name):][0] == sample[self.list_attr.index(self.attr_name)]
        return sample[self.list_attr.index(self.attr_name):]

target_transform = transforms.Compose([
    SelectAttr('Young')
])

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# dataset = torchvision.datasets.CelebA('../datasets/celeba/celeba-dataset/', split='all', transform=transform, target_transform=target_transform)

split_map = {
    "train": 0,
    "valid": 1,
    "test": 2,
    "all": None,
}

splits = pd.read_csv("../datasets/celeba/celeba-dataset/celeba/list_eval_partition.txt", delim_whitespace=True, header=None, index_col=0)
mask = slice(None) if None is None else (splits[1] == 2)

attr = pd.read_csv("../datasets/celeba/celeba-dataset/celeba/list_attr_celeba.txt", delim_whitespace=True, header=1)


common = Counter(attr[mask]['Young'].values).most_common(2)
print(common[0][1]/len(attr[mask]))
print(common[1][1]/len(attr[mask]))
