import time
import numpy as np
import pandas as pd
import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, WeightedRandomSampler, RandomSampler, DataLoader

class ShadowDataset(Dataset):
    map_split = {'train':0, 'test':1, 'valid':2}

    def __init__(self, csv_file, root_dir, split='train', architecture='a1', fcn=False, conv=False, device='cpu'):
        df = pd.read_csv(csv_file, index_col=0)
        mask_split = (df['split'] == ShadowDataset.map_split[split])
        mask_architecture = (df['architecture'] == architecture)
        df = df[mask_split & mask_architecture]

        self.proportions_frame = df.set_index(pd.Index(range(0, len(df))))

        print('training set size:', len(self.proportions_frame))
        self.root_dir = root_dir

        self.fcn = fcn # only use the weights of the linear layers
        self.conv = conv # only use the weights of the convolution layers
        self.device = device

    def __len__(self):
        return len(self.proportions_frame)

    def __getitem__(self, index):
        model_name = f'{self.proportions_frame.iloc[index].model}'
        model_path = os.path.join(self.root_dir, model_name)

        flattened_weights = ShadowDataset.get_weights(model_path, device=self.device, fcn=self.fcn, conv=self.conv)

        label = torch.Tensor([1.0]) if self.proportions_frame.iloc[index].male_dist > 0.7 else torch.Tensor([0.0])

        return flattened_weights, label

    def get_weights(path, device='cpu', fcn=False, conv=False):
        weights = torch.Tensor().to(device)
        for k, v in torch.load(path, map_location=torch.device(device)).items():
            if fcn: # only use fully connected weights
                if 'fc' in k:
                    weights = torch.cat((weights, v.view(-1).to(device)))
            elif conv: # only use weights from convolution layers
                if 'conv' in k:
                    weights = torch.cat((weights, v.view(-1).to(device)))
            else: # use all the weights
                weights = torch.cat((weights, v.view(-1).to(device)))
        return weights

class SelectAttr(object):
    list_attr = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young'.split()

    def __init__(self, attr_name):
        assert attr_name in self.list_attr
        self.attr_name = attr_name

    def __call__(self, sample):
        # return sample[self.list_attr.index(self.attr_name)]
        return sample.narrow(-1, self.list_attr.index(self.attr_name), 1)


def get_dataloader(dataset, property, size, class_proportions=None, batch_size=64):
    '''
    Return a dataloader for a given attribute of a random subset of the CelebA
    dataset. The dataloader make sure the samples follow the given class
    distribution.
    Parameters:
        - dataset: pytorch dataset from which to sample.
        - property: string. Name of the CelebA attribute
        - size: int. Number of samples
        - class_proportions: numpy array. Contains 2 elements which are the
        probabilities of drawing a sample without or with the given property.
        Elements should sum to 1.
        - batch_size: int. Number of samples contained in each batch.
    '''
    if property is None:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    elif class_proportions is None:
        sampler = RandomSampler(dataset, replacement=True, num_samples=size)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        # Get the target-values of the given property
        property_index = SelectAttr.list_attr.index(property)
        targets = dataset.attr[:,property_index]

        # Weight each sample using the given class-proportions
        _, class_count = np.unique(targets, return_counts=True)
        # print(f'CelebA {property} {_} - {class_count/np.sum(class_count)}')
        weight = class_proportions / class_count
        samples_weight = torch.tensor([weight[t] for t in targets])

        # Create the dataloader using the weights of each sample
        sampler = WeightedRandomSampler(samples_weight, size)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    return loader

def get_dataset(property, split='train'):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    target_transform = transforms.Compose([
        SelectAttr(property)
    ])

    # Load CelebA
    dataset = torchvision.datasets.CelebA('data/celeba-dataset/', split=split, transform=transform, target_transform=target_transform)

    for property in SelectAttr.list_attr:
        property_index = SelectAttr.list_attr.index(property)
        targets = dataset.attr[:,property_index]

        # Weight each sample using the given class-proportions
        _, class_count = np.unique(targets, return_counts=True)
        print(f'{property} - {class_count/np.sum(class_count)}')

    return dataset

if __name__ == '__main__':
    attribute = 'Young'
    dataset = get_dataset(attribute)
    for i in range(10):
        tic = time.time()
        dataloader = get_dataloader(dataset, attribute, 1000, np.array([0.5, 0.5]))
        toc = time.time()
        print(toc - tic)


    sum_0 = 0
    sum_1 = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        total_0 = (target == 0).sum()
        total_1 = (target == 1).sum()
        sum_0 += total_0
        sum_1 += total_1
        print("batch {}, 0/1: {}/{}".format(
        batch_idx, total_0, total_1))

    print(sum_0.item()/(sum_0.item()+sum_1.item()), sum_1.item()/(sum_0.item()+sum_1.item()))
    print((sum_0+sum_1).item())
