import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
# Create dummy data with class imbalance 99 to 1
numDataPoints = 1000
data_dim = 5
bs = 100
data = torch.randn(numDataPoints, data_dim)
target = torch.cat((torch.zeros(int(numDataPoints * 0.99), dtype=torch.long),
                    torch.ones(int(numDataPoints * 0.01), dtype=torch.long)))

print('target train 0/1: {}/{}'.format(
    (target == 0).sum(), (target == 1).sum()))

# Create subset indices
subset_idx = torch.cat((torch.arange(100), torch.arange(-5, 0)))
print(subset_idx)

# Compute samples weight (each sample should get its own weight)
class_sample_count = torch.tensor(
    [(target[subset_idx] == t).sum() for t in torch.unique(target, sorted=True)])
print(class_sample_count)
weight = 1. / class_sample_count.float()
samples_weight = torch.tensor([weight[t] for t in target[subset_idx]])
print(samples_weight)

print(data.size())
print(data[subset_idx].size())
# Create sampler, dataset, loader
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
train_dataset = torch.utils.data.TensorDataset(
    data[subset_idx], target[subset_idx])
train_loader = DataLoader(
    train_dataset, batch_size=bs, num_workers=1, sampler=sampler)

# Iterate DataLoader and check class balance for each batch
for i, (x, y) in enumerate(train_loader):
    # print(x, y)
    print("batch index {}, 0/1: {}/{}".format(
        i, (y == 0).sum(), (y == 1).sum()))
