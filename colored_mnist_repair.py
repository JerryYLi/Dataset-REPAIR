import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms

from utils.datasets import *
from utils.measure import *
from utils.repair import *
from utils.models import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)

# dataloading options
parser.add_argument('--color-std', type=float, default=0.1)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--num-workers', default=4, type=int)

# learning hyperparameters
parser.add_argument('--model', default='lenet', choices=['lenet', 'mlp'], type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--lr-w', default=10, type=float)
parser.add_argument('--epochs', default=200, type=int)

# resampling strategies
parser.add_argument('--sampling', default='threshold', choices=['threshold', 'rank', 'cls_rank', 'sample', 'uniform'], type=str)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--keep-ratio', type=float, default=0.5)
args = parser.parse_args()

torch.cuda.set_device(args.gpu)

# Sampling strategies
def get_keep_idx(w, cls_idx, mode='threshold'):
    # strategy 1: fixed threshold
    if mode == 'threshold':
        keep_idx = (w > args.threshold).nonzero().cpu().squeeze()

    # strategy 2: top k% examples
    elif mode == 'rank':
        keep_examples = round(args.keep_ratio * len(w))
        keep_idx = w.sort(descending=True)[1][:keep_examples].cpu()

    # strategy 3: top k% examples each class
    elif mode == 'cls_rank':
        keep_idx_list = []
        for c in range(10):
            c_idx = cls_idx[c].nonzero().squeeze()
            keep_examples = round(args.keep_ratio * len(c_idx))
            sort_idx = w[c_idx].sort(descending=True)[1]
            keep_idx_list.append(c_idx[sort_idx][:keep_examples])
        keep_idx = torch.cat(keep_idx_list).cpu()

    # strategy 4: sampling according to weights
    elif mode == 'sample':
        keep_idx = torch.bernoulli(w).nonzero().cpu().squeeze()

    # strategy 5: random uniform sampling
    elif mode == 'uniform':
        keep_examples = round(args.keep_ratio * len(w))
        keep_idx = torch.randperm(len(w))[:keep_examples]

    return keep_idx

# load data
train_set = datasets.MNIST('/data4/yili/github/data/', train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.MNIST('/data4/yili/github/data/', train=False, download=True, transform=transforms.ToTensor())

# biased datasets, i.e. colored mnist
print('Coloring MNIST dataset with standard deviation = {:.2f}'.format(args.color_std))
colored_train_set = ColoredDataset(train_set, classes=10, colors=[0, 1], std=args.color_std)
train_loader = DataLoader(colored_train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
colored_test_set = ColoredDataset(test_set, classes=10, colors=colored_train_set.colors, std=args.color_std)
test_loader = DataLoader(colored_test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# ground-truth datasets, i.e. grayscale mnist
gt_train_set = ColoredDataset(train_set, classes=10, colors=[1, 1], std=0)
gt_train_loader = DataLoader(gt_train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
gt_test_set = ColoredDataset(test_set, classes=10, colors=[1, 1], std=0)
gt_test_loader = DataLoader(gt_test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# measure bias & generalization before resampling
color_fn = lambda x: x.view(x.size(0), x.size(1), -1).max(2)[0]  # color of digit
color_dim = 3  # rgb
bias_0 = measure_bias(train_loader, test_loader, color_fn, color_dim, args.epochs, args.lr)[0]
model = create_mnist_model(args.model)
gt_acc_0 = measure_generalization(train_loader, [test_loader, gt_test_loader], model, args.epochs, args.lr)[1]
print('Color bias before resampling = {:.3f}'.format(bias_0 + 0))
print('Generalization accuracy before resampling = {:.2%}'.format(gt_acc_0))

# learn resampling weights
repair_dataset = IndexedDataset(ConcatDataset([colored_train_set, colored_test_set]))
train_loader = DataLoader(repair_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
w, q, cls_idx, cls_w, bias = repair(train_loader, color_fn, color_dim, args.epochs, args.lr, args.lr_w)

# plot histogram of resampling weights
sns.histplot(w.cpu(), bins=100, kde=False)
plt.xlabel('Resampling weights')
plt.ylabel('# Examples')
plt.show()

# perform resampling
print('Resampling strategy:', args.sampling)
keep_idx = get_keep_idx(w, cls_idx, mode=args.sampling)
keep_idx_train = keep_idx[keep_idx < len(colored_train_set)]
keep_idx_test = keep_idx[keep_idx >= len(colored_train_set)] - len(colored_train_set)
print('Keep examples: {}/{} ({:.2%})'.format(len(keep_idx), len(w), len(keep_idx) / len(w)))
train_loader = DataLoader(Subset(colored_train_set, keep_idx_train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_loader = DataLoader(Subset(colored_test_set, keep_idx_test), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# Measure bias & generalization after resampling
bias_1 = measure_bias(train_loader, test_loader, color_fn, color_dim, args.epochs, args.lr)[0]
model = create_mnist_model(args.model)
gt_acc_1 = measure_generalization(train_loader, [test_loader, gt_test_loader], model, args.epochs, args.lr)[1]
print('Color bias after resampling = {:.3f}'.format(bias_1 + 0))
print('Generalization accuracy after resampling = {:.2%}'.format(gt_acc_1))

# Show examples
fig, ax = plt.subplots(2, 20, figsize=(12, 4))
for c in range(10):
    idx = cls_idx[c].nonzero().squeeze()
    rnd_idx = torch.randperm(len(idx))
    _, sort_idx = w[idx].sort(descending=True)
    for k in range(2):
        plt_id = c * 2 + k
        # random sample
        i = idx[rnd_idx[k]].item()
        # print(1 - repair_dataset[i][0].permute(1, 2, 0))
        ax[0][plt_id].imshow(1 - repair_dataset[i][0].permute(1, 2, 0))
        ax[0][plt_id].axis('off')
        # max weight
        i = idx[sort_idx[k]].item()
        ax[1][plt_id].imshow(1 - repair_dataset[i][0].permute(1, 2, 0))
        ax[1][plt_id].axis('off')
# plt.tight_layout()
plt.show()
