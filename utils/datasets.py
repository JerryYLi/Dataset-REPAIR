import torch
from torch.utils.data import Dataset

from tqdm import tqdm
import colorsys


class SubClassDataset(Dataset):
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.classes = classes
        print('Subsampling dataset...')
        self.indices = [i for i, (_, y) in enumerate(tqdm(dataset)) if y in classes]
        print('Done, {}/{} examples left'.format(len(self.indices), len(self.dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        return x, self.classes.index(y)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (*self.dataset[idx], idx)


class ColoredDataset(Dataset):
    def __init__(self, dataset, classes=None, colors=[0, 1], std=0):
        self.dataset = dataset
        self.colors = colors
        if classes is None:
            classes = max([y for _, y in dataset]) + 1

        if isinstance(colors, torch.Tensor):
            self.colors = colors
        elif isinstance(colors, list):
            self.colors = torch.Tensor(classes, 3, 1, 1).uniform_(colors[0], colors[1])
        else:
            raise ValueError('Unsupported colors!')
        self.perturb = std * torch.randn(len(self.dataset), 3, 1, 1)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        color_img = (self.colors[label] + self.perturb[idx]).clamp(0, 1) * img
        return color_img, label