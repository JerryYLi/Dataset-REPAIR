import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return self.out(x)


class CNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_dim, kernel_size=3):
        super().__init__()
        layers = []
        pad_size = int(kernel_size / 2)
        for k, channels in enumerate(hidden_channels):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=kernel_size, padding=pad_size))
            layers.append(nn.ReLU())
            if k % 2 == 1 and k < len(hidden_channels) - 1:
                layers.append(nn.MaxPool2d(2))
            in_channels = channels
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(in_channels, out_dim)
    
    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


# adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
class LeNet(nn.Module):
    def __init__(self, in_channels, out_dims):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.out   = nn.Linear(84, out_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


def create_mnist_model(model_name):
    if model_name == 'lenet':
        return LeNet(3, 10).cuda()
    elif model_name == 'mlp':
        return MLP(784 * 3, [300, 100], 10).cuda()
    else:
        raise ValueError('Model not supported')