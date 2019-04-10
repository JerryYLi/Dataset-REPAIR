'''
Measure representation bias of dataset
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

def train(loader, epochs, model, optimizer, scheduler=None):
    model.train()
    with tqdm(range(1, epochs + 1)) as pbar:
        for _ in pbar:
            losses = []
            corrects = 0
            if scheduler is not None:
                scheduler.step()

            for x, y in loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                loss = F.cross_entropy(out, y)
                losses.append(loss.item())
                corrects += out.max(1)[1].eq(y).sum().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss = sum(losses) / len(losses)
            acc = 100 * corrects / len(loader.dataset)
            pbar.set_postfix(loss='%.3f' % loss, acc='%.2f%%' % acc)
    return loss, acc


def measure_bias(train_loader, test_loader, feat_fn, feat_dim, epochs=20, lr=1e-2, verbose=True):
    # class counts
    train_labels = torch.tensor([data[1] for data in train_loader.dataset]).long().cuda()
    n_cls = int(train_labels.max()) + 1

    # create models
    model = nn.Linear(feat_dim, n_cls).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 15, 0.1)

    # training
    model.train()
    pbar = tqdm(range(1, epochs + 1)) if verbose else range(1, epochs + 1)
    for _ in pbar:
        losses = []
        corrects = 0
        scheduler.step()

        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            # linear classifier
            out = model(feat_fn(x))
            loss = F.cross_entropy(out, y)
            losses.append(loss.item())
            corrects += out.max(1)[1].eq(y).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = sum(losses) / len(losses)
        acc = 100 * corrects / len(train_loader.dataset)
        if verbose:
            pbar.set_postfix(loss='%.3f' % loss, acc='%.2f%%' % acc)
    
    # testing
    model.eval()
    with torch.no_grad():
        losses = []
        corrects = 0

        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            x = x.view(x.size(0), x.size(1), -1)
            out = model(feat_fn(x))
            loss = F.cross_entropy(out, y)
            losses.append(loss.item())
            corrects += out.max(1)[1].eq(y).sum().item()

        loss = sum(losses) / len(losses)
        acc = 100 * corrects / len(test_loader.dataset)

        # measure bias
        cls_count = torch.stack([train_labels == c for c in range(n_cls)]).sum(1).float()
        cls_w = cls_count[cls_count > 0] / cls_count.sum()
        entropy = -(cls_w * cls_w.log()).sum().item()
        bias = 1 - loss / entropy
        
    return bias, loss, acc


def measure_generalization(train_loader, test_loaders, model, epochs=20, lr=1e-2):
    # create models
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.1)

    # training
    train(train_loader, epochs, model, optimizer, scheduler)
    
    # testing
    model.eval()
    test_losses = []
    test_accs = []
    with torch.no_grad():
        for k, loader in enumerate(test_loaders):
            losses = []
            corrects = 0

            for x, y in loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                loss = F.cross_entropy(out, y)
                losses.append(loss.item())
                corrects += out.max(1)[1].eq(y).sum().item()

            loss = sum(losses) / len(losses)
            test_losses.append(loss)
            acc = corrects / len(loader.dataset)
            test_accs.append(acc)
    
    return test_accs