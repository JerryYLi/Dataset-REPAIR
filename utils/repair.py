'''
REPAIR resampling of datasets minimizing representation bias
Returns a weight in [0, 1] for each example
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

def repair(loader, feat_fn, feat_dim, epochs, lr, lr_w):
    # class counts
    labels = torch.tensor([data[1] for data in loader.dataset]).long().cuda()
    n_cls = int(labels.max()) + 1
    cls_idx = torch.stack([labels == c for c in range(n_cls)]).float().cuda()

    # create models
    model = nn.Linear(feat_dim, n_cls).cuda()
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    weight_param = nn.Parameter(torch.zeros(len(loader.dataset)).cuda())
    optimizer_w = optim.SGD([weight_param], lr=lr_w)
    # optimizer_w = optim.SGD([weight_param], lr=lr_w, momentum=0.9)

    # training
    with tqdm(range(1, epochs + 1)) as pbar:
        for _ in pbar:
            losses = []
            corrects = 0
            for x, y, idx in loader:
                x, y = x.cuda(), y.cuda()

                # class probabilities
                w = torch.sigmoid(weight_param)
                z = w[idx] / w.mean()
                cls_w = cls_idx @ w
                q = cls_w / cls_w.sum()

                # linear classifier
                out = model(feat_fn(x))
                loss_vec = F.cross_entropy(out, y, reduction='none')
                loss = (loss_vec * z).mean()
                losses.append(loss.item())
                corrects += out.max(1)[1].eq(y).sum().item()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                # class weights
                optimizer_w.zero_grad()
                entropy = -(q[y].log() * z).mean()
                loss_w = 1 - loss / entropy
                loss_w.backward()
                optimizer_w.step()

            loss = sum(losses) / len(losses)
            acc = 100 * corrects / len(loader.dataset)
            pbar.set_postfix(loss='%.3f' % loss, acc='%.2f%%' % acc)

    # class probabilities & bias
    with torch.no_grad():
        w = torch.sigmoid(weight_param)
        cls_w = cls_idx @ w
        q = cls_w / cls_w.sum()
        rnd_loss = -(q * q.log()).sum().item()
        bias = 1 - loss / rnd_loss

    print('Accuracy = {:.2f}%, Loss = {:.3f}, Rnd Loss = {:.3f}, Bias = {:.3f}'.format(acc, loss, rnd_loss, bias))
    return w, q, cls_idx, cls_w, bias