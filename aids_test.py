import os
import sys
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


from external.data_loader import load_local_data
from external.torch_dataloader import GraphDataset
from models import Transformer

eps = np.finfo(float).eps


use_cuda = torch.cuda.is_available()
use_cuda = False

if use_cuda:
    device = torch.device("cuda")
    torch.cuda.set_device(1)
else:
    device = torch.device("cpu")


# Datasets
dataset_name = 'aids'
file_path = 'data/AIDS'

X, y = load_local_data(file_path, dataset_name, attributes=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_dataset = GraphDataset(X_train, y_train)
test_dataset = GraphDataset(X_test, y_test)

train_params = {'batch_size': 20, 'shuffle': True, 'num_workers': 6}
test_params = {'batch_size': 20, 'shuffle': True, 'num_workers': 6}

train_loader = DataLoader(train_dataset, **train_params)
test_loader = DataLoader(train_dataset, **test_params)

# Settings
# embeddings are attributes
EMBED_DIM = 4
num_classes = 2
num_heads = 8
depth = 6
p, q = 1, 1
num_epochs = 16

# k, num_heads, depth, seq_length, num_tokens, num_
model = Transformer(EMBED_DIM, num_heads, test_dataset.walklenght, depth, num_classes).to(device)

lr_warmup = 10000
batch_size = 16
lr = 1e-3

opt = torch.optim.Adam(lr=lr, params=model.parameters())
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (lr_warmup / batch_size), 1.0))
loss_func = nn.NLLLoss()

train_loss = []
valid_acc = []


def train(model, dataset, opt, sch, loss_func, device):
    model.train()
    batch_loss = 0
    for graph in dataset:
        x, y = graph
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)

        loss = loss_func(y_pred, y)

        batch_loss += loss.item() / x.size(0)

        opt.zero_grad()
        loss.backward()

        opt.step()
        sch.step()

    return batch_loss / len(dataset)


def validate(model, dataset, device):
    model.eval()
    batch_acc = 0

    for graph in dataset:
        x, y = graph
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)
        y_pred = model(x)

        pred = y_pred.max(dim=1)[1]
        correct = pred.eq(y).sum().item()
        correct /= y.size(0)
        batch_acc += (correct * 100)

    return batch_acc / len(dataset)

# Main epoch loop


for epoch in range(num_epochs):
    print("Epoch: {}".format(epoch))
    t_loss = train(model, train_loader, opt, sch, loss_func, device)
    train_loss.append(t_loss)
    print(t_loss)

    v_acc = validate(model, test_loader, opt, sch, loss_func, device)
    valid_acc.append(v_acc)
    print(v_acc)
