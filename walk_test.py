import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets, vocab

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from torch_cluster import random_walk

import numpy as np
import warnings
import re
import time
import os
import logging
import pandas as pd
import random
import sys
import math
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns

eps = np.finfo(float).eps


use_cuda = torch.cuda.is_available()
use_cuda = False

if use_cuda:
    device = torch.device("cuda")
    torch.cuda.set_device(1)
else:
    device = torch.device("cpu")


batch_size = 1
walk_length = 16

EMBED_DIM = 128
num_classes = 6
vocab_size = 128
num_heads = 8
depth = 6
p, q = 1, 1
num_epochs = 16


# dataset
path = '../data/ENZYMES/'
dataset = TUDataset(root=path, name='ENZYMES')
dataset = dataset.shuffle()
n = len(dataset) // 10

test_dataset = dataset[:n]
train_dataset = dataset[n:]
test_loader = DataLoader(test_dataset, batch_size=batch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size)


def nan_check_and_break(tensor, name=""):
    if isinstance(input, list) or isinstance(input, tuple):
        for tensor in input:
            return(nan_check_and_break(tensor, name))
    else:
        if nan_check(tensor, name) is True:
            exit(-1)


def nan_check(tensor, name=""):
    if isinstance(input, list) or isinstance(input, tuple):
        for tensor in input:
            return(nan_check(tensor, name))
    else:
        if torch.sum(torch.isnan(tensor)) > 0:
            print("Tensor {} with shape {} was NaN.".format(name, tensor.shape))
            return True

        elif torch.sum(torch.isinf(tensor)) > 0:
            print("Tensor {} with shape {} was Inf.".format(name, tensor.shape))
            return True

    return False


def zero_check_and_break(tensor, name=""):
    if torch.sum(tensor == 0).item() > 0:
        print("tensor {} of {} dim contained ZERO!!".format(name, tensor.shape))
        exit(-1)


lr = 1e-3


def train(dataset, device):
    for graph in dataset:
        x, edge_index, y = graph.x, graph.edge_index, graph.y
        edge_index = edge_index.to(device)
        x = x.to(device)
        y = y.to(device)
        subset = torch.arange(x.size(0), device=edge_index.device)
        # subset = subset[:np.min((50, x.size(0)))]
        # if x.size(0) >= 100:
        #     print('over 100')
        print(x.dtype)
        print(edge_index[0].dtype)
        print(edge_index[1].dtype)
        walks = random_walk(edge_index[0], edge_index[1], subset, walk_length, p, q, x.size(0))
        if (walks.size(0) == 0):
            print('zero sized walks')

        nan_check_and_break(walks, "walks nan check")

    return


# Main epoch loop
num_epochs = 15

for epoch in range(num_epochs):
    print("Epoch: {}".format(epoch))
    train(train_dataset, device)
