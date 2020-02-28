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


