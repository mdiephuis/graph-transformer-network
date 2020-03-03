import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from sklearn.model_selection import train_test_split

from external.data_loader import load_local_data
from external.torch_dataloader import GraphDataset
from models import Transformer

eps = np.finfo(float).eps

parser = argparse.ArgumentParser(description='GRAPH_ATTENTION')

parser.add_argument('--uid', type=str, default='GRAPHATT',
                    help='Staging identifier (default: GRAPHATT)')
parser.add_argument('--dataset-name', type=str, default='aids',
                    help='Name of dataset (default: aids')
parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of training epochs (default: 30)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input training batch-size')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

args = parser.parse_args()

# Set cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()

use_cuda = torch.cuda.is_available()

if use_cuda:
    device = torch.device("cuda")
    torch.cuda.set_device(1)
else:
    device = torch.device("cpu")

X, y = load_local_data(args.data_dir, args.dataset_name, attributes=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_dataset = GraphDataset(X_train, y_train)
test_dataset = GraphDataset(X_test, y_test)

train_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 6}
test_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 6}

train_loader = DataLoader(train_dataset, **train_params)
test_loader = DataLoader(train_dataset, **test_params)

# Settings
# embeddings are attributes
if args.dataset_name == 'aids':
    EMBED_DIM = 4
    num_classes = 2
    num_heads = 8
    depth = 6
    p, q = 1, 1
elif args.dataset_name == 'coildel':
    EMBED_DIM = 2
    num_classes = 100
    num_heads = 8
    depth = 6
    p, q = 1, 1

# k, num_heads, depth, seq_length, num_tokens, num_
model = Transformer(EMBED_DIM, num_heads, test_dataset.walklength, depth, num_classes).to(device)

lr_warmup = 10000

lr = 1e-3

opt = torch.optim.Adam(lr=lr, params=model.parameters())
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (lr_warmup / args.batch_size), 1.0))
loss_func = nn.NLLLoss()


def train_validate(model, loader, opt, loss_func, train, device):

    if train:
        model.train()
    else:
        model.eval()

    batch_loss = 0
    batch_acc = 0

    for graph in loader:
        x, y = graph
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)

        y_pred = model(x)

        loss = loss_func(y_pred, y)

        batch_loss += loss.item() / x.size(0)

        pred = y_pred.max(dim=1)[1]
        correct = pred.eq(y).sum().item()
        correct /= y.size(0)
        batch_acc += (correct * 100)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

    return batch_loss / len(loader), batch_acc / len(loader)


def execute_graph(model, train_loader, test_loader, opt, sch, loss_func, device):

    t_loss, t_acc = train_validate(model, train_loader, opt, loss_func, True, device)

    v_loss, v_acc = train_validate(model, test_loader, opt, loss_func, False, device)

    sch.step()

    return t_loss, v_loss, t_acc, v_acc


# Main epoch loop
train_loss = []
valid_loss = []

train_acc = []
valid_acc = []

for epoch in range(args.epochs):

    t_loss, v_loss, t_acc, v_acc = execute_graph(model, train_loader, test_loader, opt, sch, loss_func, device)

    train_loss.append(t_loss)
    train_acc.append(t_acc)

    valid_loss.append(v_loss)
    valid_acc.append(v_acc)

    print("Epoch: {} \t Train acc {} \t Valid acc {}".format(epoch, t_acc, v_acc))
