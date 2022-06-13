from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import sys
import random
import scipy.sparse as sp
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import cvae_pretrain

from tqdm import trange
from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
from grand.models import MLP

exc_path = sys.path[0]

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--input_droprate', type=float, default=0.5, help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_droprate', type=float, default=0.5, help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--dropnode_rate', type=float, default=0.5, help='Dropnode rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--order', type=int, default=5, help='Propagation step')
parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')
parser.add_argument('--dataset', type=str, default='cora', help='Data set')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

# Load data
adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)

# Normalize adj and features
features = features.toarray()
adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])) 
features_normalized = normalize_features(features)

# To PyTorch Tensor
labels = torch.LongTensor(labels)
labels = torch.max(labels, dim=1)[1]
features_normalized = torch.FloatTensor(features_normalized)
adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

Augmented_X = cvae_pretrain.feature_tensor_normalize(torch.load("{}/feature/{}_features.pt".format(exc_path, args.dataset)))


def propagate(feature, A, order):
    #feature = F.dropout(feature, args.dropout, training=training)
    x = feature
    y = feature
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        #print(y.add_(x))
        y.add_(x)
        
    return y.div_(order+1.0).detach_()


def rand_prop(features, A, training):
    n = features.shape[0]
    drop_rate = args.dropnode_rate
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
    
    if training:
            
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        if args.cuda:
            masks = masks.cuda()

        #features = masks.cuda() * features
        features = masks * features
            
    else:
            
        features = features * (1. - drop_rate)
    features = propagate(features, A, args.order)    
    return features


def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.lam * loss


if args.cuda:
    features_normalized = features_normalized.to(device)
    adj_normalized = adj_normalized.to(device)
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

all_val_acc = []
all_test_acc = []
for i in trange(args.runs, desc='Run Experiments'):
    # Model and optimizer
    model = MLP(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                input_droprate=args.input_droprate,
                hidden_droprate=args.hidden_droprate,
                use_bn = args.use_bn)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()


    loss_values = []
    acc_values = []
    bad_counter = 0
    loss_best = np.inf
    acc_best = 0.0

    loss_mn = np.inf
    acc_mx = 0.0

    best_epoch = 0
    best_model = None
    for epoch in range(args.epochs):
        X = features_normalized
        A = adj_normalized

        model.train()
        optimizer.zero_grad()
        X_list = []
        K = args.sample
        for k in range(K):
            X_list.append(rand_prop(X, A, training=True))
            X_list.append(rand_prop(Augmented_X, A, training=True))

        output_list = []
        for k in range(len(X_list)):
            output_list.append(torch.log_softmax(model(X_list[k]), dim=-1))

        loss_train = 0.
        for k in range(len(X_list)):
            loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])
        
        loss_train = loss_train/len(X_list)
        loss_consis = consis_loss(output_list)

        loss_train = loss_train + loss_consis
        acc_train = accuracy(output_list[0][idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        X = rand_prop(X, A, training=False)
        output = model(X)
        output = torch.log_softmax(output, dim=-1)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val]) 
        acc_val = accuracy(output[idx_val], labels[idx_val])

        loss_values.append(loss_val.item())
        acc_values.append(acc_val.item())

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()))
         
        
        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:
            if loss_values[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                best_X = copy.deepcopy(X)
                best_model = copy.deepcopy(model)

            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
        
    best_model.eval()
    X = rand_prop(features_normalized, A, training=False)
    output = best_model(X)
    output = torch.log_softmax(output, dim=-1)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    all_val_acc.append(acc_val.item())
    all_test_acc.append(acc_test.item())

print(np.mean(all_val_acc), np.std(all_val_acc), np.mean(all_test_acc), np.std(all_test_acc))
