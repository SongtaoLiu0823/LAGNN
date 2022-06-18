from __future__ import division
from __future__ import print_function
import random
import argparse
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from tqdm import trange
from gcnii.model import *

import cvae_pretrain

# Training settings
exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=4)
parser.add_argument("--concat", type=int, default=4)
parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')
parser.add_argument('--consistency', action='store_true', default=False)

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dataset', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')

parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

# Adapted from GRAND: https://github.com/THUDM/GRAND
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

cvae_model = torch.load("{}/model/{}.pkl".format(exc_path, args.dataset))

def get_augmented_features(concat):
    X_list = []
    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, cvae_features)
        augmented_features = cvae_pretrain.feature_tensor_normalize(augmented_features).detach()
        if args.cuda:
            X_list.append(augmented_features.to(device))
        else:
            X_list.append(augmented_features)
    return X_list


if args.cuda:
    adj_normalized = adj_normalized.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    features_normalized = features_normalized.to(device)

all_val_acc = []
all_test_acc = []
for i in trange(args.runs, desc='Run Experiments'):
    model = LAGCNII(concat=args.concat+1,
                    nfeat=features.shape[1], 
                    nlayers=args.layer, 
                    nhidden=args.hidden, 
                    nclass=int(labels.max()) + 1, 
                    dropout=args.dropout, 
                    lamda=args.lamda, 
                    alpha=args.alpha, 
                    variant=args.variant).to(device)


    optimizer = optim.Adam([
                            {'params':model.params1,'weight_decay':args.wd1},
                            {'params':model.params2,'weight_decay':args.wd2},
                            ],lr=args.lr)
    if args.cuda:
        model.to(device)

    best = 999999999
    bad_counter = 0
    best_model = None
    best_X_list = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        output_list = []
        for k in range(args.samples):
            X_list = get_augmented_features(args.concat)
            output_list.append(torch.log_softmax(model(X_list+[features_normalized], adj_normalized), dim=-1))

        loss_train = 0.
        for k in range(len(output_list)):
            loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])
        
        loss_train = loss_train/len(output_list)
        if args.consistency:
            loss_consis = consis_loss(output_list)
            loss_train = loss_train + loss_consis

        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_X_list = get_augmented_features(args.concat)
            output = model(get_augmented_features(args.concat)+[features_normalized], adj_normalized)
            output = torch.log_softmax(output, dim=1)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
         
        if loss_val < best:
            best = loss_val
            best_model = copy.deepcopy(model)
            best_X_list = copy.deepcopy(val_X_list)
            bad_counter = 0
        else:
            bad_counter += 1
        
        if bad_counter == args.patience:
            break
        
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()))
        
    #Validate and Test
    best_model.eval()
    output = best_model(best_X_list+[features_normalized], adj_normalized)
    output = torch.log_softmax(output, dim=1)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    all_val_acc.append(acc_val.item())
    all_test_acc.append(acc_test.item())

print(np.mean(all_val_acc), np.std(all_val_acc), np.mean(all_test_acc), np.std(all_test_acc))
