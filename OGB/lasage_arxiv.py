import argparse
import numpy as np
import random
import scipy.sparse as sp
import sys 

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
exc_path = sys.path[0]


class LASAGE(torch.nn.Module):
    def __init__(self, concat, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LASAGE, self).__init__()

        self.convs_initial = torch.nn.ModuleList()
        self.bns_initial = torch.nn.ModuleList()    
        for _ in range(concat):
            self.convs_initial.append(SAGEConv(in_channels, hidden_channels))
            self.bns_initial.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(concat*hidden_channels, concat*hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(concat*hidden_channels))
        self.convs.append(SAGEConv(concat*hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for conv in self.convs_initial:
            conv.reset_parameters()
        for bn in self.bns_initial:
            bn.reset_parameters()

    def forward(self, x_list, adj_t):
        hidden_list = []
        for i, conv in enumerate(self.convs_initial):
            x = conv(x_list[i], adj_t)
            x = self.bns_initial[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            hidden_list.append(x)
        x = torch.cat((hidden_list), dim=-1)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')

parser.add_argument("--concat", type=int, default=1)
parser.add_argument("--samples", type=int, default=4)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--runs', type=int, default=10)

parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')

args = parser.parse_args()
print(args)


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


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
adj_scipy = sp.csr_matrix(data.adj_t.to_scipy())
data = data.to(device)

split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)

cvae_model = torch.load("model/arxiv.pkl")
def get_augmented_features(concat):
    X_list = []
    for _ in range(concat):
        z = torch.randn([data.x.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, data.x).detach()
        X_list.append(augmented_features.to(device))
    return X_list

model = LASAGE(args.concat+1, data.num_features, args.hidden_channels,
               dataset.num_classes, args.num_layers,
               args.dropout).to(device)

evaluator = Evaluator(name='ogbn-arxiv')
logger = Logger(args.runs, args)

for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        model.train()
        optimizer.zero_grad()

        output_list = []
        for k in range(args.samples):
            X_list = get_augmented_features(args.concat)
            output_list.append(model(X_list+[data.x], data.adj_t).log_softmax(dim=-1))

        loss_train = 0.
        for k in range(len(output_list)):
            loss_train += F.nll_loss(output_list[k][train_idx], data.y.squeeze(1)[train_idx])

        loss_train = loss_train/len(output_list)
        loss_consis = consis_loss(output_list)
        loss_train = loss_train + loss_consis

        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_X_list = get_augmented_features(args.concat)
            output = model(val_X_list+[data.x], data.adj_t)
            y_pred = output.argmax(dim=-1, keepdim=True)

            train_acc = evaluator.eval({
                'y_true': data.y[split_idx['train']],
                'y_pred': y_pred[split_idx['train']],
            })['acc']
            valid_acc = evaluator.eval({
                'y_true': data.y[split_idx['valid']],
                'y_pred': y_pred[split_idx['valid']],
            })['acc']
            test_acc = evaluator.eval({
                'y_true': data.y[split_idx['test']],
                'y_pred': y_pred[split_idx['test']],
            })['acc']

            logger.add_result(run, (train_acc, valid_acc, test_acc))

            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss_train.item():.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')

    logger.print_statistics(run)
logger.print_statistics()
