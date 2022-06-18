import argparse
import numpy as np
import random
import scipy.sparse as sp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


class LASAGE(torch.nn.Module):
    def __init__(self, concat, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LASAGE, self).__init__()

        self.convs_initial = torch.nn.ModuleList()
        for _ in range(concat):
            self.convs_initial.append(SAGEConv(in_channels, hidden_channels))

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(concat*hidden_channels, concat*hidden_channels))
        self.convs.append(SAGEConv(concat*hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for conv in self.convs_initial:
            conv.reset_parameters()

    def forward(self, x_list, adj_t):
        hidden_list = []
        for i, conv in enumerate(self.convs_initial):
            x = conv(x_list[i], adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            hidden_list.append(x)
        x = torch.cat((hidden_list), dim=-1)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


parser = argparse.ArgumentParser(description='OGBN-Proteins (GNN)')

parser.add_argument("--concat", type=int, default=1)
parser.add_argument("--samples", type=int, default=2)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--runs', type=int, default=10)

args = parser.parse_args()
print(args)


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygNodePropPredDataset(name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'))
data = dataset[0]
# Move edge features to node features.
data.x = data.adj_t.mean(dim=1)
data.adj_t.set_value_(None)

split_idx = dataset.get_idx_split()
train_idx = split_idx["train"].to(device)

adj_scipy = sp.csr_matrix(data.adj_t.to_scipy())
data = data.to(device)

cvae_model = torch.load("model/proteins.pkl")
def get_augmented_features(concat):
    X_list = []
    for _ in range(concat):
        z = torch.randn([data.x.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, data.x).detach()
        X_list.append(augmented_features)
    return X_list

model = LASAGE(args.concat+1, data.num_features, args.hidden_channels,
               112, args.num_layers,
               args.dropout).to(device)

evaluator = Evaluator(name='ogbn-proteins')
logger = Logger(args.runs, args)

criterion = torch.nn.BCEWithLogitsLoss()
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        model.train()
        optimizer.zero_grad()
        
        output_list = []
        for k in range(args.samples):
            X_list = get_augmented_features(args.concat)
            output_list.append(model(X_list+[data.x], data.adj_t))
        
        loss_train = 0.
        for k in range(len(output_list)):
            loss_train += criterion(output_list[k][train_idx], data.y[train_idx].to(torch.float))
        
        loss_train = loss_train/len(output_list)
        
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_X_list = get_augmented_features(args.concat)
            y_pred = model(val_X_list+[data.x], data.adj_t)
            
            train_rocauc = evaluator.eval({
                'y_true': data.y[split_idx['train']],
                'y_pred': y_pred[split_idx['train']],
            })['rocauc']
            valid_rocauc = evaluator.eval({
                'y_true': data.y[split_idx['valid']],
                'y_pred': y_pred[split_idx['valid']],
                })['rocauc']
            test_rocauc = evaluator.eval({
                'y_true': data.y[split_idx['test']],
                'y_pred': y_pred[split_idx['test']],
            })['rocauc']

            logger.add_result(run, (train_rocauc, valid_rocauc, test_rocauc))
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss_train.item():.4f}, '
                  f'Train: {100 * train_rocauc:.2f}%, '
                  f'Valid: {100 * valid_rocauc:.2f}% '
                  f'Test: {100 * test_rocauc:.2f}%')

    logger.print_statistics(run)
logger.print_statistics()
