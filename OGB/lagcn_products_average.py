import argparse
import numpy as np
import random

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


parser = argparse.ArgumentParser(description='OGBN-Products (GNN)')

parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument("--pretrain_lr", type=float, default=1e-5)
parser.add_argument("--total_iterations", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--concat", type=int, default=1)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--runs', type=int, default=10)

args = parser.parse_args()
print(args)


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygNodePropPredDataset(name='ogbn-products', transform=T.ToSparseTensor())
data = dataset[0]
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)

# Pre-compute GCN normalization.
adj_t = data.adj_t.set_diag()
deg = adj_t.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
data.adj_t = adj_t

data = data.to(device)

cvae_model = torch.load("model/products.pkl")
def get_augmented_features(concat):
    X_list = []
    for _ in range(concat):
        z = torch.randn([data.x.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, data.x).detach()
        X_list.append(augmented_features)
    return X_list

model = GCN(data.num_features, args.hidden_channels,
            dataset.num_classes, args.num_layers,
            args.dropout).to(device)

evaluator = Evaluator(name='ogbn-products')
logger = Logger(args.runs, args)

for run in range(args.runs):
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        model.train()
        optimizer.zero_grad()

        output_list = []
        Augmented_X = get_augmented_features(args.concat)[0]
        output_list.append(torch.log_softmax(model((Augmented_X+data.x)/2, data.adj_t), dim=-1))

        loss_train = 0.
        for k in range(len(output_list)):
            loss_train += F.nll_loss(output_list[k][train_idx], data.y.squeeze(1)[train_idx])
        
        loss_train = loss_train/len(output_list)
        
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_Augmented_X = get_augmented_features(args.concat)[0]
            output = torch.log_softmax(model((val_Augmented_X+data.x)/2, data.adj_t), dim=-1)
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
