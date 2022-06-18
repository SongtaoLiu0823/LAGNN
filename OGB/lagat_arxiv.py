import argparse
import math
import random
import time
import torch
import torch.nn as nn
import dgl
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        use_attn_dst=True,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if use_attn_dst:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("attn_r", None)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class LAGAT(nn.Module):
    def __init__(
        self,
        concat,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        n_heads,
        activation,
        dropout=0.0,
        input_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        use_attn_dst=True,
        use_symmetric_norm=False,
    ):
        super().__init__()
        self.concat = concat
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs_initial = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms_initial = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            if i == 0:
                for _ in range(self.concat):
                    in_hidden = in_feats
                    out_hidden = n_hidden 
                    num_heads = n_heads
                    out_channels = n_heads
                    self.convs_initial.append(
                        GATConv(
                            in_hidden,
                            out_hidden,
                            num_heads=num_heads,
                            attn_drop=attn_drop,
                            edge_drop=edge_drop,
                            use_attn_dst=use_attn_dst,
                            use_symmetric_norm=use_symmetric_norm,
                            residual=True,
                        )
                    )
                    if i < n_layers - 1:
                        self.norms_initial.append(nn.BatchNorm1d(out_channels * out_hidden))
            else:
                in_hidden = self.concat * n_heads * n_hidden if i > 0 else in_feats
                out_hidden = self.concat * n_hidden if i < n_layers - 1 else n_classes
                num_heads = n_heads if i < n_layers - 1 else 1
                out_channels = n_heads

                self.convs.append(
                    GATConv(
                        in_hidden,
                        out_hidden,
                        num_heads=num_heads,
                        attn_drop=attn_drop,
                        edge_drop=edge_drop,
                        use_attn_dst=use_attn_dst,
                        use_symmetric_norm=use_symmetric_norm,
                        residual=True,
                    )
                )

                if i < n_layers - 1:
                    self.norms.append(nn.BatchNorm1d(out_channels * out_hidden))

        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat_list):
        for i in range(self.n_layers):
            if i == 0:
                hidden_list = []
                for j, _ in enumerate(self.convs_initial):
                    h = feat_list[j]
                    h = self.input_drop(h)
                    conv = self.convs_initial[j](graph, h)
                    h = conv
                    if i < self.n_layers - 1:
                        h = h.flatten(1)
                        h = self.norms_initial[j](h)
                        h = self.activation(h, inplace=True)
                        h = self.dropout(h)
                    hidden_list.append(h)
                h = torch.cat((hidden_list), dim=-1)
            else:
                conv = self.convs[i-1](graph, h)

                h = conv

                if i < self.n_layers - 1:
                    h = h.flatten(1)
                    h = self.norms[i-1](h)
                    h = self.activation(h, inplace=True)
                    h = self.dropout(h)

        h = h.mean(1)
        h = self.bias_last(h)

        return h


epsilon = 1 - math.log(2)

device = None

dataset = "ogbn-arxiv"
n_node_feats, n_classes = 0, 0


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def load_data(dataset):
    global n_node_feats, n_classes

    data = DglNodePropPredDataset(name=dataset)
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph


def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    model = LAGAT(
        args.concat+1,
        n_node_feats_,
        n_classes,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        attn_drop=args.attn_drop,
        edge_drop=args.edge_drop,
        use_attn_dst=not args.no_attn_dst,
        use_symmetric_norm=args.use_norm,
    )

    return model


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def count_parameters(args):
    model = gen_model(args)
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def main():
    global device, n_node_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser(
        "GAT implementation on ogbn-arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument("--n-runs", type=int, default=10, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=2000, help="number of epochs")
    argparser.add_argument(
        "--use-labels", action="store_true", help="Use labels in the training set as input features."
    )
    argparser.add_argument("--n-label-iters", type=int, default=0, help="number of label iterations")
    argparser.add_argument("--mask-rate", type=float, default=0.5, help="mask rate")
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--use-norm", action="store_true", help="Use symmetrically normalized adjacency matrix.")
    argparser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument("--n-heads", type=int, default=3, help="number of heads")
    argparser.add_argument("--n-hidden", type=int, default=125, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.75, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.0, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention drop rate")
    argparser.add_argument("--edge-drop", type=float, default=0.0, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--log-every", type=int, default=20, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot-curves", action="store_true", help="plot learning curves")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")

    argparser.add_argument("--concat", type=int, default=1)
    argparser.add_argument("--samples", type=int, default=2)
    argparser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
    argparser.add_argument('--lam', type=float, default=1., help='Lamda')
    args = argparser.parse_args()

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

    if not args.use_labels and args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    graph = preprocess(graph)

    graph, labels, train_idx, val_idx, test_idx = map(
        lambda x: x.to(device), (graph, labels, train_idx, val_idx, test_idx)
    )

    cvae_model = torch.load("model/arxiv.pkl")
    def get_augmented_features(concat):
        X_list = []
        for _ in range(concat):
            z = torch.randn([graph.ndata["feat"].size(0), cvae_model.latent_size]).to(device)
            augmented_features = cvae_model.inference(z, graph.ndata["feat"].to(device)).detach()
            X_list.append(augmented_features)
        return X_list

    # run
    val_accs, test_accs = [], []

    for i in range(args.n_runs):
        seed(args.seed + i)
        evaluator_wrapper = lambda pred, labels: evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
        )["acc"]

        # define model and optimizer
        model = gen_model(args).to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

        # training loop
        total_time = 0
        best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")

        for epoch in range(1, args.n_epochs + 1):
            tic = time.time()

            adjust_learning_rate(optimizer, args.lr, epoch)

            model.train()

            feat = graph.ndata["feat"]

            loss = 0
            output_list = []
            Augmented_X_list = []
            for _ in range(args.samples):
                Augmented_X_list.append(get_augmented_features(args.concat)[0])
            
            if args.use_labels:
                mask = torch.rand(train_idx.shape) < args.mask_rate

                train_labels_idx = train_idx[mask]
                train_pred_idx = train_idx[~mask]

                feat = add_labels(feat, labels, train_labels_idx)
                for sample_ind in range(args.samples):
                    Augmented_X_list[sample_ind] = add_labels(Augmented_X_list[sample_ind], labels, train_labels_idx)
            else:
                mask = torch.rand(train_idx.shape) < args.mask_rate

                train_pred_idx = train_idx[mask]

            optimizer.zero_grad()
            for sample_ind in range(args.samples):
                pred = model(graph, [Augmented_X_list[sample_ind], feat])
                loss += custom_loss_function(pred[train_pred_idx], labels[train_pred_idx])
                output_list.append(pred.log_softmax(dim=-1))

            loss = loss / args.samples
            loss_consis = consis_loss(output_list)
            loss += loss_consis

            loss.backward()
            optimizer.step()
            
            acc, loss = evaluator_wrapper(pred[train_idx], labels[train_idx]), loss.item()

            val_Augmented_X = get_augmented_features(args.concat)[0]
            with torch.no_grad():
                model.eval()

                feat = graph.ndata["feat"]

                if args.use_labels:
                    feat = add_labels(feat, labels, train_idx)
                    val_Augmented_X = add_labels(val_Augmented_X, labels, train_idx)
                
                pred = model(graph, [val_Augmented_X, feat])
                
                train_loss = custom_loss_function(pred[train_idx], labels[train_idx])
                val_loss = custom_loss_function(pred[val_idx], labels[val_idx])
                test_loss = custom_loss_function(pred[test_idx], labels[test_idx])

                train_acc = evaluator_wrapper(pred[train_idx], labels[train_idx])
                val_acc = evaluator_wrapper(pred[val_idx], labels[val_idx])
                test_acc = evaluator_wrapper(pred[test_idx], labels[test_idx])

            toc = time.time()
            total_time += toc - tic

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                final_test_acc = test_acc
            
            if epoch == args.n_epochs or epoch % args.log_every == 0:
                print(
                    f"Run: {i + 1}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                    f"Loss: {loss:.4f}, Acc: {acc:.4f}\n"
                    f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
                )
            
        print("*" * 50)
        print(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
        print("*" * 50)

        val_accs.append(best_val_acc)
        test_accs.append(final_test_acc)

    print(args)
    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"Number of params: {count_parameters(args)}")


if __name__ == "__main__":
    main()
