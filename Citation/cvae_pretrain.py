import sys
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from gcn.models import GCN
from cvae_models import VAE
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import copy

# Training settings
exc_path = sys.path[0]


def feature_tensor_normalize(feature):
    rowsum = torch.div(1.0, torch.sum(feature, dim=1))
    rowsum[torch.isinf(rowsum)] = 0.
    feature = torch.mm(torch.diag(rowsum), feature)
    return feature

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)

def generated_generator(args, device, adj, features, labels, features_normalized, adj_normalized, idx_train):

    x_list, c_list = [], [] 
    for i in trange(adj.shape[0]):
        x = features[adj[i].nonzero()[1]]
        c = np.tile(features[i], (x.shape[0], 1))
        x_list.append(x)
        c_list.append(c)
    features_x = np.vstack(x_list)
    features_c = np.vstack(c_list)
    del x_list
    del c_list
    gc.collect()
    

    features_x = torch.tensor(features_x, dtype=torch.float32)
    features_c = torch.tensor(features_c, dtype=torch.float32)

    cvae_features = torch.tensor(features, dtype=torch.float32)

    cvae_dataset = TensorDataset(features_x, features_c)
    cvae_dataset_sampler = RandomSampler(cvae_dataset)
    cvae_dataset_dataloader = DataLoader(cvae_dataset, sampler=cvae_dataset_sampler, batch_size=args.batch_size)
    

    hidden = 32
    dropout = 0.5
    lr = 0.01
    weight_decay = 5e-4
    epochs = 200
    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=labels.max().item() + 1,
                dropout=dropout)  

    model_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if args.cuda:
        model.to(device)
        features_normalized = features_normalized.to(device)
        adj_normalized = adj_normalized.to(device)
        cvae_features = cvae_features.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
    
    for _ in range(int(epochs / 2)):
        model.train()
        model_optimizer.zero_grad()
        output = model(features_normalized, adj_normalized)
        output = torch.log_softmax(output, dim=1)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        model_optimizer.step()
    
    # Pretrain
    cvae = VAE(encoder_layer_sizes=[features.shape[1], 256], 
               latent_size=args.latent_size, 
               decoder_layer_sizes=[256, features.shape[1]],
               conditional=args.conditional, 
               conditional_size=features.shape[1])
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=args.pretrain_lr)

    if args.cuda:
        cvae.to(device)

    # Pretrain
    t = 0
    best_augmented_features  = None
    cvae_model = None
    best_score = -float("inf")
    for _ in trange(args.pretrain_epochs, desc='Run CVAE Train'):
        for _, (x, c) in enumerate(tqdm(cvae_dataset_dataloader)):
            cvae.train()
            x, c = x.to(device), c.to(device)
            if args.conditional:
                recon_x, mean, log_var, _ = cvae(x, c)
            else:
                recon_x, mean, log_var, _ = cvae(x)
            cvae_loss = loss_fn(recon_x, x, mean, log_var)

            cvae_optimizer.zero_grad()
            cvae_loss.backward()
            cvae_optimizer.step()

            
            z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
            augmented_features = cvae.inference(z, cvae_features)
            augmented_features = feature_tensor_normalize(augmented_features).detach()
            
            total_logits = 0
            cross_entropy = 0
            for i in range(args.num_models):
                logits = model(augmented_features, adj_normalized)
                total_logits += F.softmax(logits, dim=1)
                output = F.log_softmax(logits, dim=1)
                cross_entropy += F.nll_loss(output[idx_train], labels[idx_train])
            output = torch.log(total_logits / args.num_models)
            U_score = F.nll_loss(output[idx_train], labels[idx_train]) - cross_entropy / args.num_models
            t += 1
            print("U Score: ", U_score, " Best Score: ", best_score)
            if U_score > best_score:
                best_score = U_score
                if t > args.warmup:
                    cvae_model = copy.deepcopy(cvae)
                    print("U_score: ", U_score, " t: ", t)
                    best_augmented_features = copy.deepcopy(augmented_features)
                    for i in range(args.update_epochs):
                        model.train()
                        model_optimizer.zero_grad()
                        output = model(best_augmented_features, adj_normalized)
                        output = torch.log_softmax(output, dim=1)
                        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                        loss_train.backward()
                        model_optimizer.step() 

            
    return best_augmented_features, cvae_model
