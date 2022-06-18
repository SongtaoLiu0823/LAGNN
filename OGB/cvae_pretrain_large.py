import sys
import gc
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
from cvae_models import VAE

# Training settings
exc_path = sys.path[0]


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)

def generated_generator(args, device, adj_scipy, features):
    features = features.to(torch.device('cpu')) 
    x_list, c_list = [], [] 
    for i in trange(adj_scipy.shape[0]):
        neighbors_index = list(adj_scipy[i].nonzero()[1])
        x = features[neighbors_index]
        c = np.tile(features[i], (x.shape[0], 1))
        x_list.append(x)
        c_list.append(c)
    features_x = np.vstack(x_list)
    features_c = np.vstack(c_list)
    del x_list
    del c_list
    gc.collect()

    # Pretrain
    cvae = VAE(encoder_layer_sizes=[features.shape[1], 256], 
               latent_size=args.latent_size, 
               decoder_layer_sizes=[256, features.shape[1]],
               conditional=args.conditional, 
               conditional_size=features.shape[1]).to(device)
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=args.pretrain_lr)

    for epoch in trange(args.total_iterations, desc='Run CVAE Train'):
        index = random.sample(range(features_c.shape[0]), args.batch_size)
        x, c = features_x[index], features_c[index]
        x = torch.tensor(x, dtype=torch.float32)
        c = torch.tensor(c, dtype=torch.float32)
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
    
    del(features_x)
    del(features_c)
    gc.collect()
    return cvae
