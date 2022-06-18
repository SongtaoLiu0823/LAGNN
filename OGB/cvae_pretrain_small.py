import sys
import gc
import torch
import torch.optim as optim
from tqdm import tqdm, trange
from cvae_models import VAE
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# Training settings
exc_path = sys.path[0]


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)

def generated_generator(args, device, adj_scipy, features):
    
    x_list, c_list = [], [] 
    for i in trange(adj_scipy.shape[0]):
        neighbors_index = list(adj_scipy[i].nonzero()[1])
        x = features[neighbors_index]
        c = torch.tile(features[i], (x.shape[0], 1))
        x_list.append(x)
        c_list.append(c)
    features_x = torch.vstack(x_list)
    features_c = torch.vstack(c_list)
    del x_list
    del c_list
    gc.collect()

    cvae_dataset = TensorDataset(features_x, features_c)
    cvae_dataset_sampler = RandomSampler(cvae_dataset)
    cvae_dataset_dataloader = DataLoader(cvae_dataset, sampler=cvae_dataset_sampler, batch_size=args.batch_size)
    
    # Pretrain
    cvae = VAE(encoder_layer_sizes=[features.shape[1], 256], 
               latent_size=args.latent_size, 
               decoder_layer_sizes=[256, features.shape[1]],
               conditional=args.conditional, 
               conditional_size=features.shape[1]).to(device)
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=args.pretrain_lr)

    # Pretrain
    t = 0
    for _ in trange(100, desc='Run CVAE Train'):
        for _, (x, c) in enumerate(tqdm(cvae_dataset_dataloader)):
            t += 1
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
            if t >= args.total_iterations:
                return cvae
