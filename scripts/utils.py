import os 
import sys
from tqdm import tqdm 
import torch
import numpy as np
from ase import Atoms, io 
import networkx as nx

def create_dir(name):
    if not os.path.isdir(name):
        os.mkdir(name)   

def check_CGgraph(dataset):
    frame_idx = np.random.randint(0, len(dataset), 20)

    for idx in frame_idx:
        a = dataset.props['CG_nbr_list'][idx]
        adj = [ tuple(pair.tolist()) for pair in a ]
        G = nx.Graph()
        G.add_edges_from(adj)
        connected = nx.is_connected(G)
        if not connected:
            print("One of the sampled CG graphs is not connected, training failed")
            return connected
        return True

# def KL(mu, std):
#      return -0.5 * torch.sum(1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2), dim=-1).mean()

def KL(mu1, std1, mu2, std2):
    if mu2 == None:
        return -0.5 * torch.sum(1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2), dim=-1).mean()
    else:
        return 0.5 * ( (std1.pow(2) / std2.pow(2)).sum(-1) + ((mu1 - mu2).pow(2) / std2).sum(-1) + \
            torch.log(std2.pow(2)).sum(-1) - torch.log(std1.pow(2)).sum(-1) - std1.shape[-1] ).mean()

def batch_to(batch, device):
    gpu_batch = dict()
    for key, val in batch.items():
        gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
    return gpu_batch

def loop(loader, optimizer, device, model, beta, epoch, train=True, looptext='', tqdm_flag=True):
    
    recon_loss = []
    kl_loss = []
    
    if train:
        model.train()
        mode = '{} train'.format(looptext)
    else:
        model.train() # yes, still set to train when reconstructing
        mode = '{} valid'.format(looptext)

    if tqdm_flag:
        loader = tqdm(loader, position=0, file=sys.stdout,
                         leave=True, desc='({} epoch #{})'.format(mode, epoch))
    
    for batch in loader:

        batch = batch_to(batch, device)
        S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon = model(batch)

        # loss
        loss_kl = KL(S_mu, S_sigma, H_prior_mu, H_prior_sigma) 

        loss_recon = (xyz_recon - xyz).pow(2).mean()
        loss = loss_kl * beta + loss_recon
        
        # optimize 
        if train:
            optimizer.zero_grad()
            loss.backward()

            # perfrom gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e0)

            optimizer.step()

        # logging 
        recon_loss.append(loss_recon.item())
        kl_loss.append(loss_kl.item())
        
        mean_kl = np.array(kl_loss).mean()
        mean_recon = np.array(recon_loss).mean()
        
        postfix = ['avg. KL loss={:.4f}'.format(mean_kl) , 
                   'avg. recon loss={:.4f}'.format(mean_recon)]
        
        if tqdm_flag:
            loader.set_postfix_str(' '.join(postfix))

    for result in postfix:
        print(result)
    
    return mean_kl, mean_recon, xyz, xyz_recon 

def get_all_true_reconstructed_structures(loader, device, model, atomic_nums, n_cg, atomwise_z=False, tqdm_flag=True):

    model = model.to(device)

    true_xyzs = []
    recon_xyzs = []
    cg_xyzs = []
    mus = []
    sigmas = []

    if atomwise_z == True:
        n_z = len(atomic_nums)
    else:
        n_z = n_cg 

    if tqdm_flag:
        loader = tqdm(loader, position=0, leave=True) 

    for batch in loader:
        batch = batch_to(batch, device)
        S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon = model(batch)

        true_xyzs.append(xyz.detach().cpu())
        recon_xyzs.append(xyz_recon.detach().cpu())
        cg_xyzs.append(batch['CG_nxyz'][:, 1:].detach().cpu())
        
        mus.append(S_mu.detach().cpu())
        sigmas.append(S_sigma.detach().cpu())

    true_xyzs = torch.cat(true_xyzs).reshape(-1, len(atomic_nums), 3).numpy()
    recon_xyzs = torch.cat(recon_xyzs).reshape(-1, len(atomic_nums), 3).numpy()
    cg_xyzs = torch.cat(cg_xyzs).reshape(-1, n_cg, 3).numpy()
    
    mu = torch.cat(mus).reshape(-1, n_z, S_mu.shape[-1]).mean(0)
    sigma = torch.cat(sigmas).reshape(-1, n_z, S_mu.shape[-1]).mean(0)
    
    return true_xyzs, recon_xyzs, cg_xyzs, mu, sigma

def dump_numpy2xyz(xyzs, atomic_nums, path):
    trajs = [Atoms(positions=xyz, numbers=atomic_nums.ravel()) for xyz in xyzs]
    io.write(path, trajs)