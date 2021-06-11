import os 
from nff.train import batch_to
from tqdm import tqdm 
import torch
import numpy as np
from ase import Atoms, io 

def create_dir(name):
    if not os.path.isdir(name):
        os.mkdir(name)   

def KL(mu, std):
     return -0.5 * torch.sum(1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2), dim=-1).mean()

def loop(loader, optimizer, device, model, beta, epoch, train=True, looptext=''):
    
    recon_loss = []
    kl_loss = []
    
    if train:
        model.train()
        mode = '{} train'.format(looptext)
    else:
        model.train() # yes, still set to train when reconstructing
        mode = '{} valid'.format(looptext)
    
    tqdm_data = tqdm(loader, position=0,
                     leave=True, desc='({} epoch #{})'.format(mode, epoch))
    
    for batch in tqdm_data:

        batch = batch_to(batch, device)
        S_mu, S_sigma, xyz, xyz_recon = model(batch)
        
        # loss
        loss_kl = KL(S_mu, S_sigma)
        loss_recon = (xyz_recon - xyz).pow(2).mean()
        loss = loss_kl * beta + loss_recon
        
        # optimize 
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging 
        recon_loss.append(loss_recon.item())
        kl_loss.append(loss_kl.item())
        
        mean_kl = np.array(kl_loss).mean()
        mean_recon = np.array(recon_loss).mean()
        
        postfix = ['avg. KL loss={:.4f}'.format(mean_kl) , 
                   'avg. recon loss={:.4f}'.format(mean_recon)]
        
        tqdm_data.set_postfix_str(' '.join(postfix))    
    
    
    return mean_kl, mean_recon, xyz, xyz_recon 

def get_all_true_reconstructed_structures(loader, device, model, atomic_nums, n_cg):

    model = model.to(device)

    true_xyzs = []
    recon_xyzs = []
    cg_xyzs = []
    mus = []
    sigmas = []

    tqdm_data = tqdm(loader, position=0, leave=True)    

    for batch in tqdm_data:
        batch = batch_to(batch, device)
        S_mu, S_sigma, xyz, xyz_recon = model(batch)

        true_xyzs.append(xyz.detach().cpu())
        recon_xyzs.append(xyz_recon.detach().cpu())
        cg_xyzs.append(batch['CG_nxyz'][:, 1:].detach().cpu())
        
        mus.append(S_mu.detach().cpu())
        sigmas.append(S_sigma.detach().cpu())

    true_xyzs = torch.cat(true_xyzs).reshape(-1, len(atomic_nums), 3).numpy()
    recon_xyzs = torch.cat(recon_xyzs).reshape(-1, len(atomic_nums), 3).numpy()
    cg_xyzs = torch.cat(cg_xyzs).reshape(-1, n_cg, 3).numpy()
    
    mu = torch.cat(mus).reshape(-1, n_cg, S_mu.shape[-1]).mean(0)
    sigma = torch.cat(sigmas).reshape(-1, n_cg, S_mu.shape[-1]).mean(0)
    
    return true_xyzs, recon_xyzs, cg_xyzs, mu, sigma

def dump_numpy2xyz(xyzs, atomic_nums, path):
    trajs = [Atoms(positions=xyz, numbers=atomic_nums.ravel()) for xyz in xyzs]
    io.write(path, trajs)