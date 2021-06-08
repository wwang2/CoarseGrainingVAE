
from nff.train import batch_to
from tqdm import tqdm 
import torch
import numpy as np

def KL(mu, std):
     return -0.5 * torch.sum(1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2), dim=-1).mean()

def loop(loader, optimizer, device, model, beta, epoch, train=True):
    
    recon_loss = []
    kl_loss = []
    
    if train:
        model.train()
        mode = 'train'
    else:
        model.train() # yes, still set to train when reconstructing
        mode = 'val'
    
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
        
        postfix = ['avg. KL loss={:.3f}'.format(mean_kl) , 
                   'avg. recon loss={:.3f}'.format(mean_recon)]
        
        tqdm_data.set_postfix_str(' '.join(postfix))    
    
    
    return mean_kl, mean_recon, xyz, xyz_recon 