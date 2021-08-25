import sys
sys.path.append("../scripts/")
sys.path.append("../src/")

from data import *
from diffpoolvae import * 
from conv import * 
from datasets import * 
from utils import * 
from visualization import xyz_grid_view, rotate_grid
from sampling import * 
import torch
from torch import nn
from torch.nn import Sequential 
import numpy as np
import copy
from torch_scatter import scatter_mean
from tqdm import tqdm 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

from tqdm import tqdm 


def run(params, device):

    num_features = params['num_features']
    nconv_pool = params['nconv_pool']
    batch_size = params['batch_size']
    N_cg = params['N_cg']
    enc_nconv = params['enc_nconv']
    dec_nconv = params['dec_nconv']
    cutoff = params['cutoff']
    n_rbf = params['n_rbf']
    activation = params['activation']
    tau_0 = params['tau_0']
    tau_rate = params['tau_rate']
    n_epochs = params['n_epochs']
    gamma = params['gamma']
    kappa = params['kappa']
    lr = params['lr']
    
    tau_sched = tau_0 * np.exp(-tau_rate * torch.linspace(0, n_epochs-1, n_epochs))

    label = 'dipeptide'
    traj_files = glob.glob(PROTEINFILES[label]['traj_paths'])[:200]
    pdb_file = PROTEINFILES[label]['pdb_path']
    file_type = PROTEINFILES[label]['file_type']

    trajs = [md.load_xtc(file,
                top=pdb_file) for file in traj_files]
    props = get_diffpool_data(N_cg, trajs, frame_skip=500)

    dataset = DiffPoolDataset(props)
    dataset.generate_neighbor_list(cutoff)
    
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True)
    cgpool_model = CGpool(nconv_pool, num_features, N_cg, assign_logits=None).to(device)
    
    enc = DenseEquiEncoder(n_conv=dec_nconv, 
                           n_atom_basis=num_features,
                           n_rbf=n_rbf, 
                           activation='swish', 
                           cutoff=cutoff).to(device)
    
    decoder = DenseEquivariantDecoder(n_atom_basis=num_features,
                                      n_rbf=n_rbf, cutoff=cutoff, 
                                      num_conv=dec_nconv, activation=activation).to(device)
    
    
    optimizer = torch.optim.Adam(list(cgpool_model.parameters()) + list(enc.parameters()) + list(decoder.parameters()),
                             lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, 
                                                            factor=0.6, verbose=True, 
                                                            threshold=1e-4,  min_lr=1e-7)
    
    failed = False 

    for epoch in range(n_epochs):
        batch_loss = []
        recon_batch_loss = []
        adj_batch_loss = [] 

        loader = tqdm(loader, position=0, leave=True) 
        for batch in loader:     
            batch = batch_to(batch, device=device)
            xyz = batch['xyz']        
            z = torch.ones_like( batch['z'] ) 
            nbr_list = batch['nbr_list']

            assign, assign_logits, h, H, cg_xyz, soft_cg_adj = cgpool_model(z, 
                                                                       batch['xyz'], 
                                                                       batch['bonds'], 
                                                                       tau=tau_sched[epoch],
                                                                       gumbel=True)



            cg_adj = (soft_cg_adj > 0.01).to(torch.float).to(device)
            assign_entropy = -(assign * torch.log(assign)).sum(-1).mean()

            adj_loss = soft_cg_adj.pow(2).mean()

            H, V = enc(h, H, xyz, cg_xyz, assign, nbr_list, cg_adj)
            H, V = decoder(H, cg_adj, cg_xyz)

            dx = torch.einsum('bcfe,bac->bcfe', V[:, :, :z.shape[1], :], assign).sum(1)

            x_recon = torch.einsum('bce,bac->bae', cg_xyz, assign) + dx
            loss_recon = (x_recon - xyz).pow(2).mean() 
            loss = loss_recon + gamma * adj_loss +  kappa * assign_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss_recon.item())
            adj_batch_loss.append(adj_loss.item())

            recon_batch_loss.append(loss_recon.item())

            mean_recon = np.array(recon_batch_loss).mean()

            postfix = ['avg. recon loss={:.4f}'.format(mean_recon),
                       'avg. adj loss={:.4f}'.format(adj_loss),
                       'tau = {:.4f}'.format(tau_sched[epoch])]

            loader.set_postfix_str(' '.join(postfix))

    #     if epoch % 5 == 0:
    #         plot_map(assign[0], batch['z'][0])

            if np.isnan(mean_recon):
                print("NaN encoutered, exiting...")
                failed = True
                break 

        scheduler.step(mean_recon)

    return mean_recon, failed




