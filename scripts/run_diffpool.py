import sys
sys.path.append("../scripts/")
sys.path.append("../src/")

import os 
import argparse 

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
from sklearn.model_selection import train_test_split

def plot_map(assign, z, save_path=None):
    mapping = assign.detach().cpu().numpy().transpose()
    z = z.tolist()
    z = [int(el) for el in z] 
    plt.imshow( mapping)
    plt.xticks(list(range(mapping.shape[1])), z)
    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()

def loop(loader, optimizer, device, model, tau, epoch, 
        gamma, kappa, train=True, looptext='', tqdm_flag=True):
    
    recon_loss = []
    adj_loss = []
    ent_loss = []
    
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
        batch = batch_to(batch, device=device)
        
        xyz, xyz_recon, assign, adj = model(batch, tau)
        
        # compute loss
        loss_recon = (xyz_recon - xyz).pow(2).mean()
        loss_entropy = -(assign * torch.log(assign)).sum(-1).mean()
        node_sim_mat = assign.matmul(assign.transpose(1,2))
        loss_adj = (node_sim_mat - adj).pow(2).mean()
        
        loss = loss_recon + gamma * loss_adj +  kappa * loss_entropy            

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recon_loss.append(loss_recon.item())
        adj_loss.append(loss_adj.item())
        ent_loss.append(loss_entropy.item())

        mean_recon = np.array(recon_loss).mean()
        mean_adj = np.array(adj_loss).mean()
        mean_ent = np.array(ent_loss).mean()
        
        del loss_adj, loss_entropy, loss_recon

        postfix = ['avg. recon loss={:.4f}'.format(mean_recon),
                   'avg. adj loss={:.4f}'.format(mean_adj),
                   'avg. entropy loss={:.4f}'.format(mean_ent),
                   'tau = {:.4f}'.format(tau)]

        loader.set_postfix_str(' '.join(postfix))
    
    
    return mean_recon, assign, xyz.detach().cpu(), xyz_recon.detach().cpu() 

def run(params):

    num_features = params['num_features']
    device = params['device']
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
    working_dir = params['logdir']
    tqdm_flag = params['tqdm_flag']

    create_dir(working_dir)
    
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

    # split train, test index 
    train_index, test_index = train_test_split(list(range(len(dataset))),test_size=0.2)

    trainset = get_subset_by_indices(train_index, dataset)
    testset = get_subset_by_indices(test_index, dataset)

#    import ipdb; ipdb.set_trace()

    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True)
    pooler = CGpool(nconv_pool, num_features, N_cg, assign_logits=None)
    
    encoder = DenseEquiEncoder(n_conv=dec_nconv, 
                           n_atom_basis=num_features,
                           n_rbf=n_rbf, 
                           activation=activation, 
                           cutoff=cutoff)
    
    decoder = DenseEquivariantDecoder(n_atom_basis=num_features,
                                      n_rbf=n_rbf, cutoff=cutoff, 
                                      num_conv=dec_nconv, activation=activation)
    
    model = DiffPoolVAE(encoder=encoder,decoder=decoder, pooler=pooler).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, 
                                                            factor=0.6, verbose=True, 
                                                            threshold=1e-4,  min_lr=1e-7)
    
    failed = False 

    # train     
    for epoch in range(n_epochs):
    
        mean_train_recon, assign, train_xyz, train_xyz_recon = loop(trainloader, optimizer, device, model, tau_sched[epoch], epoch, 
                                        gamma, kappa, train=True, looptext='', tqdm_flag=tqdm_flag)

        if np.isnan(mean_train_recon):
            print("NaN encoutered, exiting...")
            failed = True
            break 
            
        if epoch % 5 == 0:
            map_save_path = os.path.join(working_dir, 'map_{}.png'.format(epoch) )
            plot_map(assign[0], props['z'][0].numpy(), map_save_path)


        scheduler.step(mean_train_recon)

    dump_numpy2xyz(train_xyz_recon, props['z'][0].numpy(), os.path.join(working_dir, 'train_recon.xyz'))

    # test 
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True)
    model.eval()
    mean_test_recon, assign, test_xyz, test_xyz_recon = loop(testloader, optimizer, device, model, tau_sched[epoch], epoch, 
                                gamma, kappa, train=True, looptext='', tqdm_flag=tqdm_flag)

    # dump train recon 
    dump_numpy2xyz(test_xyz_recon, props['z'][0].numpy(), os.path.join(working_dir, 'test_recon.xyz'))


    print("train msd : {:.4f}".format(mean_train_recon))
    print("test msd : {:.4f}".format(mean_test_recon))

    return mean_test_recon, failed, assign

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-logdir', type=str)
    parser.add_argument('-device', type=int)
    parser.add_argument('-num_features',  type=int, default=512)
    parser.add_argument('-batch_size', type=int,default= 32)
    parser.add_argument('-N_cg', type=int, default= 3)
    parser.add_argument('-nconv_pool', type=int, default= 7)
    parser.add_argument('-enc_nconv', type=int, default= 4)
    parser.add_argument('-dec_nconv', type=int, default= 3)
    parser.add_argument('-cutoff', type=float, default= 8.0)
    parser.add_argument('-n_rbf', type=int,  default= 7)
    parser.add_argument('-activation', type=str,  default= 'ReLU')
    parser.add_argument('-n_epochs', type=int, default= 50)
    parser.add_argument('-tau_rate', type=float, default= 0.004 )
    parser.add_argument('-tau_0', type=float, default= 2.36)
    parser.add_argument('-gamma', type=float, default= 10.0)
    parser.add_argument('-kappa', type=float, default= 0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument("--tqdm_flag", action='store_true', default=False)

    params = vars(parser.parse_args())

    run(params)