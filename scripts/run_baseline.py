import sys
sys.path.append("../scripts/")
sys.path.append("../src/")

import os 
import argparse 

from data import *
from baseline import * 
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

def loop(loader, optimizer, device, model, epoch, train=True, looptext='', tqdm_flag=True):
    
    recon_loss = []
    
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
        
        assign, xyz, xyz_recon = model(batch)
        
        # compute loss
        loss_recon = (xyz_recon - xyz).pow(2).mean()
        loss = loss_recon 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recon_loss.append(loss_recon.item())
        mean_recon = np.array(recon_loss).mean()
        
        del loss_recon

        postfix = ['avg. recon loss={:.4f}'.format(mean_recon)]
        if tqdm_flag:
            loader.set_postfix_str(' '.join(postfix))
        else:
            for result in postfix:
                print(result)
    
    return mean_recon, assign, xyz.detach().cpu(), xyz_recon.detach().cpu() 

def run(params):

    device = params['device']
    batch_size = params['batch_size']
    N_cg = params['N_cg']
    n_epochs = params['n_epochs']
    cg_method = params['cg_method']
    working_dir = params['logdir']
    lr = params['lr']
    tqdm_flag = params['tqdm_flag']

    create_dir(working_dir)

    label = 'dipeptide'
    traj_files = glob.glob(PROTEINFILES[label]['traj_paths'])[:200]
    pdb_file = PROTEINFILES[label]['pdb_path']
    file_type = PROTEINFILES[label]['file_type']

    trajs = [md.load_xtc(file,
                top=pdb_file) for file in traj_files]

    atomic_nums, protein_index = get_atomNum(trajs[0])
    n_atoms = len(atomic_nums)

    # get cg_map 
    if cg_method == 'newman':
        protein_top = trajs[0].top.subset(protein_index)
        g = protein_top.to_bondgraph()
        paritions = get_partition(g, N_cg)
        mapping = parition2mapping(paritions, n_atoms)
        assign_idx = torch.LongTensor( np.array(mapping) ) 

    props = get_diffpool_data(N_cg, trajs, frame_skip=500)

    dataset = DiffPoolDataset(props)
    dataset.generate_neighbor_list(5.0)

    # split train, test index 
    train_index, test_index = train_test_split(list(range(len(dataset))),test_size=0.2)

    trainset = get_subset_by_indices(train_index, dataset)
    testset = get_subset_by_indices(test_index, dataset)

    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True)
    pooler = CGpool(1, 16, n_atoms=n_atoms, n_cgs=N_cg, assign_idx=assign_idx)
    
    model = Baseline(pooler=pooler, n_cgs=N_cg, n_atoms=n_atoms).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, 
                                                            factor=0.6, verbose=True, 
                                                            threshold=1e-4,  min_lr=1e-7)
    
    failed = False 

    # train     
    for epoch in range(n_epochs):
    
        mean_train_recon, assign, train_xyz, train_xyz_recon = loop(trainloader, optimizer, device, model,  epoch, train=True, looptext='', tqdm_flag=tqdm_flag)

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
    mean_test_recon, assign, test_xyz, test_xyz_recon = loop(testloader, optimizer, device, model,  epoch, train=True, looptext='', tqdm_flag=tqdm_flag)

    # dump train recon 
    dump_numpy2xyz(test_xyz_recon, props['z'][0].numpy(), os.path.join(working_dir, 'test_recon.xyz'))

    print("train msd : {:.4f}".format(mean_train_recon))
    print("test msd : {:.4f}".format(mean_test_recon))

    return mean_test_recon, failed, assign

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-logdir', type=str)
    parser.add_argument('-device', type=int)
    parser.add_argument('-batch_size', type=int,default= 32)
    parser.add_argument('-N_cg', type=int, default= 3)
    parser.add_argument('-n_epochs', type=int, default= 50)
    parser.add_argument("-cg_method", type=str, default='newman')
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument("--tqdm_flag", action='store_true', default=False)

    params = vars(parser.parse_args())

    run(params)