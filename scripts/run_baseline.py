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
from geometry import * 
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
from sklearn.model_selection import KFold
import re 
import pandas as pd

def plot_map(assign, z, save_path=None):
    mapping = assign.detach().cpu().numpy().transpose()
    z = z.tolist()
    z = [int(el) for el in z] 
    plt.imshow( mapping)
    plt.xticks(list(range(mapping.shape[1])), z)
    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()

def retrieve_recon_structures(loader, device, model, tqdm_flag=True):
    # get all structure
    # compute rmsd 
    # compute graph metric 

    all_xyz_data = []
    all_xyz_recon = []

    heavy_ged = []
    all_ged = []

    all_valid_ratios = []
    heavy_valid_ratios = []

    model.eval()

    if tqdm_flag:
        loader = tqdm(loader, position=0, file=sys.stdout,
                         leave=True, desc='eval')

    for batch in loader:     
        batch = batch_to(batch, device=device)
        assign, xyz, xyz_recon = model(batch)

        all_xyz_data.append(xyz.detach().cpu().numpy())
        all_xyz_recon.append(xyz_recon.detach().cpu().numpy())

        for i, x in enumerate(xyz):
            z = batch['z'][0].detach().cpu().numpy()
            ref_atoms = Atoms(numbers=z, positions=x.detach().cpu().numpy())
            recon_atoms = Atoms(numbers=z, positions=xyz_recon[i].detach().cpu().numpy())

            # compute ged diff 
            all_rmsds, heavy_rmsds, valid_ratio, valid_hh_ratio, graph_val_ratio, graph_hh_val_ratio = eval_sample_qualities(ref_atoms, [recon_atoms])

            heavy_ged.append(graph_val_ratio)
            all_ged.append(graph_hh_val_ratio)

            all_valid_ratios.append(valid_hh_ratio)
            heavy_valid_ratios.append(valid_ratio)

    all_valid_ratio = np.array(all_valid_ratios).mean()
    heavy_valid_ratio = np.array(heavy_valid_ratios).mean()

    return np.concatenate(all_xyz_data), np.concatenate(all_xyz_recon), np.array(heavy_ged).mean(), np.array(all_ged).mean(), heavy_valid_ratio, all_valid_ratio

def dist_loss(xyz, xyz_recon, nbr_list):

    gen_dist = (xyz_recon[nbr_list[:, 0 ], nbr_list[:, 1]] - xyz_recon[nbr_list[:, 0], nbr_list[:, 2]]).pow(2).sum(-1).sqrt()
    data_dist = (xyz[nbr_list[:, 0 ], nbr_list[:, 1]] - xyz[nbr_list[:, 0], nbr_list[:, 2]]).pow(2).sum(-1).sqrt()
    loss_dist = (gen_dist - data_dist).pow(2).mean()
    
    return loss_dist 

def get_tetra_idx(traj, g):   
    tetra_index = {}

    for atom in list(traj.top.atoms):
        if atom.element.atomic_number == 6:
            nbr_list = []
            for n in g.neighbors(atom):
                #if n.element.atomic_number == 1:
                    nbr_list.append(n.index)
            if len(nbr_list) == 4: 
                tetra_index[atom.index] = nbr_list
    return tetra_index



def compute_HCH(xyz, tetra_index):
    ch_pair = torch.triu(torch.ones(4, 4), diagonal=1).nonzero()

    for methyl in tetra_index.keys():
        dCH = xyz[:, [methyl], :] - xyz[:, tetra_index[methyl], :]
        norm = dCH.pow(2).sum(-1).sqrt()
        dCH = dCH / norm.unsqueeze(-1)
        HCH = (dCH[:, ch_pair[:,0 ]] * dCH[:, ch_pair[:,1]]).sum(-1) 
        
    return (HCH - (-0.333) ).pow(2).mean()


def loop(loader, optimizer, device, model, epoch, gamma, kappa, tetra_index, train=True, looptext='', tqdm_flag=True):
    
    epoch_recon_loss = []
    epoch_dist_loss = []
    epoch_methyl_loss = []
    epoch_dihe_loss = []
    
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
        nbr_list = batch['hyperedges']
        assign, xyz, xyz_recon = model(batch)

        dihe_recon = compute_dihedral_vec(batch['dihedrals'], xyz_recon)
        dihe_data = compute_dihedral_vec(batch['dihedrals'], xyz)
        
        # compute loss
        loss_recon = (xyz_recon - xyz).pow(2).mean() # recon loss
        loss_dihe = (dihe_recon - dihe_data).pow(2).mean()
        loss_dist = dist_loss(xyz, xyz_recon, nbr_list) # distance loss 
        loss_methyl = compute_HCH(xyz_recon, tetra_index) # methyl cap loss 
        loss = loss_recon + gamma * loss_dist + kappa * loss_dihe

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward()

        epoch_recon_loss.append(loss_recon.item())
        epoch_dist_loss.append(loss_dist.item())
        epoch_methyl_loss.append(loss_methyl.item())
        epoch_dihe_loss.append(loss_dihe.item())

        mean_recon = np.array(epoch_recon_loss).mean()
        mean_dist = np.array(epoch_dist_loss).mean()
        mean_methyl = np.array(epoch_methyl_loss).mean()
        mean_dihe = np.array(epoch_dihe_loss).mean()
        
        del loss, loss_dist, loss_methyl

        postfix = ['avg. recon loss={:.4f}'.format(mean_recon),
                    'avg. dist loss={:.4f}'.format(mean_dist),
                    'avg. dihe loss={:.4f}'.format(mean_dihe)]
        if tqdm_flag:
            loader.set_postfix_str(' '.join(postfix))

    if not tqdm_flag and epoch % 20 == 0:
        for result in postfix:
            print(result)
    
    return mean_recon, mean_dist, mean_dihe, assign, xyz.detach().cpu(), xyz_recon.detach().cpu() 

def run(params):

    device = params['device']
    batch_size = params['batch_size']
    N_cg = params['N_cg']
    n_epochs = params['n_epochs']
    cg_method = params['cg_method']
    working_dir = params['logdir']
    lr = params['lr']
    tqdm_flag = params['tqdm_flag']
    n_data = params['ndata']
    kappa = params['kappa']
    gamma = params['gamma']
    dataset_label = params['dataset']
    cutoff = params['cutoff']
    cross = params['cross']
    edgeorder = params['edgeorder']
    n_splits = params['n_splits']
    mapshuffle = params['mapshuffle']
    create_dir(working_dir)

    if params['knbr'] == 0:
        params['knbr'] = N_cg - 1
    
    traj_files = glob.glob(PROTEINFILES[dataset_label]['traj_paths'])[:200]
    pdb_file = PROTEINFILES[dataset_label]['pdb_path']
    file_type = PROTEINFILES[dataset_label]['file_type']

    trajs = [md.load_xtc(file,
                top=pdb_file, stride=50) for file in traj_files]

    traj = trajs[0]

    atomic_nums, protein_index = get_atomNum(trajs[0])
    n_atoms = len(atomic_nums)

    # get protin graph
    protein_top = trajs[0].top.subset(protein_index)
    g = protein_top.to_bondgraph()

    props = get_diffpool_data(N_cg, trajs[:n_data], n_data=n_data, edgeorder=edgeorder, pdb=pdb_file)

    dataset = DiffPoolDataset(props)
    dataset.generate_neighbor_list(cutoff)

    kf = KFold(n_splits=n_splits)

    split_iter = kf.split(list(range(len(dataset))))

    tetra_index = get_tetra_idx(trajs[0], g)


    cv_heavy_rmsd = []
    cv_all_rmsd = []
    cv_heavy_ged = []
    cv_all_ged = []

    cv_stats_pd = pd.DataFrame({'train_recon': [], 
            'test_all_recon': [],  'test_heavy_recon': [],
            'train_graph': [], 'test_graph': [],
            'train_tetra': [], 'test_tetra': [],
            'all atom ged': [], 'heavy atom ged': [], 
            'all atom graph valid ratio': [], 
            'heavy atom graph valid ratio': []} )

    for i, (train_index, test_index) in enumerate(split_iter):

        split_dir = os.path.join(working_dir, 'fold{}'.format(i)) 
        create_dir(split_dir)

        train_index, val_index = train_test_split(list(range(len(train_index))),test_size=0.1)

        trainset = get_subset_by_indices(train_index, dataset)
        valset = get_subset_by_indices(val_index, dataset)
        testset = get_subset_by_indices(test_index, dataset)

        trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True)
        valloader = DataLoader(valset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True)

        # get cg_map 
        if cg_method == 'newman':
            paritions = get_partition(g, N_cg)
            mapping = parition2mapping(paritions, n_atoms)
            assign_idx = torch.LongTensor( np.array(mapping) ) 

        elif cg_method == 'random':
            mapping = get_random_mapping(N_cg, n_atoms)
            assign_idx = torch.LongTensor( np.array(mapping) )

        elif cg_method == 'backbonepartition': 
            mapping = backbone_partition(trajs[0], N_cg)
            true_n_cgs = len(list(set(mapping.tolist())))
            
            if true_n_cgs < N_cg:
                while true_n_cgs < N_cg:
                    mapping = backbone_partition(trajs[0], N_cg)
                    true_n_cgs = len(list(set(mapping.tolist())))

            assign_idx = torch.LongTensor( np.array(mapping) )

        # shuffle if mapshuffle is on 
        if mapshuffle > 0.0:
            ran_idx = random.sample(range(assign_idx.shape[0]), int(mapshuffle * assign_idx.shape[0])  )
            idx2map = assign_idx[ran_idx]
            assign_idx[ran_idx] = idx2map

        pooler = CGpool(1, 16, n_atoms=n_atoms, n_cgs=N_cg, assign_idx=assign_idx)

        if params['model'] == 'equilinear':
            model = EquiLinear(pooler, N_cg, n_atoms, cross=cross, 
                               knn=params['knbr']).to(device)
        elif params['model'] == 'linear': 
            model = Baseline(pooler, N_cg, n_atoms).to(device)
        elif params['model'] == 'mlp':
            model = MLP(pooler, N_cg, n_atoms, width=params['width'], 
                        depth=params['depth'], activation=params['activation']).to(device)
        elif params['model'] == 'equimlp':
            # model = EquiMLP2(pooler, N_cg, n_atoms, knn=params['knbr'], 
            #                     width=params['width'], depth=params['depth'],
            #                     activation=params['activation']).to(device)

            model = edgesetMLP(pooler, N_cg, n_atoms, knn=params['knbr'], 
                    activation=params['activation'],
                    depth=params['depth'],
                    feature_dim=params['feature_dim'], 
                    cutoff=cutoff).to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, 
                                                                factor=0.6, verbose=True, 
                                                                threshold=1e-4,  min_lr=1e-7)
        
        failed = False 

        train_log = pd.DataFrame({'epoch': [], 'lr': [],
                    'train_recon': [], 'val_recon': [],
                   'train_graph': [], 'val_graph': []})

        # train     
        for epoch in range(n_epochs):
        
            mean_train_recon, mean_train_graph, mean_train_tetra, assign, train_xyz, train_xyz_recon = loop(trainloader, optimizer, device, model, 
                                                                        epoch, gamma, kappa, tetra_index, 
                                                                        train=True, looptext='', tqdm_flag=tqdm_flag)

            mean_val_recon, mean_val_graph, mean_val_tetra, assign, val_xyz, val_xyz_recon = loop(valloader, optimizer, device, model, 
                                                                        epoch, gamma, kappa, tetra_index, 
                                                                        train=False, looptext='', tqdm_flag=tqdm_flag)

            if np.isnan(mean_train_recon):
                print("NaN encoutered, exiting...")
                failed = True
                break 
                
            # if epoch % 5 == 0:
            #     map_save_path = os.path.join(working_dir, 'map_{}.png'.format(epoch) )
            #     #plot_map(assign[0], props['z'][0].numpy(), map_save_path)

            scheduler.step(mean_val_recon)
            # early stopping 
            if optimizer.param_groups[0]['lr'] <= 1.5e-7:
                break

            stats = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 
                    'train_recon': mean_train_recon, 'val_recon': mean_val_recon,
                    'train_graph': mean_train_graph, 'val_graph': mean_val_graph}

            train_log = train_log.append(stats, ignore_index=True)
            train_log.to_csv(os.path.join(split_dir, 'train_log.csv'),  index=False)

        dump_numpy2xyz(train_xyz_recon, props['z'][0].numpy(), os.path.join(split_dir, 'train_recon.xyz'))

        # test 
        testloader = DataLoader(testset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True)
        model.eval()

        mean_test_recon, mean_test_graph, mean_test_tetra, assign, test_xyz, test_xyz_recon = loop(testloader, optimizer, device, model, 
                                                                    epoch, gamma, kappa, tetra_index, 
                                                                    train=False, looptext='', tqdm_flag=tqdm_flag)

        all_test_xyz_data, all_test_xyz_recon, heavy_ged, all_ged, heavy_valid_ratio, all_valid_ratio = retrieve_recon_structures(testloader, device, model, tqdm_flag=True)

        # compute rmsd 
        atomic_nums = props['z'][0].numpy()
        heavy_filter = atomic_nums != 1.

        test_all_dxyz = (all_test_xyz_data - all_test_xyz_recon) #.reshape(-1)
        test_heavy_dxyz = (all_test_xyz_data - all_test_xyz_recon)[:, heavy_filter, :]#.reshape(-1)

        test_all_rmsd = np.sqrt(np.power(test_all_dxyz, 2).sum(-1).mean())
        test_heavy_rmsd = np.sqrt(np.power(test_heavy_dxyz, 2).sum(-1).mean())

        # dump train recon 
        dump_numpy2xyz(all_test_xyz_recon[:32], props['z'][0].numpy(), os.path.join(split_dir, 'test_recon.xyz'))

        print("split {}".format(i))
        print("test rmsd (all atoms): {:.4f}".format(test_all_rmsd))
        print("test rmsd (heavy atoms): {:.4f}".format(test_heavy_rmsd))
        print("test rel ged (all atoms) : {:.4f}".format(all_ged))
        print("test rel ged (heavy atoms) : {:.4f}".format(heavy_ged))

        cv_all_rmsd.append(test_all_rmsd)
        cv_heavy_rmsd.append(test_heavy_rmsd)
        cv_all_ged.append(all_ged)
        cv_heavy_ged.append(heavy_ged)

        test_stats = { 'train_recon': mean_train_recon, 
            'test_all_recon': test_all_rmsd,  'test_heavy_recon': test_heavy_rmsd,
            'train_graph': mean_train_graph, 'test_graph': mean_test_graph,
            'train_tetra': mean_train_tetra, 'test_tetra': mean_test_tetra,
            'all atom ged': all_ged, 'heavy atom ged': heavy_ged, 
            'all atom graph valid ratio': all_valid_ratio, 
            'heavy atom graph valid ratio': heavy_valid_ratio} 

        cv_stats_pd = cv_stats_pd.append(test_stats, ignore_index=True)
        cv_stats_pd.to_csv(os.path.join(working_dir, 'cv_stats.csv'),  index=False)

        if failed:
            break 

    cv_all_rmsd = np.array(cv_all_rmsd)
    cv_heavy_rmsd = np.array(cv_heavy_rmsd)
    cv_all_ged = np.array(cv_all_ged)
    cv_heavy_ged = np.array(cv_heavy_ged)

    print("CV results N = {}".format(N_cg))
    print("heavy rmsd {} +- {}".format(cv_heavy_rmsd.mean(), cv_heavy_rmsd.std()) )
    print("all rmsd {} +- {}".format(cv_all_rmsd.mean(), cv_all_rmsd.std()) )
    print("heavy ged diff {} +- {}".format(cv_heavy_ged.mean(), cv_heavy_ged.std()) )
    print("all ged diff {} +- {}".format(cv_all_ged.mean(), cv_all_ged.std()) )

    return cv_all_rmsd, cv_all_ged, failed, assign

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-logdir', type=str)
    parser.add_argument('-model', type=str, default='equilinear')
    parser.add_argument("-dataset", type=str, default='dipeptide')
    parser.add_argument('-device', type=int)
    parser.add_argument('-cutoff', type=float, default=2.5)
    parser.add_argument('-batch_size', type=int,default= 32)
    parser.add_argument('-N_cg', type=int, default= 3)
    parser.add_argument('-width', type=int, default= 1)
    parser.add_argument('-depth', type=int, default= 1)
    parser.add_argument('-edgeorder', type=int, default= 2)
    parser.add_argument('-n_splits', type=int, default= 3)
    parser.add_argument('-n_epochs', type=int, default= 50)
    parser.add_argument('-ndata', type=int, default=2000)
    parser.add_argument('-knbr', type=int, default=0)
    parser.add_argument("-cg_method", type=str, default='newman')
    parser.add_argument("-activation", type=str, default='ReLU')
    parser.add_argument("-mapshuffle", type=float, default=0.0)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-gamma', type=float, default=0.0)
    parser.add_argument('-kappa', type=float, default=0.0)
    parser.add_argument("--tqdm_flag", action='store_true', default=False)
    parser.add_argument("--cross", action='store_true', default=False)

    params = vars(parser.parse_args())

    run(params)