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
from sklearn.model_selection import KFold
from torch.nn import Sequential 
import numpy as np
import copy
from torch_scatter import scatter_mean
from tqdm import tqdm 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import json
import pandas as pd 
import torch.autograd.profiler as profiler
import statsmodels.api as sm

def plot_map(assign, z, save_path=None):
    mapping = assign.detach().cpu().numpy().transpose()
    z = z.tolist()
    z = [int(el) for el in z] 
    plt.imshow( mapping)
    plt.xticks(list(range(mapping.shape[1])), z)
    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()


# sampling code 

def sample(loader, device, model, tau_min, tqdm_flag=True, working_dir=None):

    model.eval()

    if tqdm_flag:
        loader = tqdm(loader, position=0, file=sys.stdout,
                         leave=True, desc='sampling')

    xyz_samples = []

    for i, batch in enumerate(loader):     
        batch = batch_to(batch, device=device)
        xyz, H_mu, H_sigma = model.sample(batch, tau_min)

        xyz_samples.append(xyz.detach().cpu().numpy())

    xyz_samples = np.concatenate(xyz_samples)
    # get atoms object 
    atomic_nums = batch['z'][0].detach().cpu().numpy()

    ref_atoms = Atoms(numbers=atomic_nums, positions=batch['xyz'][0].detach().cpu().numpy())

    # compute graph metrics 
    heavy_valid_rmsds = []
    all_valid_rmsds = []
    heavy_geds = []
    all_geds = []
    all_valid_ratios = []
    heavy_valid_ratios = []


    for xyz_sample in xyz_samples:
        sample_atoms = Atoms(positions=xyz_sample, numbers=atomic_nums)
        # all_sample_atoms.append(sample_atoms)
        # compute ged 
        heavy_valid_ids, heavy_valid_ratio, heavy_ged = count_valid_graphs(ref_atoms, [sample_atoms], heavy_only=True)
        all_valid_ids, all_valid_ratio, all_ged = count_valid_graphs(ref_atoms, [sample_atoms], heavy_only=False)

        heavy_rmsds = compute_rmsd([sample_atoms], ref_atoms, heavy_valid_ids)
        all_rmsds = compute_rmsd([sample_atoms], ref_atoms, all_valid_ids) 

        if heavy_rmsds is not None:
            heavy_valid_rmsds += heavy_rmsds[:, 1].tolist()
        if all_rmsds is not None:
            all_valid_rmsds += all_rmsds[:, 0].tolist()

        heavy_geds += heavy_ged
        all_geds += all_ged

        all_valid_ratios.append(all_valid_ratio)
        heavy_valid_ratios.append(heavy_valid_ratio)

    if working_dir is not None:
        np.savetxt(os.path.join(working_dir, 'heavy_geds.txt'),  np.array(heavy_geds))
        np.savetxt(os.path.join(working_dir, 'all_geds.txt'),  np.array(all_geds))
        if len(heavy_valid_rmsds) != 0 :
            np.savetxt(os.path.join(working_dir, 'heavy_rmsds.txt'),  np.array(heavy_valid_rmsds))
        if len(all_valid_rmsds) != 0 :    
            np.savetxt(os.path.join(working_dir, 'all_rmsds.txt'),  np.array(all_valid_rmsds))

    # print out something 
    dump_numpy2xyz(xyz_samples[:50], atomic_nums, os.path.join(working_dir, 'test_samples.xyz'))

    all_valid_ratio = np.array(all_valid_ratios).mean()
    heavy_valid_ratio = np.array(heavy_valid_ratio).mean()

    return all_valid_rmsds, heavy_valid_rmsds, all_geds, heavy_geds, all_valid_ratio, heavy_valid_ratio


def pretrain(loader, optimizer, device, model, tau, target_mapping, tqdm_flag):

    natoms = target_mapping.shape[0]
    target = torch.zeros(natoms, target_mapping.max() + 1)
    target[list(range(natoms)), target_mapping] = 1. # n X N
    target = target.to(device)

    if tqdm_flag:
        loader = tqdm(loader, position=0, file=sys.stdout,
                         leave=True, desc='(pretrain epoch)')

    all_loss = []
    for i, batch in enumerate(loader):     
        batch = batch_to(batch, device=device)
        xyz, xyz_recon, soft_assign, adj, cg_xyz, soft_cg_adj, H_prior_mu, H_prior_sigma, H_mu, H_sigma = model(batch, tau)

        cg_xyz_lift = torch.einsum('bce,bac->bae', cg_xyz, soft_assign) # soft_assign is not normalized 
        
        #loss = (cg_xyz_lift - xyz).pow(2).mean()

        loss = (soft_assign - target[None, ...]).pow(2).mean()
        #loss = (model.pooler.assign_map - target * 3.0).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss.append(loss.item())

        postfix = ['avg. loss={:.4f}'.format(np.array(all_loss).mean())]

        if tqdm_flag:
            loader.set_postfix_str(' '.join(postfix))

    return np.array(all_loss).mean(), soft_assign
    
def loop(loader, optimizer, device, model, tau_sched, epoch, beta, eta,
        gamma, kappa, train=True, looptext='', tqdm_flag=True, recon_weight=1.0, tau_min=None):

    all_loss = []
    recon_loss = []
    adj_loss = []
    ent_loss = []
    KL_loss = []
    graph_loss = []
    
    if train:
        model.train()
        mode = '{} train'.format(looptext)
    else:
        model.eval() # yes, still set to train when reconstructing
        mode = '{} valid'.format(looptext)
        
        
    if tqdm_flag:
        loader = tqdm(loader, position=0, file=sys.stdout,
                         leave=True, desc='({} epoch #{})'.format(mode, epoch))
        
        
    for i, batch in enumerate(loader):     
        batch = batch_to(batch, device=device)
        
        if tau_min == None:
            tau = tau_sched[ len(loader) * epoch +  i]
        else:
            tau = tau_min

        xyz, xyz_recon, assign, adj, cg_xyz, soft_cg_adj, H_prior_mu, H_prior_sigma, H_mu, H_sigma = model(batch, tau)
        
        # compute a loss that penalize atoms that assigned too far away
        # compute loss
        loss_recon = (xyz_recon - xyz).pow(2).mean()
        loss_entropy = -(assign * torch.log(assign)).sum(-1).mean()

        node_sim_mat = assign.matmul(assign.transpose(1,2))
        loss_adj = ((node_sim_mat - adj).pow(2).sum(-1).sum(-1) + EPS).sqrt().mean()
       # loss_adj = soft_cg_adj.diagonal(dim1=1, dim2=2 ).mean()
        
        loss_kl = KL(H_mu, H_sigma, H_prior_mu, H_prior_sigma) 
        
        nbr_list = batch['hyperedges']
        gen_dist = ((xyz_recon[nbr_list[:, 0 ], nbr_list[:, 1]] - xyz_recon[nbr_list[:, 0], nbr_list[:, 2]]).pow(2).sum(-1) + EPS).sqrt()
        data_dist = ((xyz[nbr_list[:, 0 ], nbr_list[:, 1]] - xyz[nbr_list[:, 0], nbr_list[:, 2]]).pow(2).sum(-1) + EPS).sqrt()
        loss_graph = (gen_dist - data_dist).pow(2).mean()

        #loss = recon_weight * loss_recon + beta * loss_kl +  gamma * loss_graph + eta * loss_adj #+  kappa * loss_entropy #+ 0.0001 * prior_reg
        loss = eta * loss_adj + loss_recon +  gamma * loss_graph + beta * loss_kl + kappa * loss_entropy


        loss_main = (loss_recon +  gamma * loss_graph + beta * loss_kl).item()
        # if  torch.isnan(loss):
        #     del loss_recon, loss_kl, loss_adj, loss_entropy
        #     continue 
        #if epoch % 5 == 0:
        #    print(H_prior_mu.mean().item(), H_prior_sigma.mean().item(), H_mu.mean().item(), H_sigma.mean().item())
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss.backward()

        all_loss.append(loss_main)
        recon_loss.append(loss_recon.item())
        adj_loss.append(loss_adj.item())
        ent_loss.append(loss_entropy.item())
        KL_loss.append(loss_kl.item())
        graph_loss.append(loss_graph.item())

        mean_recon = np.array(recon_loss).mean()
        mean_graph = np.array(graph_loss).mean()
        mean_adj = np.array(adj_loss).mean()
        mean_ent = np.array(ent_loss).mean()
        mean_KL = np.array(KL_loss).mean()
        mean_loss = np.array(all_loss).mean()
        
        del loss_recon, loss_kl, loss_adj, loss_entropy, adj, cg_xyz, soft_cg_adj, H_prior_mu, H_prior_sigma, H_mu, H_sigma

        postfix = ['avg. total loss={:.4f}'.format(mean_loss),
                    'avg. recon loss={:.4f}'.format(mean_recon),
                    'avg. graph loss={:.4f}'.format(mean_graph),
                    'avg. KL loss={:.4f}'.format(mean_KL),
                   'avg. adj loss={:.4f}'.format(mean_adj),
                   'avg. entropy loss={:.4f}'.format(mean_ent),
                   'tau = {:.4f}'.format(tau)]
        if tqdm_flag:
            loader.set_postfix_str(' '.join(postfix))
    
    if not tqdm_flag:
        for result in postfix:
            print(result)
    
    return mean_loss, mean_recon, mean_graph, mean_KL, assign, xyz.detach().cpu(), xyz_recon.detach().cpu() 

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
    beta = params['beta']
    eta = params['eta']
    gamma = params['gamma']
    kappa = params['kappa']
    lr = params['lr']
    working_dir = params['logdir']
    tqdm_flag = params['tqdm_flag']
    cg_method = params['cg_method']
    n_data = params['n_data']
    label =params['dataset']
    tau_min = params['tau_min']
    det = params['det']
    cg_cutoff = params['cg_cutoff']
    tau_pre = params['tau_pre']
    nsplits = params['nsplits']

    print(eta, gamma)
    create_dir(working_dir)

    if label in PROTEINFILES.keys():
        traj = load_protein_traj(label)
        atomic_nums, protein_index = get_atomNum(traj)
        n_atoms = len(atomic_nums)
    else:
        raise ValueError("data label {} not recognized".format(label))

    # get cg_map 

    # compute mapping using Girvan Newman 
    protein_top = traj.top.subset(protein_index)
    g = protein_top.to_bondgraph()
    paritions = get_partition(g, N_cg)
    newman_mapping = parition2mapping(paritions, n_atoms)

    if cg_method == 'newman':
        assign_idx = torch.LongTensor( np.array(newman_mapping) ) 
    elif cg_method == 'random':
        mapping = get_random_mapping(N_cg, n_atoms)
        assign_idx = torch.LongTensor( np.array(mapping) ) 

    elif cg_method == 'diff':
        assign_idx = None

    props = get_diffpool_data(N_cg, [traj], n_data=n_data, edgeorder=params['edgeorder'],
                             pdb=None #PROTEINFILES[label]['pdb_path']
                             )

    dataset = DiffPoolDataset(props)
    dataset.generate_neighbor_list(cutoff)

    cv_stats_pd = pd.DataFrame( { 'train_recon': [], 'test_recon': [],
                'train_KL': [], 'test_KL': [], 
                'train_graph': [], 'test_graph': [],
                'all atom ged': [], 'heavy atom ged': [], 
                'all atom graph valid ratio': [], 
                'heavy atom graph valid ratio': [],
                'all atom rmsd': [], 'heavy atom rmsd':[]} )

    # split train, test index 
    # train_index, test_index = train_test_split(list(range(len(dataset))),test_size=0.2)
    # train_index, val_index = train_test_split(list(range(len(train_index))),test_size=0.1)

    if nsplits == 1:
        split_iter = [ train_test_split(list(range(len(dataset))),test_size=0.2) ]
    else:
        kf = KFold(n_splits=nsplits, shuffle=True)
        split_iter = kf.split(list(range(n_data)))

    for i, (train_index, test_index) in enumerate(split_iter):

        split_dir = os.path.join(working_dir, 'fold{}'.format(i))
        create_dir(split_dir)

        train_index, val_index = train_test_split(list(range(len(train_index))),test_size=0.1)

        trainset = get_subset_by_indices(train_index, dataset)
        valset = get_subset_by_indices(val_index, dataset)
        testset = get_subset_by_indices(test_index, dataset)

        trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True)
        valloader = DataLoader(valset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True)

        n_train_iters = len(trainloader) * n_epochs + 10
        n_val_iters = len(valloader) * n_epochs + 10
        tau_train_sched = (tau_0 - tau_min) * np.exp(-tau_rate * torch.linspace(0, n_train_iters-1, n_train_iters)) + tau_min
        tau_val_sched = (tau_0 - tau_min) * np.exp(-tau_rate * torch.linspace(0, n_train_iters-1, n_val_iters)) + tau_min

        pooler = CGpool(nconv_pool, num_features, n_atoms=n_atoms, n_cgs=N_cg, assign_idx=assign_idx)
        
        encoder = DenseEquiEncoder(n_conv=enc_nconv, 
                               n_atom_basis=num_features,
                               n_rbf=n_rbf, 
                               activation=activation, 
                               cutoff=cutoff)
        
        decoder = DenseEquivariantDecoder(n_atoms=n_atoms, n_atom_basis=num_features,
                                          n_rbf=n_rbf, cutoff=cg_cutoff, 
                                          num_conv=dec_nconv, activation=activation)

        prior = DenseCGPrior(n_atoms=n_atoms, n_atom_basis=num_features,
                                          n_rbf=n_rbf, cutoff=cg_cutoff, 
                                          num_conv=enc_nconv, activation=activation)

        atom_mu = nn.Sequential(nn.Linear(num_features, num_features), nn.ReLU(), nn.Linear(num_features, num_features))
        atom_sigma = nn.Sequential(nn.Linear(num_features, num_features), nn.ReLU(), nn.Linear(num_features, num_features))
        
        model = DiffPoolVAE(encoder=encoder,decoder=decoder, pooler=pooler, atom_munet=atom_mu, atom_sigmanet=atom_sigma, prior=prior, det=det).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, 
                                                                factor=0.6, verbose=True, 
                                                                threshold=1e-4,  min_lr=1e-8)

        early_stopping = EarlyStopping(patience=params['patience'])
        
        failed = False 


        if cg_method == 'diff':
            for epoch in range(params['n_pretrain']):
                model.train()
                graph_loss, assign = pretrain(trainloader, optimizer, device, model, params['tau_pre'], newman_mapping, tqdm_flag=tqdm_flag)

                if np.isnan(graph_loss):
                    print("NaN encoutered, exiting...")
                    failed = True
                    break 

                if epoch % params['mapsavefreq'] == 0:
                    map_save_path = os.path.join(split_dir, 'pretrain_map_{}.png'.format(epoch) )
                    plot_map(assign[0], props['z'][0].numpy(), map_save_path)

        loss_log = []

        train_log = pd.DataFrame({'epoch': [], 'lr': [], 'train_loss': [], 'val_loss': [], 
                    'train_recon': [], 'val_recon': [],
                   'train_KL': [], 'val_KL': [], 'train_graph': [], 'val_graph': []})

        # train     
        for epoch in range(n_epochs):
            model.train()
            mean_train_loss, mean_train_recon, mean_train_graph, mean_train_KL, assign, train_xyz, train_xyz_recon = loop(trainloader, optimizer, device, model, tau_train_sched, epoch, 
                                            beta, eta, gamma,  kappa, train=True, looptext='', tqdm_flag=tqdm_flag)

            mean_val_loss, mean_val_recon, mean_val_graph, mean_val_KL, assign, val_xyz, val_xyz_recon = loop(valloader, optimizer, device, model, tau_val_sched, epoch, 
                                            beta, eta, gamma,  kappa, train=False, looptext='', tqdm_flag=tqdm_flag)

            if np.isnan(mean_train_loss):
                print("NaN encoutered, exiting...")
                failed = True
                break 

            stats = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 
                    'train_loss': mean_train_loss, 'val_loss': mean_val_loss, 
                    'train_recon': mean_train_recon, 'val_recon': mean_val_recon,
                    'train_KL': mean_train_KL, 'val_KL': mean_val_KL, 
                    'train_graph': mean_train_graph, 'val_graph': mean_val_graph}

            train_log = train_log.append(stats, ignore_index=True)

            smooth = sm.nonparametric.lowess(train_log['val_loss'].values,  # y
                                            train_log['epoch'].values, # x
                                            frac=0.2)
            smoothed_valloss = smooth[-1, 1]


            if epoch % params['mapsavefreq'] == 0:
                map_save_path = os.path.join(split_dir, 'map_{}.png'.format(epoch) )
                plot_map(assign[0], props['z'][0].numpy(), map_save_path)

            scheduler.step(smoothed_valloss)
            loss_log.append([mean_train_recon, mean_val_recon])

            # early stopping 
            if optimizer.param_groups[0]['lr'] <= 5e-8:
                break

            early_stopping(smoothed_valloss)
            if early_stopping.early_stop:
                break

        # dump log 
    #    np.savetxt(os.path.join(split_dir, 'train_val_recon.txt'),  np.array(loss_log))
        train_log.to_csv(os.path.join(split_dir, 'train_log.csv'),  index=False)

        dump_numpy2xyz(train_xyz_recon, props['z'][0].numpy(), os.path.join(split_dir, 'train_recon.xyz'))

        # test 
        testloader = DataLoader(testset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True)
        mean_test_loss, mean_test_recon, mean_test_graph, mean_test_KL, assign, test_xyz, test_xyz_recon = loop(testloader, optimizer, device, model, tau_val_sched, epoch, beta, 
                                    gamma, eta, kappa, train=False, looptext='testing', tqdm_flag=tqdm_flag, tau_min=tau_min)


        test_stats = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 
                'train_recon': mean_train_recon, 'test_recon': mean_test_recon,
                'train_KL': mean_train_KL, 'test_KL': mean_test_KL, 
                'train_graph': mean_train_graph, 'test_graph': mean_test_graph}

        with open(os.path.join(split_dir, 'train_test_stats.json'), 'w') as f:
            json.dump(test_stats, f)

        # sampling 
        all_rmsds, heavy_rmsds, all_geds, heavy_geds, all_valid_ratio, heavy_valid_ratio = sample(testloader,  device, model, tau_min, tqdm_flag=True, working_dir=split_dir)

        mean_all_ged = np.array(all_geds).mean()
        mean_heavy_ged = np.array(heavy_geds).mean()

        if len(all_rmsds) != 0:
            mean_all_rmsd = np.array(all_rmsds).mean()
        else:
            mean_all_rmsd = None

        if len(heavy_rmsds) != 0:
            mean_heavy_rmsd = np.array(heavy_rmsds).mean()
        else:
            mean_heavy_rmsd = None

        # dump train recon 
        dump_numpy2xyz(test_xyz_recon, props['z'][0].numpy(), os.path.join(split_dir, 'test_recon.xyz'))

        print("train msd : {:.4f}".format(mean_train_recon))
        print("test msd : {:.4f}".format(mean_test_recon))

        test_stats = { 'train_recon': mean_train_recon, 'test_recon': mean_test_recon,
                'train_KL': mean_train_KL, 'test_KL': mean_test_KL, 
                'train_graph': mean_train_graph, 'test_graph': mean_test_graph,
                'all atom ged': mean_all_ged, 'heavy atom ged': mean_heavy_ged, 
                'all atom graph valid ratio': all_valid_ratio, 
                'heavy atom graph valid ratio': heavy_valid_ratio,
                'all atom rmsd': mean_all_rmsd, 'heavy atom rmsd':mean_heavy_rmsd}

        cv_stats_pd = cv_stats_pd.append(test_stats, ignore_index=True)
        cv_stats_pd.to_csv(os.path.join(working_dir, 'cv_stats.csv'),  index=False)

        with open(os.path.join(split_dir, 'train_test_stats.json'), 'w') as f:
            json.dump(test_stats, f)

    # dump cv stats 

    return mean_test_recon, mean_all_ged, failed, assign

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-logdir', type=str)
    parser.add_argument('-device', type=int)
    parser.add_argument('-dataset', type=str, default='dipeptide')
    parser.add_argument('-num_features',  type=int, default=512)
    parser.add_argument('-batch_size', type=int,default= 32)
    parser.add_argument('-n_data', type=int, default= 1000)
    parser.add_argument('-patience', type=int, default= 10)
    parser.add_argument('-N_cg', type=int, default= 3)
    parser.add_argument('-nsplits', type=int, default=1)
    parser.add_argument('-nconv_pool', type=int, default= 7)
    parser.add_argument('-enc_nconv', type=int, default= 4)
    parser.add_argument('-dec_nconv', type=int, default= 3)
    parser.add_argument('-mapsavefreq', type=int, default= 5)
    parser.add_argument('-cutoff', type=float, default= 8.0)
    parser.add_argument('-cg_cutoff', type=float, default=8.0)
    parser.add_argument('-n_rbf', type=int,  default= 7)
    parser.add_argument('-activation', type=str,  default= 'ELU')
    parser.add_argument('-n_epochs', type=int, default= 50)
    parser.add_argument('-edgeorder', type=int, default= 2)
    parser.add_argument("-n_pretrain", type=int, default= 10)
    parser.add_argument('-tau_rate', type=float, default= 0.004 )
    parser.add_argument('-tau_0', type=float, default= 2.36)
    parser.add_argument('-tau_min', type=float, default= 0.3)
    parser.add_argument('-tau_pre', type=float, default= 0.3)
    parser.add_argument('-beta', type=float, default= 0.1)
    parser.add_argument('-gamma', type=float, default= 0.1)
    parser.add_argument('-eta', type=float, default= 0.1)
    parser.add_argument('-kappa', type=float, default= 0.1)
    parser.add_argument('-lr', type=float, default=5e-5)
    parser.add_argument("--tqdm_flag", action='store_true', default=False)
    parser.add_argument("--det", action='store_true', default=False)
    parser.add_argument("-cg_method", type=str, default='diff')

    params = vars(parser.parse_args())

    run(params)