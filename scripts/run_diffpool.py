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
import torch.autograd.profiler as profiler

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

    if working_dir is not None:
        np.savetxt(os.path.join(working_dir, 'heavy_geds.txt'),  np.array(heavy_geds))
        np.savetxt(os.path.join(working_dir, 'all_geds.txt'),  np.array(all_geds))
        if len(heavy_valid_rmsds) != 0 :
            np.savetxt(os.path.join(working_dir, 'heavy_rmsds.txt'),  np.array(heavy_valid_rmsds))
        if len(all_valid_rmsds) != 0 :    
            np.savetxt(os.path.join(working_dir, 'all_rmsds.txt'),  np.array(all_valid_rmsds))

    # print out something 
    dump_numpy2xyz(xyz_samples[:50], atomic_nums, os.path.join(working_dir, 'test_samples.xyz'))


    return all_rmsds, heavy_rmsds, all_geds, heavy_geds


def loop(loader, optimizer, device, model, tau_sched, epoch, beta, eta,
        gamma, kappa, train=True, looptext='', tqdm_flag=True, recon_weight=1.0, tau_min=None):
    
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

        xyz, xyz_recon, assign, adj, soft_cg_adj, H_prior_mu, H_prior_sigma, H_mu, H_sigma = model(batch, tau)
        
        # compute loss
        loss_recon = (xyz_recon - xyz).pow(2).mean()
        loss_entropy = soft_cg_adj.diagonal(dim1=1, dim2=2).std(-1).mean()# -(assign * torch.log(assign)).sum(-1).mean()
        node_sim_mat = assign.matmul(assign.transpose(1,2))
        loss_adj = (node_sim_mat - adj).pow(2).mean()

        loss_kl = KL(H_mu, H_sigma, H_prior_mu, H_prior_sigma) 
        
        nbr_list = batch['nbr_list']
        gen_dist = (xyz_recon[nbr_list[:, 0 ], nbr_list[:, 1]] - xyz_recon[nbr_list[:, 0], nbr_list[:, 2]]).pow(2).sum(-1)#.sqrt()
        data_dist = (xyz[nbr_list[:, 0 ], nbr_list[:, 1]] - xyz[nbr_list[:, 0], nbr_list[:, 2]]).pow(2).sum(-1)#.sqrt()
        loss_graph = (gen_dist - data_dist).pow(2).mean()

        loss = recon_weight * loss_recon + beta * loss_kl +  gamma * loss_graph + eta * loss_adj +  kappa * loss_entropy #+ 0.0001 * prior_reg

        #if epoch % 5 == 0:
        #    print(H_prior_mu.mean().item(), H_prior_sigma.mean().item(), H_mu.mean().item(), H_sigma.mean().item())
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss.backward()

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
        
        del loss_adj, loss_entropy, loss_recon

        postfix = ['avg. recon loss={:.4f}'.format(mean_recon),
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

    create_dir(working_dir)

    if label in PROTEINFILES.keys():
        traj = load_protein_traj(label)
        atomic_nums, protein_index = get_atomNum(traj)
        n_atoms = len(atomic_nums)
    else:
        raise ValueError("data label {} not recognized".format(label))

    # get cg_map 
    if cg_method == 'newman':
        protein_top = traj.top.subset(protein_index)
        g = protein_top.to_bondgraph()
        paritions = get_partition(g, N_cg)
        mapping = parition2mapping(paritions, n_atoms)

        assign_idx = torch.LongTensor( np.array(mapping) ) 
    elif cg_method == 'random':
        mapping = get_random_mapping(N_cg, n_atoms)
        assign_idx = torch.LongTensor( np.array(mapping) ) 

    elif cg_method == 'diff':
        assign_idx = None

    props = get_diffpool_data(N_cg, [traj], n_data=n_data)

    dataset = DiffPoolDataset(props)
    dataset.generate_neighbor_list(cutoff)

    # split train, test index 
    train_index, test_index = train_test_split(list(range(len(dataset))),test_size=0.2)
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
                                      n_rbf=n_rbf, cutoff=cutoff, 
                                      num_conv=dec_nconv, activation=activation)

    prior = DenseCGPrior(n_atoms=n_atoms, n_atom_basis=num_features,
                                      n_rbf=n_rbf, cutoff=cutoff, 
                                      num_conv=enc_nconv, activation=activation)

    atom_mu = nn.Sequential(nn.Linear(num_features, num_features), nn.ReLU(), nn.Linear(num_features, num_features))
    atom_sigma = nn.Sequential(nn.Linear(num_features, num_features), nn.ReLU(), nn.Linear(num_features, num_features))
    
    model = DiffPoolVAE(encoder=encoder,decoder=decoder, pooler=pooler, atom_munet=atom_mu, atom_sigmanet=atom_sigma, prior=prior, det=det).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, 
                                                            factor=0.6, verbose=True, 
                                                            threshold=1e-4,  min_lr=1e-7)
    
    failed = False 

    for epoch in range(params['n_pretrain']):
        model.train()
        mean_train_recon, assign, train_xyz, train_xyz_recon = loop(trainloader, optimizer, device, model, tau_train_sched, epoch, 
                                    beta=0.0, gamma=0.0, eta=eta, kappa=gamma, train=True, looptext='pretrain',
                                    tqdm_flag=tqdm_flag, recon_weight=0.0)

        if np.isnan(mean_train_recon):
            print("NaN encoutered, exiting...")
            failed = True
            break 

        if epoch % 5 == 0:
            map_save_path = os.path.join(working_dir, 'pretrain_map_{}.png'.format(epoch) )
            plot_map(assign[0], props['z'][0].numpy(), map_save_path)

    loss_log = []
    # train     
    for epoch in range(n_epochs):
        model.train()
        mean_train_recon, assign, train_xyz, train_xyz_recon = loop(trainloader, optimizer, device, model, tau_train_sched, epoch, 
                                        beta, gamma, eta,  kappa, train=True, looptext='', tqdm_flag=tqdm_flag)

        mean_val_recon, assign, val_xyz, val_xyz_recon = loop(valloader, optimizer, device, model, tau_val_sched, epoch, 
                                        beta, gamma, eta,  kappa, train=False, looptext='', tqdm_flag=tqdm_flag)

        # # dump recon loss periodically 
        # if epoch % 20 == 0: 
        #     testloader = DataLoader(testset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True)
        #     model.eval()
        #     mean_test_recon, assign, test_xyz, test_xyz_recon = loop(testloader, optimizer, device, model, tau_sched, epoch, beta, 
        #                     gamma, eta, kappa, train=False, looptext='', tqdm_flag=tqdm_flag)

        #     dump_numpy2xyz(test_xyz_recon, props['z'][0].numpy(), os.path.join(working_dir, 'test_recon_{}.xyz'.format(epoch)))

        # validation for each epoch 

        if np.isnan(mean_train_recon):
            print("NaN encoutered, exiting...")
            failed = True
            break 
            
        if epoch % 5 == 0:
            map_save_path = os.path.join(working_dir, 'map_{}.png'.format(epoch) )
            plot_map(assign[0], props['z'][0].numpy(), map_save_path)

        scheduler.step(mean_val_recon)
        loss_log.append([mean_train_recon, mean_val_recon])

        # early stopping 
        if optimizer.param_groups[0]['lr'] <= 1e-7:
            break

    # dump log 
    np.savetxt(os.path.join(working_dir, 'train_val_recon.txt'),  np.array(loss_log))

    dump_numpy2xyz(train_xyz_recon, props['z'][0].numpy(), os.path.join(working_dir, 'train_recon.xyz'))

    # test 
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True)
    mean_test_recon, assign, test_xyz, test_xyz_recon = loop(testloader, optimizer, device, model, tau_val_sched, epoch, beta, 
                                gamma, eta, kappa, train=False, looptext='testing', tqdm_flag=tqdm_flag, tau_min=tau_min)


    # sampling 
    all_rmsds, heavy_rmsds, all_geds, heavy_geds = sample(testloader,  device, model, tau_min, tqdm_flag=True, working_dir=working_dir)

    # dump train recon 
    dump_numpy2xyz(test_xyz_recon, props['z'][0].numpy(), os.path.join(working_dir, 'test_recon.xyz'))

    print("train msd : {:.4f}".format(mean_train_recon))
    print("test msd : {:.4f}".format(mean_test_recon))

    return mean_test_recon, np.array(all_geds).mean(), failed, assign

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-logdir', type=str)
    parser.add_argument('-device', type=int)
    parser.add_argument('-dataset', type=str, default='dipeptide')
    parser.add_argument('-num_features',  type=int, default=512)
    parser.add_argument('-batch_size', type=int,default= 32)
    parser.add_argument('-n_data', type=int, default= 1000)
    parser.add_argument('-N_cg', type=int, default= 3)
    parser.add_argument('-nconv_pool', type=int, default= 7)
    parser.add_argument('-enc_nconv', type=int, default= 4)
    parser.add_argument('-dec_nconv', type=int, default= 3)
    parser.add_argument('-cutoff', type=float, default= 8.0)
    parser.add_argument('-n_rbf', type=int,  default= 7)
    parser.add_argument('-activation', type=str,  default= 'ELU')
    parser.add_argument('-n_epochs', type=int, default= 50)
    parser.add_argument("-n_pretrain", type=int, default= 10)
    parser.add_argument('-tau_rate', type=float, default= 0.004 )
    parser.add_argument('-tau_0', type=float, default= 2.36)
    parser.add_argument('-tau_min', type=float, default= 0.3)
    parser.add_argument('-beta', type=float, default= 0.1)
    parser.add_argument('-gamma', type=float, default= 0.1)
    parser.add_argument('-eta', type=float, default= 0.1)
    parser.add_argument('-kappa', type=float, default= 0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument("--tqdm_flag", action='store_true', default=False)
    parser.add_argument("--det", action='store_true', default=False)
    parser.add_argument("-cg_method", type=str, default='diff')

    params = vars(parser.parse_args())

    run(params)