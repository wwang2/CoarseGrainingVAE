import sys
sys.path.append("../scripts/")
sys.path.append("../src/")

import os 
import argparse 
from data import *
from cgvae import * 
from conv import * 
from datasets import * 
from utils import * 
from visualization import xyz_grid_view, rotate_grid
#from plots import *
from sampling import * 
import torch
from torch import nn
from torch.nn import Sequential 
import numpy as np
import copy
from torch_scatter import scatter_mean
from tqdm import tqdm 
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import json
import time
from datetime import timedelta

optim_dict = {'adam':  torch.optim.Adam, 'sgd':  torch.optim.SGD}

def run_cv(params):
    failed = False
    working_dir = params['logdir']
    device  = params['device']
    n_cgs  = params['n_cgs']
    n_basis  = params['n_basis']
    n_rbf  = params['n_rbf']
    atom_cutoff = params['atom_cutoff']
    cg_cutoff = params['cg_cutoff']
    enc_nconv  = params['enc_nconv']
    dec_nconv  = params['dec_nconv']
    batch_size  = params['batch_size']
    beta  = params['beta']
    nsplits = params['nsplits']
    ndata = params['ndata']
    nsamples = params['nsamples']
    nepochs = params['nepochs']
    lr = params['lr']
    activation = params['activation']
    optim = optim_dict[params['optimizer']]
    dataset_label = params['dataset']
    shuffle_flag = params['shuffle']
    dir_mp_flag = params['dir_mp']
    # dec_type = params['dec_type']
    cg_mp_flag = params['cg_mp']
    atom_decode_flag = params['atom_decode']
    nevals = params['nevals']
    graph_eval = params['graph_eval']
    tqdm_flag = params['tqdm_flag']
    n_ensemble = params['n_ensemble']
    det = params['det']
    gamma = params['gamma']
    factor = params['factor']
    patience = params['patience']
    min_lr = 1e-8

    # download data from mdshare 
    mdshare.fetch('pentapeptide-impl-solv.pdb', working_directory='../data')
    mdshare.fetch('pentapeptide-*-500ns-impl-solv.xtc', working_directory='../data')
    mdshare.fetch('alanine-dipeptide-nowater.pdb', working_directory='../data')
    mdshare.fetch('alanine-dipeptide-*-250ns-nowater.xtc', working_directory='../data')

    if dataset_label in PROTEINFILES.keys():
        traj = load_protein_traj(dataset_label)
        atomic_nums, protein_index = get_atomNum(traj)

    else:
        raise ValueError("data label {} not recognized".format(dataset_label))

    # mapping options: alpha carbon, backbone, Girvan-Newman
    
    mapping, frames, cg_coord = get_cg_and_xyz(traj, cg_method=params['cg_method'], n_cgs=params['n_cgs'])
    
    frames = frames[:ndata]
    if cg_coord is not None:
        cg_coord = cg_coord[:ndata]

    dataset = build_dataset(mapping,
                        frames, 
                        atom_cutoff, 
                        cg_cutoff,
                        atomic_nums,
                        cg_traj=cg_coord)
    # get n_atoms 
    n_atoms = atomic_nums.shape[0]
    n_cgs = mapping.max().item() + 1

    dataset.generate_neighbor_list(atom_cutoff=atom_cutoff, cg_cutoff=cg_cutoff, device=device, undirected=True)

    # check CG nbr_list connectivity 

    if not check_CGgraph(dataset):
        print("CG graph not connected")
        return np.NaN, np.NaN, True

    # create subdirectory 
    create_dir(working_dir)
        
    kf = KFold(n_splits=nsplits)
    cv_all_rmsd = []
    cv_heavy_rmsd = []
    cv_sample_rmsd = []
    cv_sample_valid = []
    cv_sample_hh_valid = []

    cv_graph_diff = []
    cv_graph_hh_diff = []

    split_iter = kf.split(list(range(len(dataset))))

    for i, (train_index, test_index) in enumerate(split_iter):

        # start timing 
        start =  time.time()

        split_dir = os.path.join(working_dir, 'fold{}'.format(i)) 
        create_dir(split_dir)

        trainset = get_subset_by_indices(train_index, dataset)
        testset = get_subset_by_indices(test_index, dataset)

        trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=CG_collate, shuffle=shuffle_flag, pin_memory=True)
        testloader = DataLoader(testset, batch_size=batch_size, collate_fn=CG_collate, shuffle=shuffle_flag, pin_memory=True)
        
        # initialize model 
        atom_mu = nn.Sequential(nn.Linear(n_basis, n_basis), nn.Tanh(), nn.Linear(n_basis, n_basis))
        atom_sigma = nn.Sequential(nn.Linear(n_basis, n_basis), nn.Tanh(), nn.Linear(n_basis, n_basis))

        # register encoder 

        decoder = EquivariantDecoder(n_atom_basis=n_basis, n_rbf = n_rbf, 
                                      cutoff=atom_cutoff, num_conv = dec_nconv, activation=activation, 
                                      atomwise_z=atom_decode_flag)

        encoder = EquiEncoder(n_conv=enc_nconv, n_atom_basis=n_basis, 
                                       n_rbf=n_rbf, cutoff=cg_cutoff, activation=activation,
                                        cg_mp=cg_mp_flag, dir_mp=dir_mp_flag, atomwise_z=atom_decode_flag)

        # define prior 
        cgPrior = CGprior(n_conv=enc_nconv, n_atom_basis=n_basis, 
                                       n_rbf=n_rbf, cutoff=cg_cutoff, activation=activation,
                                         dir_mp=dir_mp_flag)

        
        model = CGequiVAE(encoder, decoder, atom_mu, atom_sigma, n_atoms, n_cgs, feature_dim=n_basis, prior_net=cgPrior,
                            atomwise_z=atom_decode_flag, det=det).to(device)
        
        optimizer = optim(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, 
                                                                factor=factor, verbose=True, 
                                                                threshold=5e-5,  min_lr=min_lr)
        
        model.train()

        recon_loss_hist = []
        kl_loss_hist = []
        recon_hist = []

        for epoch in range(nepochs):
            # train
            mean_kl, mean_recon, xyz_train, xyz_train_recon = loop(trainloader, optimizer, device,
                                                       model, beta, epoch, 
                                                       train=True,
                                                        gamma=gamma,
                                                        looptext='Ncg {} Fold {}'.format(n_cgs, i),
                                                        tqdm_flag=tqdm_flag)
            
            scheduler.step(mean_recon)

            recon_hist.append(xyz_train_recon.detach().cpu().numpy().reshape(-1, n_atoms, 3))
            kl_loss_hist.append(mean_kl)
            recon_loss_hist.append(mean_recon)

            # check NaN
            if np.isnan(mean_recon):
                print("NaN encoutered, exiting...")
                break 

            if optimizer.param_groups[0]['lr'] <= min_lr:
                print('converged')
                break

            del mean_kl, mean_recon, xyz_train, xyz_train_recon

        # dump model hyperparams 
        with open(os.path.join(split_dir, 'modelparams.json'), "w") as outfile: 
            json.dump(params, outfile)

        # dump training curve 
        np.savetxt(os.path.join(split_dir, 'train_loss_curve.txt'), np.vstack((np.array([recon_loss_hist]), np.array([kl_loss_hist]))))

        # dump learning trajectory 
        recon_hist = np.concatenate(recon_hist)
        dump_numpy2xyz(recon_hist, atomic_nums, os.path.join(split_dir, 'recon_hist.xyz'))
            
        # save sampled geometries 
        trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=CG_collate, shuffle=True)
        train_true_xyzs, train_recon_xyzs, train_cg_xyzs, mu, sigma = get_all_true_reconstructed_structures(trainloader, 
                                                                                             device,
                                                                                             model,
                                                                                             atomic_nums,
                                                                                             n_cg=n_cgs,
                                                                                             atomwise_z=atom_decode_flag,
                                                                                             tqdm_flag=tqdm_flag)

        # sample geometries 
        train_samples = sample(trainloader, mu, sigma, device, model, atomic_nums, n_cgs, atomwise_z=atom_decode_flag)

        cg_types = np.array([6] * n_cgs)

        dump_numpy2xyz(train_samples[:nsamples], atomic_nums, os.path.join(split_dir, 'train_samples.xyz'))
        dump_numpy2xyz(train_true_xyzs[:nsamples], atomic_nums, os.path.join(split_dir, 'train_original.xyz'))
        dump_numpy2xyz(train_recon_xyzs[:nsamples], atomic_nums, os.path.join(split_dir, 'train_recon.xyz'))
        dump_numpy2xyz(train_cg_xyzs[:nsamples], cg_types, os.path.join(split_dir, 'train_cg.xyz'))

        testloader = DataLoader(testset, batch_size=batch_size, collate_fn=CG_collate, shuffle=True)
        test_true_xyzs, test_recon_xyzs, test_cg_xyzs, mu, sigma = get_all_true_reconstructed_structures(testloader, 
                                                                                             device,
                                                                                             model,
                                                                                             atomic_nums,
                                                                                             n_cg=n_cgs,
                                                                                             atomwise_z=atom_decode_flag,
                                                                                             tqdm_flag=tqdm_flag)

        # sample geometries 
        test_samples = sample(testloader, mu, sigma, device, model, atomic_nums, n_cgs, atomwise_z=atom_decode_flag)

        dump_numpy2xyz(test_samples[:nsamples], atomic_nums, os.path.join(split_dir, 'test_samples.xyz'))
        dump_numpy2xyz(test_true_xyzs[:nsamples], atomic_nums, os.path.join(split_dir, 'test_original.xyz'))
        dump_numpy2xyz(test_recon_xyzs[:nsamples], atomic_nums, os.path.join(split_dir, 'test_recon.xyz'))
        dump_numpy2xyz(test_cg_xyzs[:nsamples], cg_types, os.path.join(split_dir, 'test_cg.xyz'))

        # compute loss and metrics 
        heavy_filter = atomic_nums != 1.
        test_all_dxyz = (test_recon_xyzs - test_true_xyzs).reshape(-1)
        test_heavy_dxyz = (test_recon_xyzs - test_true_xyzs)[:, heavy_filter, :].reshape(-1)

        unaligned_test_all_rmsd = np.sqrt(np.power(test_all_dxyz, 2).mean())
        unaligned_test_heavy_rmsd = np.sqrt(np.power(test_heavy_dxyz, 2).mean())

        # dump test rmsd 
        np.savetxt(os.path.join(split_dir, 'test_all_rmsd{:.4f}.txt'.format(unaligned_test_all_rmsd)), np.array([unaligned_test_all_rmsd]))
        np.savetxt(os.path.join(split_dir, 'test_heavy_rmsd{:.4f}.txt'.format(unaligned_test_heavy_rmsd)), np.array([unaligned_test_heavy_rmsd]))

        # reconsturction loss 
        cv_all_rmsd.append(unaligned_test_all_rmsd)
        cv_heavy_rmsd.append(unaligned_test_heavy_rmsd)

        # save model 
        model = model.to('cpu')
        torch.save(model.state_dict(), os.path.join(split_dir, 'model.pt'))

        ##### generate rotating movies for visualization #####
        n_w = 3
        n_h = 3
        n_frames = n_w * n_h

 
        sampleloader = DataLoader(testset, batch_size=1, collate_fn=CG_collate, shuffle=False)

        sample_xyzs, data_xyzs, cg_xyzs, recon_xyzs, all_rmsds, all_heavy_rmsds, sample_valid, sample_hh_valid, sample_graph_val_ratio_list, sample_graph_hh_val_ratio_list = sample_ensemble(sampleloader, mu, sigma, device, 
                                                                                model, atomic_nums, 
                                                                                n_cgs, n_sample=n_ensemble,
                                                                                graph_eval=graph_eval)

        if graph_eval:
            sample_valid = np.array(sample_valid).mean()
            sample_hh_valid = np.array(sample_hh_valid).mean()

            if all_rmsds is not None:
                mean_all_rmsd = np.array(all_rmsds)[:, 0].mean()
            else:
                mean_all_rmsd = None

            if all_heavy_rmsds is not None:
                mean_heavy_rmsd = np.array(all_heavy_rmsds)[:, 1].mean()
            else:
                mean_heavy_rmsd = None

            cv_sample_valid.append(sample_valid)
            cv_sample_hh_valid.append(sample_hh_valid)
            
            mean_graph_diff = np.array(sample_graph_val_ratio_list).mean()
            mean_graph_hh_diff = np.array(sample_graph_hh_val_ratio_list).mean()
            cv_graph_diff.append(mean_graph_diff)
            cv_graph_hh_diff.append(mean_graph_hh_diff)

            print("Sample Quality Stats: ")
            print("sample RMSD (heavy atoms) : {}".format(mean_heavy_rmsd))
            print("sample RMSD (all atoms) : {}".format(mean_all_rmsd))
            print("sample validity (heavy atoms): {}".format(sample_valid))
            print("sample validity (all atoms): {}".format(sample_hh_valid))
            print("sample graph difference ratio (heavy atoms): {}".format(mean_graph_diff))
            print("sample graph difference ratio (all atoms): {}".format(mean_graph_hh_diff))


        # compute maxium dimension
        ref_xyz = data_xyzs[0]
        ref_xyz = ref_xyz - ref_xyz.mean(0)
        geom_max_dim = (ref_xyz.max() - ref_xyz.min()) * 1.45

        # loop over all the ensembles and dump individual samples
        for sample_id in range(n_ensemble): 
            snapshot = sample_xyzs.reshape(-1, n_ensemble, n_atoms, 3)[:, sample_id, :, :]
            ensemble_atoms = xyz_grid_view(torch.Tensor(snapshot).reshape(-1, 3),
                            atomic_nums, [n_atoms] * n_frames, n_w, n_h, grid_dim=geom_max_dim, grid_scale=1.4)

            rotate_ensemble = rotate_grid(ensemble_atoms, n_frames, axis='y')

            io.write(os.path.join(split_dir, 'rotate_test_ensemble_{}.xyz'.format(sample_id)), rotate_ensemble)

        data_atoms = xyz_grid_view(torch.Tensor(data_xyzs).reshape(-1, 3),
                      atomic_nums, [n_atoms] * n_frames, n_w, n_h, grid_dim=geom_max_dim, grid_scale=1.4)

        recon_atoms = xyz_grid_view(torch.Tensor(recon_xyzs).reshape(-1, 3),
                      atomic_nums, [n_atoms] * n_frames, n_w, n_h, grid_dim=geom_max_dim, grid_scale=1.4)

        cg_atoms = xyz_grid_view(torch.Tensor(cg_xyzs).reshape(-1, 3),
                      np.ones(n_cgs) * 6, [n_cgs] * n_frames, n_w, n_h, grid_dim=geom_max_dim, grid_scale=1.4)


        rotate_data = rotate_grid(data_atoms, n_frames, axis='y')
        rotate_recon = rotate_grid(recon_atoms, n_frames, axis='y')
        rotate_cg = rotate_grid(cg_atoms, n_frames, axis='y')
        #rotate_ensemble = rotate_grid(ensemble_atoms, n_frames, axis='y')

        io.write(os.path.join(split_dir, 'rotate_test_data.xyz'), rotate_data)
        io.write(os.path.join(split_dir, 'rotate_test_recon.xyz'), rotate_recon)
        io.write(os.path.join(split_dir, 'rotate_test_cg.xyz'), rotate_cg)
        #io.write(os.path.join(split_dir, 'rotate_test_ensemble.xyz'), rotate_ensemble)

        # dump rmsd distributions 
        if graph_eval:
            if all_rmsds is not None:
                np.savetxt(os.path.join(split_dir, 'valid_all_rmsds.txt'), np.array(all_rmsds))
            if all_heavy_rmsds is not None:
                np.savetxt(os.path.join(split_dir, 'valid_heavy_rmsds.txt'), np.array(all_heavy_rmsds))

        end = time.time()

        temp = end-start
        hours = temp//3600
        temp = temp - 3600*hours
        minutes = temp//60
        seconds = temp - 60*minutes
        format_time = '%d:%d:%d' %(hours,minutes,seconds)

        np.savetxt(os.path.join(split_dir, '{}.txt'.format(format_time)), np.ones(10))

        print("time elapsed: {}".format(format_time))

        ########################################################


    # save test score 
    np.savetxt(os.path.join(working_dir, 'cv_all_rmsd.txt'), np.array(cv_all_rmsd))
    np.savetxt(os.path.join(working_dir, 'cv_heavy_rmsd.txt'), np.array(cv_heavy_rmsd))

    if graph_eval:
        np.savetxt(os.path.join(working_dir, 'cv_valid.txt'), np.array(cv_sample_valid))
        np.savetxt(os.path.join(working_dir, 'cv_hh_valid.txt'), np.array(cv_sample_hh_valid))
        np.savetxt(os.path.join(working_dir, 'cv_graph_diff.txt'), np.array(cv_graph_diff))
        np.savetxt(os.path.join(working_dir, 'cv_hh_graph_diff.txt'), np.array(cv_graph_hh_diff))

    if failed:
        with open(os.path.join(split_dir, 'FAILED.txt'), "w") as text_file:
            print("TRAINING FAILED", file=text_file)

    all_rmsd_mean = np.array(cv_all_rmsd).mean()
    all_rmsd_std = np.array(cv_all_rmsd).std()

    if graph_eval:
        all_graph_diff_mean = np.array(cv_graph_hh_diff).mean()
        all_graph_diff_std = np.array(cv_graph_hh_diff).std() 
    else:
        all_graph_diff_mean = None 
        all_graph_diff_std = None

    return all_rmsd_mean, all_rmsd_std, all_graph_diff_mean, all_graph_diff_std, failed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-logdir", type=str)
    parser.add_argument("-device", type=int)
    parser.add_argument("-n_cgs", type=int)
    parser.add_argument("-lr", type=float, default=2e-4)
    parser.add_argument("-dataset", type=str, default='dipeptide')
    parser.add_argument("-n_basis", type=int, default=256)
    parser.add_argument("-n_rbf", type=int, default=10)
    parser.add_argument("-activation", type=str, default='swish')
    parser.add_argument("-cg_method", type=str, default='minimal')
    parser.add_argument("-atom_cutoff", type=float, default=4.0)
    parser.add_argument("-optimizer", type=str, default='adam')
    parser.add_argument("-cg_cutoff", type=float, default=4.0)
    parser.add_argument("-enc_nconv", type=int, default=4)
    parser.add_argument("-dec_nconv", type=int, default=4)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-nepochs", type=int, default=2)
    parser.add_argument("-ndata", type=int, default=200)
    parser.add_argument("-nsamples", type=int, default=200)
    parser.add_argument("-n_ensemble", type=int, default=16)
    parser.add_argument("-nevals", type=int, default=36)
    parser.add_argument("-beta", type=float, default=0.001)
    parser.add_argument("-gamma", type=float, default=0.01)
    parser.add_argument("-nsplits", type=int, default=5)
    parser.add_argument("-patience", type=int, default=5)
    parser.add_argument("-factor", type=int, default=0.6)
    parser.add_argument("--dec_type", type=str, default='EquivariantDecoder')
    parser.add_argument("--graph_eval", action='store_true', default=False)
    parser.add_argument("--randommap", action='store_true', default=False)
    parser.add_argument("--shuffle", action='store_true', default=False)
    parser.add_argument("--cg_mp", action='store_true', default=False)
    parser.add_argument("--dir_mp", action='store_true', default=False)
    parser.add_argument("--atom_decode", action='store_true', default=False)
    parser.add_argument("--tqdm_flag", action='store_true', default=False)
    parser.add_argument("--det", action='store_true', default=False)

    params = vars(parser.parse_args())
    
    run_cv(params)