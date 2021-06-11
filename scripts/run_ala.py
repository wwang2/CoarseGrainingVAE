import sys
sys.path.append("../scripts/")
sys.path.append("../src/")
sys.path.append("/home/wwj/Repo/packages/NeuralForceField/")

import os 
import argparse 
from data import *
from model import * 
from conv import * 
from datasets import * 
from utils import * 
from plots import *
from sampling import * 
import torch
from torch import nn
from torch.nn import Sequential 
import numpy as np
import copy
from nff.utils.scatter import scatter_add
from torch_scatter import scatter_mean
from nff.train import batch_to
from tqdm import tqdm 
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int)
parser.add_argument("-n_atoms", type=int)
parser.add_argument("-n_cgs", type=int)
parser.add_argument("-n_basis", type=int, default=256)
parser.add_argument("-n_rbf", type=int, default=10)
parser.add_argument("-cutoff", type=float, default=4.0)
parser.add_argument("-enc_nconv", type=int, default=4)
parser.add_argument("-dec_nconv", type=int, default=4)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-nepochs", type=int, default=2)
parser.add_argument("-ndata", type=int, default=200)
parser.add_argument("-nsamples", type=int, default=200)
parser.add_argument("-beta", type=float, default=0.001)
parser.add_argument("-nsplits", type=int, default=5)
parser.add_argument("--randommap", action='store_true', default=False)
params = vars(parser.parse_args())

working_dir = params['logdir']
device  = params['device']
n_atoms = params['n_atoms']
n_cgs  = params['n_cgs']
n_basis  = params['n_basis']
n_rbf  = params['n_rbf']
cutoff = params['cutoff']
enc_nconv  = params['enc_nconv']
dec_nconv  = params['dec_nconv']
batch_size  = params['batch_size']
beta  = params['beta']
nsplits = params['nsplits']
ndata = params['ndata']
nsamples = params['nsamples']
nepochs = params['nepochs']

# generate mapping 
if params['randommap']:
    mapping = get_random_mapping(n_cgs, n_atoms)
else:
    mapping = get_mapping('dipeptide', 2.0, n_atoms, n_cgs)

# combine directory 
atomic_nums, dataset = get_alanine_dipeptide_dataset(cutoff,
                                                     label='dipeptide',
                                                     mapping=mapping,
                                                     n_frames=ndata, 
                                                     n_cg=n_cgs)

# create subdirectory 
create_dir(working_dir)
    
kf = KFold(n_splits=nsplits)
cv_rmsd = []
split_iter = kf.split(list(range(len(dataset))))\

for i, (train_index, test_index) in enumerate(split_iter):

    split_dir = os.path.join(working_dir, 'fold{}'.format(i)) 
    create_dir(split_dir)

    trainset = get_subset_by_indices(train_index, dataset)
    testset = get_subset_by_indices(test_index, dataset)

    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=CG_collate, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=CG_collate, shuffle=True)
    
    # initialize model 
    atom_mu = nn.Sequential(nn.Linear(n_basis, n_basis), nn.Tanh(), nn.Linear(n_basis, n_basis))
    atom_sigma = nn.Sequential(nn.Linear(n_basis, n_basis), nn.Tanh(), nn.Linear(n_basis, n_basis))

    cgconv = EquivariantConv(n_atom_basis=n_basis, n_rbf = n_rbf, 
                             cutoff=cutoff, num_conv = dec_nconv)

    encoder = CGequivariantEncoder(n_conv=enc_nconv, n_atom_basis=n_basis, 
                                   n_rbf=n_rbf, cutoff=cutoff)

    model = CGequiVAE(encoder, cgconv, atom_mu, atom_sigma, n_atoms, n_cgs).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5, factor=0.5)
    
    model.train()
    for epoch in range(nepochs):
        # train
        mean_kl, mean_recon, xyz_train, xyz_train_recon = loop(trainloader, optimizer, device,
                                                   model, beta, epoch, train=True,
                                                    looptext='Ncg {} Fold {}'.format(n_cgs, i))
        # check NaN
        if np.isnan(mean_recon):
            break 

        scheduler.step(mean_recon)
        
    # save sampled geometries 
    trainloader = DataLoader(trainset, batch_size=128, collate_fn=CG_collate, shuffle=True)
    train_true_xyzs, train_recon_xyzs, mu, sigma = get_all_true_reconstructed_structures(trainloader, 
                                                                                         device,
                                                                                         model,
                                                                                         atomic_nums,
                                                                                         n_cg=n_cgs)

    # sample geometries 
    train_samples = sample(trainloader, mu, sigma, device, model, atomic_nums, n_cgs)

    dump_numpy2xyz(train_samples[:nsamples], atomic_nums, os.path.join(split_dir, 'train_samples.xyz'))
    dump_numpy2xyz(train_true_xyzs[:nsamples], atomic_nums, os.path.join(split_dir, 'train_original.xyz'))
    dump_numpy2xyz(train_recon_xyzs[:nsamples], atomic_nums, os.path.join(split_dir, 'train_recon.xyz'))


    testloader = DataLoader(testset, batch_size=128, collate_fn=CG_collate, shuffle=True)
    test_true_xyzs, test_recon_xyzs, mu, sigma = get_all_true_reconstructed_structures(testloader, 
                                                                                         device,
                                                                                         model,
                                                                                         atomic_nums,
                                                                                         n_cg=n_cgs)

    # sample geometries 
    test_samples = sample(trainloader, mu, sigma, device, model, atomic_nums, n_cgs)

    dump_numpy2xyz(test_samples[:nsamples], atomic_nums, os.path.join(split_dir, 'test_samples.xyz'))
    dump_numpy2xyz(test_true_xyzs[:nsamples], atomic_nums, os.path.join(split_dir, 'test_original.xyz'))
    dump_numpy2xyz(test_recon_xyzs[:nsamples], atomic_nums, os.path.join(split_dir, 'test_recon.xyz'))

    # compute loss and metrics 
    test_dxyz = (test_recon_xyzs - test_true_xyzs).reshape(-1)
    unaligned_test_rmsd = np.sqrt(np.power(test_recon_xyzs, 2)).mean()

    # reconsturction loss 
    cv_rmsd.append(unaligned_test_rmsd)

# save CV score 
np.savetxt(os.path.join(working_dir, 'cv_rmsd.txt'), np.array(cv_rmsd))

