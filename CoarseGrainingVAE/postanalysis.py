import os 
import argparse 
from data import *
from .cgvae import * 
from .conv import * 
from .datasets import * 
from .utils import * 
from .visualization import xyz_grid_view, rotate_grid
#from plots import *
from .sampling import * 
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

from plots import *
from run_ala import *

def tica(label, filename):

    labels = []
    pdb = PROTEINFILES[label]['pdb_path']
    datafiles = glob.glob(PROTEINFILES[label]['traj_paths'])
    
    traj = load_protein_traj(label)
    alltop = traj.top
    protein_index = traj.top.select("protein")
    
    protientop = traj.top.subset(protein_index)
    
    distances_feat = pyemma.coordinates.featurizer(alltop)
    distances_feat.add_distances(distances_feat.pairs(distances_feat.select_Backbone(),
                                                      excluded_neighbors=2), periodic=False)
    
    true_distances_data = pyemma.coordinates.load(datafiles, features=distances_feat)
    labels += ['backbone atom\ndistances']

    distances_feat = pyemma.coordinates.featurizer(protientop)
    distances_feat.add_distances(
        distances_feat.pairs(distances_feat.select_Backbone(), excluded_neighbors=2), periodic=False)
    smaple_distances_data = pyemma.coordinates.load([filename], features=distances_feat)
    smaple_xyz_data = pyemma.coordinates.load([filename], top=protientop)
    labels += ['backbone atom\ndistances']
    
    tica = pyemma.coordinates.tica(true_distances_data, lag=100)
    tica_output = tica.get_output()
    data_tica = np.concatenate(tica_output)
    
    sample_tica_output = tica.transform([smaple_distances_data])
    gen_tica = np.concatenate(sample_tica_output)
    
    # torsions_feat = pyemma.coordinates.featurizer(PROTEINFILES[label]['pdb_path'])
    # torsions_feat.add_backbone_torsions(cossin=True, periodic=False)
    # true_torsions_data = pyemma.coordinates.load(files, features=torsions_feat)
    # labels = ['backbone\ntorsions']

    # torsions_feat = pyemma.coordinates.featurizer(top)
    # torsions_feat.add_backbone_torsions(cossin=True, periodic=False)
    # sample_torsions_data = pyemma.coordinates.load(sample_files, features=torsions_feat)
    # labels = ['backbone\ntorsions']
    
    return data_tica, gen_tica
     

def test(exp_dir, label, skip=1000):
    
    # Opening JSON file
    with open(os.path.join(exp_dir, 'modelparams.json')) as json_file:
        params = json.load(json_file)
        
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
    dataset_label = params['dataset']
    dir_mp_flag = params['dir_mp']
    cg_mp_flag = params['cg_mp']
    atom_decode_flag = params['atom_decode']
    nevals = params['nevals']
    graph_eval = params['graph_eval']
    mapping = torch.LongTensor(params['mapping'])
    
    n_atoms = mapping.shape[0]
    traj = load_protein_traj(label)
    atomic_nums, protein_index = get_atomNum(traj)
    dataset,mapping = build_split_dataset(traj[::skip], params, mapping=mapping)
    
    
    # initialize model 

    atom_mu = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))
    atom_sigma = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))

    # register encoder 

    decoder = EquivariantDecoder(n_atom_basis=n_basis, n_rbf = n_rbf, 
                                  cutoff=atom_cutoff, num_conv = dec_nconv, activation=activation, cross_flag=True,
                                )

    encoder = EquiEncoder(n_conv=enc_nconv, n_atom_basis=n_basis, 
                                   n_rbf=n_rbf, cutoff=cg_cutoff, activation=activation,
                                    cg_mp=cg_mp_flag, dir_mp=dir_mp_flag)

    # define prior 
    cgPrior = CGprior(n_conv=enc_nconv, n_atom_basis=n_basis, 
                                   n_rbf=n_rbf, cutoff=cg_cutoff, activation=activation,
                                     dir_mp=dir_mp_flag)


    model = CGequiVAE(encoder, decoder, atom_mu, atom_sigma, n_cgs, feature_dim=n_basis, prior_net=cgPrior,
                      ).to(device)

    # load pretrained model 
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'model.pt')))

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=CG_collate, shuffle=False)

    true_xyzs, recon_xyzs, cg_xyzs, mu, sigma, _, _, = get_all_true_reconstructed_structures(loader, 
                                                                                         device,
                                                                                         model,
                                                                                         atomic_nums,
                                                                                         n_cg=n_cgs,
                                                                                       )

    samples = sample(loader, device, model, atomic_nums, n_cgs)
    
    return recon_xyzs, cg_xyzs, true_xyzs, samples, atomic_nums
