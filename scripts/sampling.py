from ase import io
import numpy as np
import torch
import mdshare
import pyemma
from tqdm import tqdm

from nff.train import batch_to

def batch_to(batch, device):
    gpu_batch = dict()
    for key, val in batch.items():
        gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
    return gpu_batch

def sample_single(batch, mu, sigma, model, n_batch, device):

    model = model.to(device)

    batch = batch_to(batch, device)

    z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs = model.get_inputs(batch)

    sample_xyzs = []
    trajs = []

    for i in range(n_batch):
        S_I = torch.normal(mu, sigma).to(cg_xyz.device)
        xyz_recon = model.decoder(cg_xyz, CG_nbr_list, S_I, mapping, num_CGs)
        sample_xyzs.append(xyz_recon.detach().cpu())
        atoms = Atoms(numbers=atomic_nums.ravel(), positions=xyz_recon.detach().cpu().numpy())
        trajs.append(atoms)

    sample_xyzs = torch.cat(sample_xyzs).detach().cpu().numpy()
    
    z = np.concatenate( [atomic_nums] * n_batch )
    atoms = Atoms(numbers=z, positions=sample_xyzs)
    
    # compute dihedral angle 
    io.write('tmp.xyz', trajs)

    pdb = mdshare.fetch('alanine-dipeptide-nowater.pdb', working_directory='data')
    feat = pyemma.coordinates.featurizer(pdb)
    feat.add_backbone_torsions() 
    data = pyemma.coordinates.load('tmp.xyz', features=feat)
    
    
    return atoms, data

def sample(loader, mu, sigma, device, model, atomic_nums, n_cg):

    model = model.to(device)

    true_xyzs = []
    recon_xyzs = []
    mus = []
    sigmas = []

    tqdm_data = tqdm(loader, position=0, leave=True)    

    for batch in tqdm_data:
        batch = batch_to(batch, device)
        
        z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs = model.get_inputs(batch)
        
        # sample latent vectors
        S_I_list = []
        for i in range(len(num_CGs)):
            S_I_list.append( torch.normal(mu, sigma).to(cg_xyz.device))
            
        S_I = torch.cat(S_I_list)
        xyz_recon = model.decoder(cg_xyz, CG_nbr_list, S_I, mapping, num_CGs)
            
        recon_xyzs.append(xyz_recon.detach().cpu())
        
    recon_xyzs = torch.cat(recon_xyzs).reshape(-1, len(atomic_nums), 3).numpy()
    
    return recon_xyzs