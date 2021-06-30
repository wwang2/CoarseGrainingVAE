from ase import io, Atoms
import numpy as np
import torch
import mdshare
import pyemma
from tqdm import tqdm
from torch_scatter import scatter_mean

def batch_to(batch, device):
    gpu_batch = dict()
    for key, val in batch.items():
        gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
    return gpu_batch

def sample_normal(mu, sigma):
    eps = torch.randn_like(sigma)
    z= eps.mul(sigma).add_(mu)
    return z 

def sample_single(batch, mu, sigma, model, n_batch, atomic_nums, device):

    model = model.to(device)

    batch = batch_to(batch, device)

    z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs = model.get_inputs(batch)

    sample_xyzs = []
    trajs = []

    for i in range(n_batch):
        H = torch.normal(mu, sigma).to(cg_xyz.device)
        xyz_recon = model.decoder(cg_xyz, CG_nbr_list, H, H, mapping, num_CGs)
        sample_xyzs.append(xyz_recon.detach().cpu())
        atoms = Atoms(numbers=atomic_nums.ravel(), positions=xyz_recon.detach().cpu().numpy())
        trajs.append(atoms)

    sample_xyzs = torch.cat(sample_xyzs).detach().cpu().numpy()
    
    z = np.concatenate( [atomic_nums] * n_batch )
    atoms = Atoms(numbers=z, positions=sample_xyzs)
    
    # # compute dihedral angle 
    # io.write('tmp.xyz', trajs)

    # pdb = mdshare.fetch('alanine-dipeptide-nowater.pdb', working_directory='data')
    # feat = pyemma.coordinates.featurizer(pdb)
    # feat.add_backbone_torsions() 
    # data = pyemma.coordinates.load('tmp.xyz', features=feat)
    
    
    return atoms #, data


def sample_ensemble(loader, mu, sigma, device, model, atomic_nums, n_cgs, n_sample):
    '''
    conditional sampling based on CG geometry, only works for batch_size = 1
    '''

    sample_xyz_list = []
    n_sample = n_sample
    n_atoms = len(atomic_nums)
    for batch in loader:    
        sample_atoms = sample_single(batch, mu, sigma, model, n_sample, atomic_nums, device)
        sample_xyz_list.append(sample_atoms.get_positions())

    sample_xyzs = np.vstack(sample_xyz_list).reshape(-1, n_sample * n_atoms, 3)
    
    return sample_xyzs


def sample(loader, mu, sigma, device, model, atomic_nums, n_cgs, atomwise_z=False):

    model = model.to(device)

    true_xyzs = []
    recon_xyzs = []
    ensemble_atoms = []
    mus = []
    sigmas = []

    if atomwise_z:
        n_z = len(atomic_nums)
    else:
        n_z = n_cgs

    tqdm_data = tqdm(loader, position=0, leave=True)    

    for batch in tqdm_data:
        batch = batch_to(batch, device)
        
        z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs = model.get_inputs(batch)
        
        # sample latent vectors
        z_list = []
        for i in range(len(num_CGs)):

           #z_list.append( torch.normal(mu, sigma).to(cg_xyz.device))
           z_list.append(sample_normal(mu, sigma))
            
        z = torch.cat(z_list).to(cg_xyz.device)

        if atomwise_z:
            H = scatter_mean(z, mapping, dim=0)
            h = z 
        else: 
            H = z 
            h = z 

        xyz_recon = model.decoder(cg_xyz, CG_nbr_list, H, z, mapping, num_CGs)
            
        recon_xyzs.append(xyz_recon.detach().cpu())

    recon_xyzs = torch.cat(recon_xyzs).reshape(-1, len(atomic_nums), 3).numpy()
    
    return recon_xyzs