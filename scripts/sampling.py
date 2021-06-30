from ase import io, Atoms
import numpy as np
import torch
import mdshare
import pyemma
from tqdm import tqdm
from torch_scatter import scatter_mean
import xyz2mol
from ase import io
from rdkit import Chem 

def ase2mol(atoms, ignoreHH):
    
    atomic_nums = atoms.get_atomic_numbers().tolist()
    xyz = atoms.get_positions()
    mol = xyz2mol.xyz2mol(atomic_nums, xyz, ignoreHH=ignoreHH)
        
    return mol[0]

def infer_smiles_from_geoms(atoms_list):

    inferred_smiles = []

    for atoms in atoms_list:

        try:
            mol = ase2mol(atoms, ignoreHH=True)
            mol = Chem.rdmolops.RemoveHs(mol)
            inferred_smiles.append(canonicalize_smiles(mol))

        except:
            inferred_smiles.append("invalid")
        
    return inferred_smiles


def compute_rmsd(atoms_list, ref_atoms) :
    rmsd = []
    # todo: need to include alignment 
    for atoms in atoms_list:
        test_dxyz = (atoms.get_positions() - ref_atoms.get_positions()).reshape(-1)
        unaligned_test_rmsd = np.sqrt(np.power(test_dxyz, 2).mean())
        rmsd.append(unaligned_test_rmsd)
    
    return np.array(rmsd).mean()

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

    # TODO: included batched sampling 
    model = model.to(device)
    batch = batch_to(batch, device)

    z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs = model.get_inputs(batch)

    sample_xyzs = []
    recon_atoms_list = []

    # get ref Atoms 
    ref_atoms = Atoms(positions=xyz.cpu().numpy(), numbers=atomic_nums) 

    for i in range(n_batch):
        H = torch.normal(mu, sigma).to(cg_xyz.device)
        xyz_recon = model.decoder(cg_xyz, CG_nbr_list, H, H, mapping, num_CGs)
        sample_xyzs.append(xyz_recon.detach().cpu())
        atoms = Atoms(numbers=atomic_nums.ravel(), positions=xyz_recon.detach().cpu().numpy())
        recon_atoms_list.append(atoms)

    # get base smiles 
    ref_mol = ase2mol(ref_atoms, ignoreHH=True)
    ref_smiles = xyz2mol.canonicalize_smiles(ref_mol)
    infer_smiles = infer_smiles_from_geoms(recon_atoms_list)

    # infer recon smiles 
    valid = 0
    for smiles in infer_smiles:
        if smiles == ref_smiles:
            valid += 1
            
    valid_ratio = valid/len(infer_smiles)
    mean_rmsd = compute_rmsd(recon_atoms_list, ref_atoms)

    # compute sample diversity 
    sample_xyzs = torch.cat(sample_xyzs).detach().cpu().numpy()
    
    z = np.concatenate( [atomic_nums] * n_batch )
    atoms = Atoms(numbers=z, positions=sample_xyzs)
    
    # # compute dihedral angle 
    # io.write('tmp.xyz', trajs)

    # pdb = mdshare.fetch('alanine-dipeptide-nowater.pdb', working_directory='data')
    # feat = pyemma.coordinates.featurizer(pdb)
    # feat.add_backbone_torsions() 
    # data = pyemma.coordinates.load('tmp.xyz', features=feat)
    
    
    return atoms, mean_rmsd, valid_ratio


def sample_ensemble(loader, mu, sigma, device, model, atomic_nums, n_cgs, n_sample):
    '''
    conditional sampling based on CG geometry, only works for batch_size = 1
    '''

    sample_xyz_list = []
    n_sample = n_sample
    n_atoms = len(atomic_nums)

    sample_rmsd = []
    sample_valid = []

    for batch in loader:    
        sample_atoms, mean_rmsd, valid_ratio = sample_single(batch, mu, sigma, model, n_sample, atomic_nums, device)
        sample_xyz_list.append(sample_atoms.get_positions())

        # record sampling validity/diversity 
        sample_rmsd.append(mean_rmsd)
        sample_valid.append(valid_ratio)

    sample_xyzs = np.vstack(sample_xyz_list).reshape(-1, n_sample * n_atoms, 3)
    
    return sample_xyzs, sample_rmsd, sample_valid


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