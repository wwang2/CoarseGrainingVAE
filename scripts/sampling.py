from ase import io, Atoms
import numpy as np
import torch
import mdshare
import pyemma
from tqdm import tqdm
from torch_scatter import scatter_mean
#import xyz2mol
from ase import io
#from rdkit import Chem 

COVCUTOFFTABLE = {1: 0.23,
                 2: 0.93,
                 3: 0.68,
                 4: 0.35,
                 5: 0.83,
                 6: 0.68,
                 7: 0.68,
                 8: 0.68,
                 9: 0.64,
                 10: 1.12,
                 11: 0.97,
                 12: 1.1,
                 13: 1.35,
                 14: 1.2,
                 15: 0.75,
                 16: 1.02,
                 17: 0.99,
                 18: 1.57,
                 19: 1.33,
                 20: 0.99,
                 21: 1.44,
                 22: 1.47,
                 23: 1.33,
                 24: 1.35,
                 25: 1.35,
                 26: 1.34,
                 27: 1.33,
                 28: 1.5,
                 29: 1.52,
                 30: 1.45,
                 31: 1.22,
                 32: 1.17,
                 33: 1.21,
                 34: 1.22,
                 35: 1.21,
                 36: 1.91,
                 37: 1.47,
                 38: 1.12,
                 39: 1.78,
                 40: 1.56,
                 41: 1.48,
                 42: 1.47,
                 43: 1.35,
                 44: 1.4,
                 45: 1.45,
                 46: 1.5,
                 47: 1.59,
                 48: 1.69,
                 49: 1.63,
                 50: 1.46,
                 51: 1.46,
                 52: 1.47,
                 53: 1.4,
                 54: 1.98,
                 55: 1.67,
                 56: 1.34,
                 57: 1.87,
                 58: 1.83,
                 59: 1.82,
                 60: 1.81,
                 61: 1.8,
                 62: 1.8,
                 63: 1.99,
                 64: 1.79,
                 65: 1.76,
                 66: 1.75,
                 67: 1.74,
                 68: 1.73,
                 69: 1.72,
                 70: 1.94,
                 71: 1.72,
                 72: 1.57,
                 73: 1.43,
                 74: 1.37,
                 75: 1.35,
                 76: 1.37,
                 77: 1.32,
                 78: 1.5,
                 79: 1.5,
                 80: 1.7,
                 81: 1.55,
                 82: 1.54,
                 83: 1.54,
                 84: 1.68,
                 85: 1.7,
                 86: 2.4,
                 87: 2.0,
                 88: 1.9,
                 89: 1.88,
                 90: 1.79,
                 91: 1.61,
                 92: 1.58,
                 93: 1.55,
                 94: 1.53,
                 95: 1.51,
                 96: 1.5,
                 97: 1.5,
                 98: 1.5,
                 99: 1.5,
                 100: 1.5,
                 101: 1.5,
                 102: 1.5,
                 103: 1.5,
                 104: 1.57,
                 105: 1.49,
                 106: 1.43,
                 107: 1.41}

def compute_bond_cutoff(atoms, scale=1.3):
    atomic_nums = atoms.get_atomic_numbers()
    vdw_array = torch.Tensor( [COVCUTOFFTABLE[int(el)] for el in atomic_nums] )
    
    cutoff_array = (vdw_array[None, :] + vdw_array[:, None]) * scale 
    
    return cutoff_array

def compute_distance_mat(atoms, device='cpu'):
    
    xyz = torch.Tensor( atoms.get_positions() ).to(device)
    dist = (xyz[:, None, :] - xyz[None, :, :]).pow(2).sum(-1).sqrt()
    
    return dist

def dropH(atoms):
    
    positions = atoms.get_positions()
    atomic_nums = atoms.get_atomic_numbers()
    mask = atomic_nums != 1
    
    heavy_pos = positions[mask]
    heavy_nums = atomic_nums[mask]
    
    new_atoms = Atoms(numbers=heavy_nums, positions=heavy_pos)
    
    return new_atoms

def compare_graph(ref_atoms, atoms, scale):

    ref_bonds = get_bond_graphs(ref_atoms, scale=scale)

    bonds = get_bond_graphs(atoms, scale=scale)

    diff = (bonds != ref_bonds).sum().item()
    
    return diff

def get_bond_graphs(atoms, device='cpu', scale=1.3):
    dist = compute_distance_mat(atoms, device=device)
    cutoff = compute_bond_cutoff(atoms, scale=scale)
    bond_mat = (dist < cutoff.to(device))
    bond_mat[np.diag_indices(len(atoms))] = 0
    
    del dist, cutoff

    return bond_mat.to(torch.long).to('cpu')

# compare graphs 

def count_valid_graphs(ref_atoms, atoms_list, heavy_only=True, scale=1.3):
    
    if heavy_only:
        ref_atoms = dropH(ref_atoms)
    
    valid_ids = []
    graph_diff_ratio_list = []

    for idx, atoms in enumerate(atoms_list):
        
        if heavy_only:
            atoms = dropH(atoms)

        if compare_graph(ref_atoms, atoms, scale=scale) == 0:
            valid_ids.append(idx)

        gen_graph = get_bond_graphs(atoms, scale=scale)
        ref_graph = get_bond_graphs(ref_atoms, scale=scale )

        graph_diff_ratio = (ref_graph - gen_graph).sum().abs() / ref_graph.sum()
        graph_diff_ratio_list.append(graph_diff_ratio.item())

    valid_ratio = len(valid_ids)/len(atoms_list)
    
    return valid_ids, valid_ratio, graph_diff_ratio_list


def ase2mol(atoms, ignoreHH):
    
    atomic_nums = atoms.get_atomic_numbers().tolist()
    xyz = atoms.get_positions()
    mol = xyz2mol.xyz2mol(atomic_nums, xyz, ignoreHH=ignoreHH)
        
    return mol[0]

def infer_smiles_from_geoms(atoms_list, ignoreHH=True):

    inferred_smiles = []

    for atoms in atoms_list:
        try:
            mol = ase2mol(atoms, ignoreHH=ignoreHH)
            mol = Chem.rdmolops.RemoveHs(mol)
            inferred_smiles.append(xyz2mol.canonicalize_smiles(mol))
        except:
            inferred_smiles.append("invalid")
        
    return inferred_smiles


def compute_rmsd(atoms_list, ref_atoms, valid_ids) :
    rmsd_array = []
    # todo: need to include alignment 
    for i, atoms in enumerate(atoms_list):
        z = atoms.get_atomic_numbers() 

        heavy_filter = z != 1. 

        aa_test_dxyz = (atoms.get_positions() - ref_atoms.get_positions()).reshape(-1)
        aa_rmsd = np.sqrt(np.power(aa_test_dxyz, 2).mean())

        heavy_test_dxyz = (atoms.get_positions()[heavy_filter] - ref_atoms.get_positions()[heavy_filter]).reshape(-1)
        heavy_rmsd = np.sqrt(np.power(heavy_test_dxyz, 2).mean())
        
        if i in valid_ids:
            rmsd_array.append([aa_rmsd, heavy_rmsd])
    if len(valid_ids) != 0:
        return np.array(rmsd_array)
    else:
        return None

def batch_to(batch, device):
    gpu_batch = dict()
    for key, val in batch.items():
        gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
    return gpu_batch

def sample_normal(mu, sigma):
    eps = torch.randn_like(sigma)
    z= eps.mul(sigma).add_(mu)
    return z 

def sample_single(batch, model, n_batch, atomic_nums, device, graph_eval=True, reflection=False):

    # TODO: included batched sampling 
    model = model.to(device)

    if reflection: 
        xyz = batch['nxyz'][:,1:]
        xyz[:, 1] *= -1 # reflect around x-z plane
        cgxyz = batch['CG_nxyz'][:,1:]
        cgxyz[:, 1] *= -1 

    batch = batch_to(batch, device)

    z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs = model.get_inputs(batch)

    # compute cg prior 
    H_prior_mu, H_prior_sigma = model.prior_net(cg_z, cg_xyz, CG_nbr_list)

    sample_xyzs = []
    recon_atoms_list = []

    # get ref Atoms 
    ref_atoms = Atoms(positions=xyz.cpu().numpy(), numbers=atomic_nums) 

    for i in range(n_batch):
        H = sample_normal(H_prior_mu, H_prior_sigma)
        H = H.to(device)
        xyz_decode = model.decoder(cg_xyz, CG_nbr_list, H, H, mapping, num_CGs)
        sample_xyzs.append(xyz_decode.detach().cpu())
        atoms = Atoms(numbers=atomic_nums.ravel(), positions=xyz_decode.detach().cpu().numpy())
        recon_atoms_list.append(atoms)

        del xyz_decode, H 

    # save origina ldata 
    data_atoms = Atoms(numbers=atomic_nums.ravel(), positions=xyz.detach().cpu().numpy())

    # save cg atoms 
    cg_atoms = Atoms(numbers=[6] * len(cg_xyz), positions=cg_xyz.detach().cpu().numpy())

    # save reconstructed atoms 
    S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon = model(batch)
    recon_atoms = Atoms(numbers=atomic_nums.ravel(), positions=xyz_recon.detach().cpu().numpy())

    # compute sample diversity 
    sample_xyzs = torch.cat(sample_xyzs).detach().cpu().numpy()
    
    z = np.concatenate( [atomic_nums] * n_batch )
    ensemble_atoms = Atoms(numbers=z, positions=sample_xyzs)

    # evaluate sample qualities 

    del S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon

    if graph_eval:
        all_rmsds, heavy_rmsds, valid_ratio, valid_allatom_ratio, graph_val_ratio, graph_allatom_val_ratio = eval_sample_qualities(ref_atoms, recon_atoms_list)
        return ensemble_atoms, data_atoms, recon_atoms, cg_atoms, all_rmsds, heavy_rmsds, valid_ratio, valid_allatom_ratio, graph_val_ratio, graph_allatom_val_ratio
    
    else:
        return ensemble_atoms, data_atoms, recon_atoms, cg_atoms, None, None, None, None, None, None

def count_valid_smiles(true_smiles, inferred_smiles):

    valid_ids = []
    for idx, smiles in enumerate(inferred_smiles):
        if smiles == true_smiles:
            valid_ids.append(idx)  
    valid_ratio = len(valid_ids)/len(inferred_smiles)

    return valid_ids, valid_ratio


def eval_sample_qualities(ref_atoms, atoms_list, scale=1.3): 

    valid_ids, valid_ratio, graph_val_ratio = count_valid_graphs(ref_atoms, atoms_list, heavy_only=True, scale=scale)
    valid_allatom_ids, valid_allatom_ratio, graph_allatom_val_ratio = count_valid_graphs(ref_atoms, atoms_list, heavy_only=False, scale=scale)

    # keep track of heavy and all-atom rmsds separately
    heavy_rmsds = compute_rmsd(atoms_list, ref_atoms, valid_ids) # rmsds for valid heavy atom graphs 
    all_rmsds = compute_rmsd(atoms_list, ref_atoms, valid_allatom_ids) # rmsds for valid allatom graphs 

    return all_rmsds, heavy_rmsds, valid_ratio, valid_allatom_ratio, graph_val_ratio, graph_allatom_val_ratio

def sample_ensemble(loader, device, model, atomic_nums, n_cgs, n_sample, reflection, graph_eval=True):
    '''
    conditional sampling based on CG geometry, only works for batch_size = 1
    '''

    sample_xyz_list = []
    recon_xyz_list = []
    cg_xyz_list = []
    data_xyz_list = []

    n_sample = n_sample
    n_atoms = len(atomic_nums)

    sample_all_rmsd = []
    sample_heavy_rmsd = []

    sample_valid = []
    sample_allatom_valid = []

    sample_graph_val_ratio_list = []
    sample_graph_allatom_val_ratio_list = []

    for batch in loader:    
        sample_atoms, data_atoms, recon_atoms, cg_atoms, all_rmsds, \
         heavy_rmsds, valid_ratio, valid_allatom_ratio, graph_val_ratio, \
         graph_allatom_val_ratio = sample_single(batch, model, n_sample, atomic_nums, device, graph_eval=graph_eval, reflection=reflection)

        sample_xyz_list.append(sample_atoms.get_positions())
        data_xyz_list.append(data_atoms.get_positions())
        cg_xyz_list.append(cg_atoms.get_positions())
        recon_xyz_list.append(recon_atoms.get_positions())

        # record sampling validity/diversity 
        if all_rmsds is not None:
            sample_all_rmsd.append(all_rmsds)

        if heavy_rmsds is not None:
            sample_heavy_rmsd.append(heavy_rmsds)

        sample_valid.append(valid_ratio)
        sample_allatom_valid.append(valid_allatom_ratio)

        sample_graph_val_ratio_list.append(graph_val_ratio)
        sample_graph_allatom_val_ratio_list.append(graph_allatom_val_ratio)

    sample_xyzs = np.vstack(sample_xyz_list).reshape(-1, n_sample * n_atoms, 3)
    data_xyzs = np.vstack(data_xyz_list).reshape(-1, n_atoms, 3)
    cg_xyzs = np.vstack(cg_xyz_list).reshape(-1, n_cgs, 3)
    recon_xyzs = np.vstack(recon_xyz_list).reshape(-1, n_atoms, 3)

    if graph_eval:

        if len(sample_heavy_rmsd) != 0:
            all_heavy_rmsds = np.concatenate(sample_heavy_rmsd) # list of valid structure rmsds 
        else:
            all_heavy_rmsds = None

        if len(sample_all_rmsd) != 0:
            all_rmsds = np.concatenate(sample_all_rmsd) # list of valid structure rmsds 
        else:
            all_rmsds = None

        return sample_xyzs, data_xyzs, cg_xyzs, recon_xyzs, all_rmsds, all_heavy_rmsds, sample_valid, sample_allatom_valid, sample_graph_val_ratio_list, sample_graph_allatom_val_ratio_list
    else:
        return sample_xyzs, data_xyzs, cg_xyzs, recon_xyzs, None, None, None, None, None, None 


def sample(loader, device, model, atomic_nums, n_cgs, tqdm_flag=False):

    model = model.to(device)

    true_xyzs = []
    recon_xyzs = []
    ensemble_atoms = []

    n_z = n_cgs

    if tqdm_flag:
        loader = tqdm(loader, position=0, leave=True)    

    for batch in loader:
        batch = batch_to(batch, device)

        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs = model.get_inputs(batch)
        atomic_nums = z.detach().cpu().numpy()
        # compute cg prior 
        H_prior_mu, H_prior_sigma = model.prior_net(cg_z, cg_xyz, CG_nbr_list)

        # sample latent vectors
        z = sample_normal(H_prior_mu, H_prior_sigma)

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