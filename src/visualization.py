
from ase import Atoms 
import torch
import numpy as np 
from copy import copy, deepcopy

def xyz_grid_view(xyzs, atomic_nums, num_atoms, n_w, n_h, grid_scale=1.5, grid_dim=None):
    
    assert len(num_atoms) == n_w * n_h

    ref_xyz = xyzs[:num_atoms[0]]
    ref_xyz = ref_xyz - ref_xyz.mean(0)[None, ...]
    geom_max_dim = ref_xyz.max() - ref_xyz.min()

    if grid_dim is None:
        grid_dim = geom_max_dim * grid_scale

    x_basis = torch.Tensor([grid_dim, 0.0, 0.0])
    y_basis = torch.Tensor([0.0, grid_dim, 0.0])

    grid_centers = []

    for i in range(n_w):
        for j in range(n_h): 
            grid_centers.append(x_basis * i + y_basis * j)

    grid_centers = torch.stack(grid_centers)

    xyzs = torch.split(xyzs.detach().cpu()[:np.sum(num_atoms)], num_atoms)
    xyzs_list = []

    for grid, xyz in zip(grid_centers, xyzs):
        shift_xyz = xyz - xyz.mean(0) + grid
        xyzs_list.append(shift_xyz)

    allxyzs = torch.stack( xyzs_list ).reshape(-1, 3).detach().cpu().numpy()
    allzs = np.concatenate( [atomic_nums] * n_h * n_w )

    atoms = Atoms(positions=allxyzs, numbers=allzs)
    
    return atoms 

def rotate_grid(allatoms, nsamples, axis='x', skip=3):

    rotate_trajs = []
    start_frame = allatoms.positions.reshape(nsamples, -1, 3)
    atomic_nums = allatoms.get_atomic_numbers()[:start_frame.shape[1]]

    for deg in np.linspace(0, 360, 360)[::skip]:
        rotate_frames = []

        for geom in start_frame:
            atoms = Atoms(positions=geom, numbers=atomic_nums)
            atoms.rotate(deg, axis, center='COM')    
            rotate_frames.append(atoms.positions)

        newallatoms =deepcopy(allatoms)
        newallatoms.positions = np.concatenate(rotate_frames)
        rotate_trajs.append(newallatoms)
    
    return rotate_trajs