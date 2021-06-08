
from ase import Atoms 
import torch
import numpy as np 

def xyz_grid_view(xyzs, num_atoms, n_w, n_h, grid_size):
    
    assert len(num_atoms) == n_w * n_h

    x_basis = torch.Tensor([grid_size, 0.0, 0.0])
    y_basis = torch.Tensor([0.0, grid_size, 0.0])

    grid_centers = []

    for i in range(n_w):
        for j in range(n_h): 
            grid_centers.append(x_basis * i + y_basis * j)

    grid_centers = torch.stack(grid_centers)

    xyzs = torch.split(xyz_recon.detach().cpu(), batch['num_atoms'].tolist())
    xyzs_list = []

    for grid, xyz in zip(grid_centers, xyzs):
        shift_xyz = xyz - xyz.mean(0) + grid
        xyzs_list.append(shift_xyz)

    allxyzs = torch.stack( xyzs_list ).reshape(-1, 3).detach().cpu().numpy()
    allzs = np.concatenate( [atomic_nums] * n_h * n_w )

    atoms = Atoms(positions=allxyzs, numbers=allzs)
    
    return atoms 