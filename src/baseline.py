import torch
from torch import nn
from conv import * 
from torch_scatter import scatter_mean, scatter_add


class Baseline(nn.Module):
    def __init__(self, pooler, n_cgs, n_atoms):
        nn.Module.__init__(self)
        self.pooler = pooler 
        self.B = nn.Parameter(torch.zeros(n_cgs, n_atoms))
        
    def forward(self, batch):
    
        xyz = batch['xyz']        
        device = xyz.device
        
        z = batch['z'] # torch.ones_like( batch['z'] ) 
        nbr_list = batch['nbr_list']

        soft_assign, h, H, adj, cg_xyz, soft_cg_adj = self.pooler(z, 
                                                                   batch['xyz'], 
                                                                   batch['bonds'], 
                                                                   tau=0.0,
                                                                   gumbel=True)

        x_recon = torch.einsum("bce,ca->bae", cg_xyz, self.B)
        
        return soft_assign, xyz, x_recon