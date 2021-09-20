import torch
from torch import nn
from conv import * 
from torch_scatter import scatter_mean, scatter_add


class Baseline(nn.Module):
    def __init__(self, pooler, n_cgs, n_atoms):
        nn.Module.__init__(self)
        self.pooler = pooler 
        self.B = nn.Parameter(0.0 * torch.randn(n_cgs, n_atoms))
        
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


class EquiLinear(nn.Module):
    def __init__(self, pooler, n_cgs, n_atoms, cross):
        nn.Module.__init__(self)
        self.pooler = pooler 
        if cross:
            self.B = nn.Parameter(0.01 * torch.randn(n_atoms, n_cgs ** 2 + (n_cgs ** 2)**2 ) )
        else:
            self.B = nn.Parameter(0.01 * torch.randn(n_atoms, n_cgs ** 2 ) )

        self.cross = cross
        
    def forward(self, batch):
    
        xyz = batch['xyz']        
        device = xyz.device
        
        z = batch['z'] # torch.ones_like( batch['z'] ) 
        nbr_list = batch['nbr_list']

        soft_assign, assign_norm, h, H, adj, cg_xyz, soft_cg_adj = self.pooler(z, 
                                                                   batch['xyz'], 
                                                                   batch['bonds'], 
                                                                   tau=0.0,
                                                                   gumbel=True)
        basis = cg_xyz.unsqueeze(1) - cg_xyz.unsqueeze(2)
        B = basis.reshape(h.shape[0], -1, 3)
        

        if self.cross:
            npair = B.shape[1]
            nbatch = B.shape[0]
            
            Bij = B.unsqueeze(1).expand(-1, npair, -1, -1).reshape(nbatch, -1,  3)
            Bjk = B.unsqueeze(2).expand(-1, -1, npair, -1).reshape(nbatch, -1, 3)
            
            cross_basis = torch.cross(Bij, Bjk, dim=-1)
            B = torch.cat((B, cross_basis), dim=1)

        #dx = xyz - cg_xyz[:, self.pooler.assign_idx, :]
        
        dx_recon = torch.einsum("ije,nj->ine", B, self.B )

        # recentering 
        cg_offset = torch.einsum("bin,bij->bjn", dx_recon, assign_norm)
        cg_offset_lift = cg_offset[:, self.pooler.assign_idx, :]

        xyz_recon = cg_xyz[:, self.pooler.assign_idx, :] - cg_offset_lift + dx_recon
    
        return soft_assign, xyz, xyz_recon