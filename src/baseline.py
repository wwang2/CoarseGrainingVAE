import torch
from torch import nn
from conv import * 
from torch_scatter import scatter_mean, scatter_add


class Baseline(nn.Module):
    def __init__(self, pooler, n_cgs, n_atoms):
        nn.Module.__init__(self)
        self.pooler = pooler 
        self.B = nn.Parameter(0.01 * torch.randn(n_cgs, n_atoms))
        
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

        x_recon = torch.einsum("bce,ca->bae", cg_xyz, self.B)
        
        return soft_assign, xyz, x_recon


class MLP(nn.Module):
    def __init__(self, pooler, n_cgs, n_atoms):
        nn.Module.__init__(self)
        self.pooler = pooler 

        input_dim = n_cgs * 3 
        output_dim = n_atoms * 3 

        self.n_cgs = n_cgs 
        self.n_atoms = n_atoms

        self.mlp = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim), 
                                    nn.ReLU(), 
                                    torch.nn.Linear(output_dim, output_dim), 
                                    nn.ReLU(),  
                                    torch.nn.Linear(output_dim, output_dim))
        
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

        x_recon = self.mlp(cg_xyz.reshape(-1, self.n_cgs * 3))

        x_recon = x_recon.reshape(-1, self.n_atoms, 3)
        
        return soft_assign, xyz, x_recon


class EquiMLP(nn.Module):
    def __init__(self, pooler, n_cgs, n_atoms, width=1, depth=1, activation='ReLU'):
        nn.Module.__init__(self)
        self.pooler = pooler 

        self.input_dim = int( n_cgs  * (n_cgs - 1) / 2 )
        self.output_dim = self.input_dim * n_atoms
        self.layer_width = self.output_dim * width

        self.n_cgs = n_cgs 
        self.n_atoms = n_atoms

        layer_list = [ torch.nn.Linear(self.input_dim, self.layer_width ) ] + \
                    [to_module(activation), torch.nn.Linear(self.layer_width, self.layer_width)] * depth + \
                    [to_module(activation), torch.nn.Linear(self.layer_width, self.output_dim)]

        self.mlp = torch.nn.Sequential(*layer_list)
        
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

        dist = basis.pow(2).sum(-1).sqrt()
        triu_indx = torch.ones_like(dist[0]).triu(diagonal=1).nonzero()

        dist_input = dist[:, triu_indx[:,0 ], triu_indx[:,1]]
        basis_triu = basis[:,triu_indx[:,0 ], triu_indx[:,1] ]

        coeffs = self.mlp(dist_input).reshape(dist_input.shape[0], self.n_atoms, self.input_dim)

        dx_recon = torch.einsum("ije,inj->ine", basis_triu, coeffs )

        # recentering 
        cg_offset = torch.einsum("bin,bij->bjn", dx_recon, assign_norm)
        cg_offset_lift = cg_offset[:, self.pooler.assign_idx, :]

        xyz_recon = cg_xyz[:, self.pooler.assign_idx, :] - cg_offset_lift + dx_recon
        
        return soft_assign, xyz, xyz_recon


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