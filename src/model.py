
import torch
from torch import nn
from conv import * 

from nff.utils.scatter import scatter_add
from torch_scatter import scatter_mean


class CGequiVAE(nn.Module):
    def __init__(self, encoder, equivaraintconv, 
                     atom_munet, atom_sigmanet,
                    n_atoms, n_cgs):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.equivaraintconv = equivaraintconv
        self.atom_munet = atom_munet
        self.atom_sigmanet = atom_sigmanet
        
        self.n_atoms = n_atoms
        self.n_cgs = n_cgs
        
    def get_inputs(self, batch):

        xyz = batch['nxyz'][:, 1:]

        cg_xyz = batch['CG_nxyz'][:, 1:]
        batch['nxyz'][:, 1:]

        cg_z = batch['CG_nxyz'][:, 0]
        z = batch['nxyz'][:, 0]

        mapping = batch['CG_mapping']

        nbr_list = batch['nbr_list']
        CG_nbr_list = batch['CG_nbr_list']
        
        num_CGs = batch['num_CGs']
        
        return z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs
        
    def reparametrize(self, mu, sigma):
        if self.training:
            eps = torch.randn_like(sigma)
            S_I = eps.mul(sigma).add_(mu)
        else:
            S_I = sigma
            
        return S_I
        
    def decoder(self, cg_xyz, CG_nbr_list, S_I, mapping, num_CGs):
        
        cg_s, cg_v = self.equivaraintconv(cg_xyz, CG_nbr_list ,S_I)

        pooled_vector = []
        v_i_splits = torch.split(cg_v, num_CGs.tolist()) 

        for vec in v_i_splits:
            pooled_vector.append(vec)

        pooled_vector = torch.stack(pooled_vector)
        xyz_recon = pooled_vector.mean(1)[:, :self.n_atoms, :].reshape(-1, 3) + cg_xyz[mapping]
        
        return xyz_recon
        
    def forward(self, batch):

        z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs= self.get_inputs(batch)
        
        S_I = self.encoder(z, xyz, cg_xyz, mapping, nbr_list)
        
        S_mu = self.atom_munet(S_I)
        S_logvar = self.atom_sigmanet(S_I)
        S_sigma = torch.exp(S_logvar / 2)
        S_I = self.reparametrize(S_mu, S_sigma)
        
        xyz_recon = self.decoder(cg_xyz, CG_nbr_list, S_I, mapping, num_CGs)
        
        return S_mu, S_sigma, xyz, xyz_recon
    