
import torch
from torch import nn
from conv import * 
from torch_scatter import scatter_mean, scatter_add

class EquivariantDecoder(nn.Module):
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv, activation ):   
        
        nn.Module.__init__(self)
        # distance transform
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                  cutoff=cutoff,
                                  feat_dim=n_atom_basis,
                                  learnable_k=False,
                                  dropout=0.0)

        self.message_blocks = nn.ModuleList(
            [MessageBlock(feat_dim=n_atom_basis,
                          activation=activation,
                          n_rbf=n_rbf,
                          cutoff=cutoff,
                          learnable_k=False,
                          dropout=0.0)
             for _ in range(num_conv)]
        )

        self.update_blocks = nn.ModuleList(
            [UpdateBlock(feat_dim=n_atom_basis,
                         activation=activation,
                         dropout=0.0)
             for _ in range(num_conv)]
        )
    
    
    def forward(self, cg_xyz, CG_nbr_list, cg_s):
    
        CG_nbr_list, _ = make_directed(CG_nbr_list)
        r_ij = cg_xyz[CG_nbr_list[:, 1]] - cg_xyz[CG_nbr_list[:, 0]]
        
        v_i = torch.zeros(cg_s.shape[0], cg_s.shape[1], 3 ).to(cg_s.device)
        s_i = cg_s

        # inputs need to come from atomwise feature toulene_dft
        for i, message_block in enumerate(self.message_blocks):
            
            # message block
            ds_message, dv_message = message_block(s_j=s_i,
                                                   v_j=v_i,
                                                   r_ij=r_ij,
                                                   nbrs=CG_nbr_list,
                                                   )
            s_i = s_i + ds_message
            v_i = v_i + dv_message

            # update block
            update_block = self.update_blocks[i]
            ds_update, dv_update = update_block(s_i=s_i,
                                                v_i=v_i)
            s_i = s_i + ds_update
            v_i = v_i + dv_update
            
        return s_i, v_i 


class CGEncoder(nn.Module):
    
    def __init__(self,
             n_conv,
             n_atom_basis,
             n_rbf,
             activation,
             cutoff,
             dir_mp=False):
        super().__init__()

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)
        # distance transform
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                  cutoff=cutoff,
                                  feat_dim=n_atom_basis,
                                  learnable_k=False,
                                  dropout=0.0)

        self.message_blocks = nn.ModuleList(
            [MessageBlock(feat_dim=n_atom_basis,
                          activation=activation,
                          n_rbf=n_rbf,
                          cutoff=cutoff,
                          learnable_k=False,
                          dropout=0.0)
             for _ in range(n_conv)]
        )

        self.update_blocks = nn.ModuleList(
            [UpdateBlock(feat_dim=n_atom_basis,
                         activation=activation,
                         dropout=0.0)
             for _ in range(n_conv)]
        )

        self.cg_message_blocks = nn.ModuleList(
            [MessageBlock(feat_dim=n_atom_basis,
                          activation=activation,
                          n_rbf=n_rbf,
                          cutoff=cutoff,
                          learnable_k=False,
                          dropout=0.0)
             for _ in range(n_conv)]
        )

        self.cg_update_blocks = nn.ModuleList(
            [UpdateBlock(feat_dim=n_atom_basis,
                         activation=activation,
                         dropout=0.0)
             for _ in range(n_conv)]
        )

        self.cgmessage_layers = nn.ModuleList(
        [ContractiveMessageBlock(feat_dim=n_atom_basis,
                                         activation=activation,
                                         n_rbf=n_rbf,
                                         cutoff=cutoff,
                                         learnable_k=False,
                                         dropout=0.0)
         for _ in range(n_conv)])
        
        self.n_conv = n_conv
        self.dir_mp = dir_mp
    
    def forward(self, z, xyz, cg_xyz, mapping, nbr_list, cg_nbr_list):
        
        # atomic embedding
        if not self.dir_mp:
            nbr_list, _ = make_directed(nbr_list)
        else:
            pass

        s_i = self.atom_embed(z.long())
        v_i = torch.zeros(s_i.shape[0], s_i.shape[1], 3).to(s_i.device)

        r_ij = xyz[nbr_list[:, 1]] - xyz[nbr_list[:, 0]]
        #r_IJ = cg_xyz[cg_nbr_list[:, 1]] - xyz[cg_nbr_list[:, 0]]

        # edge features
        r_iI = (xyz - cg_xyz[mapping])

        for i in range(self.n_conv):
            ds_message, dv_message = self.message_blocks[i](s_j=s_i,
                                                   v_j=v_i,
                                                   r_ij=r_ij,
                                                   nbrs=nbr_list)
            s_i = s_i + ds_message
            v_i = v_i + dv_message

            # update block
            ds_update, dv_update = self.update_blocks[i](s_i=s_i, v_i=v_i)
            s_i = s_i + ds_update # atom message 
            v_i = v_i + dv_update

            # contruct atom messages 
            if i == 0:
                S_I = scatter_mean(s_i, mapping, dim=0)
                V_I = torch.zeros(S_I.shape[0], S_I.shape[1], 3 ).to(S_I.device)

            # CG message passing 
            delta_S_I, delta_V_I = self.cgmessage_layers[i](s_i, v_i, r_iI, mapping)

            S_I = S_I + delta_S_I
            V_I = V_I + delta_V_I

            # dS_message, dV_message = self.cg_message_blocks[i](s_j=S_I,
            #                                        v_j=V_I,
            #                                        r_ij=r_IJ,
            #                                        nbrs=cg_nbr_list)
            # S_I = S_I + dS_message
            # V_I = V_I + dV_message

            # dS_update, dV_update = self.cg_update_blocks[i](s_i=S_I, v_i=V_I)
            # S_I = S_I + dS_update # atom message 
            # V_I = V_I + dV_update
        
        return S_I


class CGequiVAE(nn.Module):
    def __init__(self, encoder, equivaraintconv, 
                     atom_munet, atom_sigmanet,
                    n_atoms, n_cgs, feature_dim):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.equivaraintconv = equivaraintconv
        self.atom_munet = atom_munet
        self.atom_sigmanet = atom_sigmanet
        
        self.n_atoms = n_atoms
        self.n_cgs = n_cgs
        self.atomdense = nn.Linear(feature_dim, n_atoms)
        
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
        xyz_rel = torch.einsum("ji,kin->kjn", self.atomdense.weight, pooled_vector.mean(1)).reshape(-1, 3)
        #xyz_recon = pooled_vector.mean(1)[:, :self.n_atoms, :].reshape(-1, 3) + cg_xyz[mapping]
        xyz_recon = xyz_rel + cg_xyz[mapping]
        
        return xyz_recon
        
    def forward(self, batch):

        z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs= self.get_inputs(batch)
        
        S_I = self.encoder(z, xyz, cg_xyz, mapping, nbr_list, CG_nbr_list)
        
        S_mu = self.atom_munet(S_I)
        S_logvar = self.atom_sigmanet(S_I)
        S_sigma = torch.exp(S_logvar / 2)
        S_I = self.reparametrize(S_mu, S_sigma)
        
        xyz_recon = self.decoder(cg_xyz, CG_nbr_list, S_I, mapping, num_CGs)
        
        return S_mu, S_sigma, xyz, xyz_recon

    