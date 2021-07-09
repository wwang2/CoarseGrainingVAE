
import torch
from torch import nn
from conv import * 
from torch_scatter import scatter_mean, scatter_add


class ENDecoder(nn.Module):
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv, activation ):   
        
        nn.Module.__init__(self)
        # distance transform
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                  cutoff=cutoff,
                                  feat_dim=n_atom_basis,
                                  dropout=0.0)

        self.message_blocks = nn.ModuleList(
            [ENMessageBlock(feat_dim=n_atom_basis,
                          activation=activation,
                          n_rbf=n_rbf,
                          cutoff=cutoff,
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
            
        return s_i, v_i 

class EquivariantDecoder(nn.Module):
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv, activation, atomwise_z=False):   
        
        nn.Module.__init__(self)
        # distance transform
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                  cutoff=cutoff,
                                  feat_dim=n_atom_basis,
                                  dropout=0.0)

        self.message_blocks = nn.ModuleList(
            [EquiMessageBlock(feat_dim=n_atom_basis,
                          activation=activation,
                          n_rbf=n_rbf,
                          cutoff=cutoff,
                          dropout=0.0)
             for _ in range(num_conv)]
        )

        self.update_blocks = nn.ModuleList(
            [UpdateBlock(feat_dim=n_atom_basis,
                         activation=activation,
                         dropout=0.0)
             for _ in range(num_conv)]
        )
        self.atomwise_z = atomwise_z

        # layers to couple H and h

        self.atom2CGcoupling = nn.Sequential(Dense(in_features=2 * n_atom_basis,
                                           out_features=2 * n_atom_basis,
                                           bias=True,
                                           activation=to_module(activation)),
                                     Dense(in_features=2 * n_atom_basis,
                                           out_features=n_atom_basis,
                                           bias=True))
    
    def forward(self, cg_xyz, CG_nbr_list, mapping, H, h):
    
        CG_nbr_list, _ = make_directed(CG_nbr_list)
        r_ij = cg_xyz[CG_nbr_list[:, 1]] - cg_xyz[CG_nbr_list[:, 0]]
        
        V = torch.zeros(H.shape[0], H.shape[1], 3 ).to(H.device)

        for i, message_block in enumerate(self.message_blocks):
            
            # message block
            dH_message, dV_message = message_block(s_j=H,
                                                   v_j=V,
                                                   r_ij=r_ij,
                                                   nbrs=CG_nbr_list,
                                                   )
            H = H + dH_message
            V = V + dV_message

            # update block
            dH_update, dV_update = self.update_blocks[i](s_i=H,
                                                v_i=V)
            H = H + dH_update
            V = V + dV_update

            if self.atomwise_z:
                # perform coupling 
                # contraction 
                Hatom = scatter_mean(h, mapping, dim=0)
                # concat
                H_h_cat =  torch.cat((Hatom, H), dim=-1)
                # coupling layer 
                dH = self.atom2CGcoupling(H_h_cat)
                H = H + dH

        return H, V 


class EquiEncoder(nn.Module):
    
    def __init__(self,
             n_conv,
             n_atom_basis,
             n_rbf,
             activation,
             cutoff,
             dir_mp=False,
             cg_mp=False,
             atomwise_z=False):
        super().__init__()

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)
        # distance transform
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                  cutoff=cutoff,
                                  feat_dim=n_atom_basis,
                                  dropout=0.0)

        self.message_blocks = nn.ModuleList(
            [EquiMessageBlock(feat_dim=n_atom_basis,
                          activation=activation,
                          n_rbf=n_rbf,
                          cutoff=cutoff,
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
            [EquiMessageBlock(feat_dim=n_atom_basis,
                          activation=activation,
                          n_rbf=n_rbf,
                          cutoff=cutoff,
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
                                         dropout=0.0)
         for _ in range(n_conv)])
        
        self.atom2CGcoupling = nn.Sequential(Dense(in_features=2 * n_atom_basis,
                                           out_features=2 * n_atom_basis,
                                           bias=True,
                                           activation=to_module(activation)),
                                     Dense(in_features=2 * n_atom_basis,
                                           out_features=n_atom_basis,
                                           bias=True))

        self.n_conv = n_conv
        self.dir_mp = dir_mp
        self.cg_mp = cg_mp
        self.atomwise_z = atomwise_z
    
    def forward(self, z, xyz, cg_xyz, mapping, nbr_list, cg_nbr_list):
        
        # atomic embedding
        if not self.dir_mp:
            nbr_list, _ = make_directed(nbr_list)

        h = self.atom_embed(z.long())
        v = torch.zeros(h.shape[0], h.shape[1], 3).to(h.device)

        r_ij = xyz[nbr_list[:, 1]] - xyz[nbr_list[:, 0]]
        r_IJ = cg_xyz[cg_nbr_list[:, 1]] - xyz[cg_nbr_list[:, 0]]

        # edge features
        r_iI = (xyz - cg_xyz[mapping])

        for i in range(self.n_conv):
            ds_message, dv_message = self.message_blocks[i](s_j=h,
                                                   v_j=v,
                                                   r_ij=r_ij,
                                                   nbrs=nbr_list)
            h = h + ds_message
            v = v + dv_message

            # update block
            ds_update, dv_update = self.update_blocks[i](s_i=h, v_i=v)
            h = h + ds_update # atom message 
            v = v + dv_update

            # contruct atom messages 
            if i == 0:
                H = scatter_mean(h, mapping, dim=0)
                V = torch.zeros(H.shape[0], H.shape[1], 3 ).to(H.device)

            # CG message passing 
            delta_H, delta_V = self.cgmessage_layers[i](h, v, r_iI, mapping)

            H = H + delta_H
            V = V + delta_V

            dH_update, dV_update = self.cg_update_blocks[i](s_i=H, v_i=V)
            H = H + dH_update # atom message 
            V = V + dV_update

            if self.atomwise_z:
                # perform coupling 
                # contraction 
                # concat
                H_h_cat =  torch.cat((h, H[mapping]), dim=-1)
                # coupling layer 
                dh = self.atom2CGcoupling(H_h_cat)
                h = h + dh
            # if self.cg_mp:
            #     dS_message, dV_message = self.cg_message_blocks[i](s_j=S_I,
            #                                            v_j=V_I,
            #                                            r_ij=r_IJ,
            #                                            nbrs=cg_nbr_list)
            #     S_I = S_I + dS_message
            #     V_I = V_I + dV_message

            #     dS_update, dV_update = self.cg_update_blocks[i](s_i=S_I, v_i=V_I)
            #     S_I = S_I + dS_update # atom message 
            #     V_I = V_I + dV_update
          
        return H, h


class CGequiVAE(nn.Module):
    def __init__(self, encoder, equivaraintconv, 
                     atom_munet, atom_sigmanet,
                    n_atoms, n_cgs, feature_dim, atomwise_z=False):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.equivaraintconv = equivaraintconv
        self.atom_munet = atom_munet
        self.atom_sigmanet = atom_sigmanet
        
        self.n_atoms = n_atoms
        self.n_cgs = n_cgs
        self.atomdense = nn.Linear(feature_dim, n_atoms)
        self.atomwise_z = atomwise_z
        
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

    def CG2ChannelIdx(self, CG_mapping):

        CG2atomChannel = torch.zeros_like(CG_mapping)

        for cg_type in torch.unique(CG_mapping): 
            cg_filter = CG_mapping == cg_type
            num_contri_atoms = cg_filter.sum().item()
            CG2atomChannel[cg_filter] = torch.LongTensor(list(range(num_contri_atoms))).to(CG_mapping.device)
            
        return CG2atomChannel.detach()
            
    def decoder(self, cg_xyz, CG_nbr_list, S_I, s_i, mapping, num_CGs):
        
        cg_s, cg_v = self.equivaraintconv(cg_xyz, CG_nbr_list, mapping,S_I, s_i)

        # pooled_vector = []
        # v_i_splits = torch.split(cg_v, num_CGs.tolist()) 

        # for vec in v_i_splits:
        #     pooled_vector.append(vec)

        # pooled_vector = torch.stack(pooled_vector)
        # xyz_rel = torch.einsum("ji,kin->kjn", self.atomdense.weight, pooled_vector.mean(1)).reshape(-1, 3)
        # #xyz_recon = pooled_vector.mean(1)[:, :self.n_atoms, :].reshape(-1, 3) + cg_xyz[mapping]
        # xyz_recon = xyz_rel + cg_xyz[mapping]

        CG2atomChannel = self.CG2ChannelIdx(mapping)
        xyz_rel = cg_v[mapping, CG2atomChannel, :]
        xyz_recon = xyz_rel + cg_xyz[mapping]
        
        return xyz_recon
        
    def forward(self, batch):

        atomic_nums, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs= self.get_inputs(batch)
        
        S_I, s_i = self.encoder(atomic_nums, xyz, cg_xyz, mapping, nbr_list, CG_nbr_list)
        
        if self.atomwise_z:
            z = s_i 
        else:
            z = S_I

        mu = self.atom_munet(z)
        logvar = self.atom_sigmanet(z)
        sigma = 1e-6 + torch.exp(logvar / 2)
        z_sample = self.reparametrize(mu, sigma)

        if self.atomwise_z:
            S_I = scatter_mean(s_i, mapping, dim=0)
            s_i = z_sample 
        else:
            S_I = z_sample # s_i not used in decoding 
        
        xyz_recon = self.decoder(cg_xyz, CG_nbr_list, S_I, s_i, mapping, num_CGs)
        
        return mu, sigma, xyz, xyz_recon

    