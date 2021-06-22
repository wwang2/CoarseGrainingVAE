
import torch
from torch import nn
from conv import * 

from nff.utils.scatter import scatter_add
from torch_scatter import scatter_mean


class CGEquivariantEncoder(nn.Module):
    
    def __init__(self,
             n_conv,
             n_atom_basis,
             n_rbf,
             cutoff):
        super().__init__()

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)

        self.atom_mpnns = nn.ModuleList(
                            [EquivariantMPlayer(feat_dim=n_atom_basis,
                                                n_rbf=n_rbf, 
                                                activation='swish',
                                                cutoff=cutoff, 
                                                dropout=0.0)
                             for _ in range(n_conv)]
        )

        self.cg_mpnns = nn.ModuleList(
        [ContractiveEquivariantMPlayer(feat_dim=n_atom_basis,
                                         activation='swish',
                                         n_rbf=n_rbf,
                                         cutoff=cutoff,
                                         dropout=0.0)
         for _ in range(n_conv)])

        self.update_blocks = nn.ModuleList(
            [UpdateBlock(feat_dim=n_atom_basis,
                         activation="swish",
                         dropout=0.0)
             for _ in range(n_conv)]
        )
        
        self.n_conv = n_conv
    
    def init_embed(self, z, xyz, cg_xyz, mapping, nbr_list):
    
        nbr_list, _ = make_directed(nbr_list)

        r_ij = xyz[nbr_list[:, 1]] - xyz[nbr_list[:, 0]]
        d_ij = ((r_ij ** 2).sum(-1)) ** 0.5

        # intialize embeddings 
        h_i = self.atom_embed(z.to(torch.long))
        v_i = torch.zeros(len(h_i), h_i.shape[1],  3).to(cg_xyz.device)

        H_I = torch.zeros(len(cg_xyz), h_i.shape[1]).to(cg_xyz.device)
        V_I = torch.zeros(len(cg_xyz), h_i.shape[1], 3).to(cg_xyz.device)

        r_iI = (xyz - cg_xyz[mapping])
        
        d_iI, unit_r_iI = preprocess_r(r_iI)
        d_ij, unit_r_ij = preprocess_r(r_ij)
        
        return h_i, v_i, H_I, V_I, r_ij, d_ij, d_iI, unit_r_ij, unit_r_iI, nbr_list
    
    def forward(self, z, xyz, cg_xyz, mapping, nbr_list):
        
        h_i, v_i, H_I, V_I, r_ij, d_ij, d_iI, unit_r_ij, unit_r_iI, nbr_list = self.init_embed(z, xyz, 
                                                                                     cg_xyz, 
                                                                                     mapping, 
                                                                                     nbr_list)

        for i in range(self.n_conv):
            dh_i, dv_i = self.atom_mpnns[i](h_i=h_i,
                                            v_i=v_i,
                                            d_ij=d_ij,
                                            unit_r_ij=unit_r_ij,
                                            nbrs=nbr_list)
            
            h_i = h_i + dh_i
            v_i = v_i + dv_i


            dh_update, dv_update = self.update_blocks[i](h_i, v_i)
            h_i = h_i + dh_update
            v_i = v_i + dv_update

            if i == 0:
                H_I = H_I + scatter_mean(h_i, mapping, dim=0)

            dH_I, dV_I = self.cg_mpnns[i](h_i, v_i, d_iI, unit_r_iI, mapping)

            H_I = H_I + dH_I
            V_I = V_I + dV_I
        
        return H_I
    
class CGEquivariantDecoder(nn.Module):
    def __init__(self, n_atom_basis, n_rbf, cutoff, n_conv ):   
        nn.Module.__init__(self)
        
        self.decode_mpnns = nn.ModuleList(
                            [EquivariantMPlayer(feat_dim=n_atom_basis,
                                                n_rbf=n_rbf, 
                                                activation='swish',
                                                cutoff=cutoff, 
                                                dropout=0.0)
                             for _ in range(n_conv)]
        )
        self.n_conv = n_conv


        self.update_blocks = nn.ModuleList(
            [UpdateBlock(feat_dim=n_atom_basis,
                         activation="swish",
                         dropout=0.0)
             for _ in range(n_conv)]
        )
    
    
    def init_embed(self, cg_xyz, cg_nbr_list, H_I):

        cg_nbr_list, _ = make_directed(cg_nbr_list)
        
        r_ij = cg_xyz[cg_nbr_list[:, 1]] - cg_xyz[cg_nbr_list[:, 0]]
        
        V_i = torch.zeros(H_I.shape[0], H_I.shape[1], 3 ).to(H_I.device)
        
        d_ij, unit_r_ij = preprocess_r(r_ij)
        
        return V_i, d_ij, unit_r_ij, cg_nbr_list

    
    def forward(self, cg_xyz, CG_nbr_list, H_I):
        
        V_I, d_IJ, unit_r_IJ, CG_nbr_list = self.init_embed(cg_xyz, CG_nbr_list, H_I)

        for i in range(self.n_conv):
            
            dH_I, dV_I = self.decode_mpnns[i](h_i=H_I,
                                            v_i=V_I,
                                            d_ij=d_IJ,
                                            unit_r_ij=unit_r_IJ,
                                            nbrs=CG_nbr_list)
            
            H_I = H_I + dH_I 
            V_I = V_I + dV_I 

            update_block = self.update_blocks[i]
            dH_I, dV_I = update_block(s_i=H_I, v_i=V_I)

            H_I = H_I + dH_I 
            V_I = V_I + dV_I
            
        return H_I, V_I 


class EquivariantConv(nn.Module):
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv ):   
        
        nn.Module.__init__(self)
        # distance transform
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                  cutoff=cutoff,
                                  feat_dim=n_atom_basis,
                                  learnable_k=False,
                                  dropout=0.0)

        self.message_blocks = nn.ModuleList(
            [MessageBlock(feat_dim=n_atom_basis,
                          activation="swish",
                          n_rbf=n_rbf,
                          cutoff=cutoff,
                          learnable_k=False,
                          dropout=0.0)
             for _ in range(num_conv)]
        )

        self.update_blocks = nn.ModuleList(
            [UpdateBlock(feat_dim=n_atom_basis,
                         activation="swish",
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
             cutoff):
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
                          activation="swish",
                          n_rbf=n_rbf,
                          cutoff=cutoff,
                          learnable_k=False,
                          dropout=0.0)
             for _ in range(n_conv)]
        )

        self.update_blocks = nn.ModuleList(
            [UpdateBlock(feat_dim=n_atom_basis,
                         activation="swish",
                         dropout=0.0)
             for _ in range(n_conv)]
        )

        self.cgmessage_layers = nn.ModuleList(
        [ContractiveMessageBlock(feat_dim=n_atom_basis,
                                         activation='swish',
                                         n_rbf=n_rbf,
                                         cutoff=cutoff,
                                         learnable_k=False,
                                         dropout=0.0)
         for _ in range(n_conv)])
        
        self.n_conv = n_conv
    
    def forward(self, z, xyz, cg_xyz, mapping, nbr_list):
        
        # atomic embedding
        s_i = self.atom_embed(z.long())
        v_i = torch.zeros(s_i.shape[0], s_i.shape[1], 3).to(s_i.device)

        r_ij = xyz[nbr_list[:, 1]] - xyz[nbr_list[:, 0]]

        # edge features

        #dist, unit = preprocess_r(r_ij)

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

            delta_S_I, delta_V_I = self.cgmessage_layers[i](s_i, v_i, r_iI, mapping)

            S_I += delta_S_I
            V_I += delta_V_I
        
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
        
        S_I = self.encoder(z, xyz, cg_xyz, mapping, nbr_list)
        
        S_mu = self.atom_munet(S_I)
        S_logvar = self.atom_sigmanet(S_I)
        S_sigma = torch.exp(S_logvar / 2)
        S_I = self.reparametrize(S_mu, S_sigma)
        
        xyz_recon = self.decoder(cg_xyz, CG_nbr_list, S_I, mapping, num_CGs)
        
        return S_mu, S_sigma, xyz, xyz_recon

    