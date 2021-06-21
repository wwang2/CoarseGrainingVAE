import torch

from nff.nn.modules import (
    SchNetConv,
    SchNetEdgeUpdate,
    NodeMultiTaskReadOut
)

from torch import nn 

from modules import * 

from nff.utils.tools import make_directed

from torch_scatter import scatter_mean, scatter_add
from torch import nn

from nff.nn.modules.painn import (DistanceEmbed, MessageBlock, UpdateBlock,
                                  EmbeddingBlock, ReadoutBlock, 
                                  to_module, norm, InvariantMessage, preprocess_r, InvariantDense)


def to_module(activation):
    return layer_types[activation]()

def preprocess_r(r_ij):

    dist = ((r_ij ** 2 + 1e-8).sum(-1)) ** 0.5
    unit = r_ij / dist.reshape(-1, 1)

    return dist, unit


class CGequivariantEncoder(nn.Module):
    
    def __init__(self,
             n_conv,
             n_atom_basis,
             n_rbf,
             cutoff):
        super().__init__()

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)

        self.embed_block = EmbeddingBlock(feat_dim=n_atom_basis)
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
        s_i, v_i = self.embed_block(z.long())
        r_ij = xyz[nbr_list[:, 1]] - xyz[nbr_list[:, 0]]

        # edge features
        dis_vec = r_ij
        dis = norm(r_ij)
        xyz_adjoint = dis_vec / dis.unsqueeze(-1)

        r_iI = (xyz - cg_xyz[mapping])


        for i in range(self.n_conv):
            ds_message, dv_message = self.message_blocks[i](s_j=s_i,
                                                   v_j=v_i,
                                                   r_ij=r_ij,
                                                   nbrs=nbr_list,
                                                   e_ij=None)
            s_i = s_i + ds_message
            v_i = v_i + dv_message

            # update block
            ds_update, dv_update = self.update_blocks[i](s_i=s_i,
                                                v_i=v_i)
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


class ContractiveMessageBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 learnable_k,
                 dropout):
        super().__init__()

        
        self.inv_dense = InvariantDense(dim=feat_dim,
                                        activation=activation,
                                        dropout=dropout)
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                        cutoff=cutoff,
                                        feat_dim=feat_dim,
                                        learnable_k=learnable_k,
                                        dropout=dropout)
        self.edge_embed = Dense(in_features=feat_dim,
                                out_features=3 * feat_dim,
                                bias=True,
                                dropout_rate=dropout)


    def forward(self,
                s_i,
                v_i,
                r_iI,
                mapping):

        dist, unit = preprocess_r(r_iI)
        phi = self.inv_dense(s_i)
        
        w_s = self.dist_embed(dist)
            
        inv_out = phi * w_s

        # split into three components, so the tensor now has
        # shape n_atoms x 3 x feat_dim
        inv_out = inv_out.reshape(inv_out.shape[0], 3, -1)

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_iI = unit_add + split_0 * v_i
        delta_s_iI = split_1

        # add results from neighbors of each node

        delta_v_I = scatter_mean(src=delta_v_iI,
                                index=mapping,
                                dim=0)

        delta_s_I = scatter_mean(src=delta_s_iI,
                                index=mapping,
                                dim=0)

        return delta_s_I, delta_v_I


class EquivariantMPlayer(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 dropout):
        super().__init__()
        self.dist_embed = SchNetEdgeFilter(n_gaussians=n_rbf,
                                        cutoff=cutoff,
                                        n_filters=feat_dim)

        self.layers = nn.Sequential(Dense(in_features=feat_dim,
                                                  out_features=feat_dim,
                                                  bias=True,
                                                  dropout_rate=dropout,
                                                  activation=to_module(activation)),
                                            Dense(in_features=feat_dim,
                                                  out_features=feat_dim,
                                                  bias=True,
                                                  dropout_rate=dropout))
        
        self.inv2equi_filters = nn.ModuleList([Dense(in_features=feat_dim,
                                                      out_features=feat_dim,
                                                      bias=True,
                                                      dropout_rate=dropout)
                                                 for _ in range(3)]
                                            )
        
    def forward(self, h_i, v_i, d_ij, unit_r_ij, nbrs):
        
        phi = self.layers(h_i)
        edge_inv = phi[nbrs[:, 0]] * self.dist_embed(d_ij.unsqueeze(-1))
        
        dv = self.inv2equi_filters[0](edge_inv).unsqueeze(-1) * unit_r_ij.unsqueeze(1) + \
            self.inv2equi_filters[1](edge_inv).unsqueeze(-1) * v_i[nbrs[:, 1]]

        dh = self.inv2equi_filters[1](edge_inv)
           
        # perform aggregation 
        h_i = h_i + scatter_add(dh, nbrs[:,0], dim=0, dim_size=len(h_i))
        v_i = v_i + scatter_add(dv, nbrs[:,0], dim=0, dim_size=len(h_i))
        
        return h_i, v_i 


class ContractiveEquivariantMPlayer(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 dropout):
        super().__init__()

        self.dist_embed = SchNetEdgeFilter(n_gaussians=n_rbf,
                                        cutoff=cutoff,
                                        n_filters=feat_dim)

        self.layers = nn.Sequential(Dense(in_features=feat_dim,
                                                  out_features=feat_dim,
                                                  bias=True,
                                                  dropout_rate=dropout,
                                                  activation=to_module(activation)),
                                            Dense(in_features=feat_dim,
                                                  out_features=feat_dim,
                                                  bias=True,
                                                  dropout_rate=dropout))
        
        self.inv2equi_filters = nn.ModuleList([Dense(in_features=feat_dim,
                                                      out_features=feat_dim,
                                                      bias=True,
                                                      dropout_rate=dropout)
                                                 for _ in range(3)]
                                            ) # todo: use an MLP instead 
        
    def forward(self, h_i, v_i, d_iI, unit_r_iI, mapping):

        phi = self.layers(h_i)
        edge_inv = h_i * self.dist_embed(d_iI.unsqueeze(-1))
        
        dv = self.inv2equi_filters[0](edge_inv).unsqueeze(-1) * unit_r_iI.unsqueeze(1) + \
            self.inv2equi_filters[1](edge_inv).unsqueeze(-1) * v_i

        dh = self.inv2equi_filters[1](edge_inv)
           
        # perform aggregation 
        dh_i = scatter_mean(dh, mapping, dim=0)
        dv_i = scatter_mean(dv, mapping, dim=0)
        
        return dh_i, dv_i 
    
    
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
            
        return H_I, V_I 


class EquivariantConv(nn.Module):
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv ):   
        
        nn.Module.__init__(self)
        # embedding layers
        self.embed_block = EmbeddingBlock(feat_dim=n_atom_basis)
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

        dis_vec = r_ij
        dis = norm(r_ij)
        e_ij = self.dist_embed(dis)

        # inputs need to come from atomwise feature toulene_dft
        for i, message_block in enumerate(self.message_blocks):
            
            # message block
            ds_message, dv_message = message_block(s_j=s_i,
                                                   v_j=v_i,
                                                   r_ij=r_ij,
                                                   nbrs=CG_nbr_list,
                                                   e_ij=None)
            s_i = s_i + ds_message
            v_i = v_i + dv_message

            # update block
            update_block = self.update_blocks[i]
            ds_update, dv_update = update_block(s_i=s_i,
                                                v_i=v_i)
            s_i = s_i + ds_update
            v_i = v_i + dv_update
            
        return s_i, v_i 
