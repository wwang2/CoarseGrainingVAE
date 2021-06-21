import torch

from nff.nn.modules import (
    SchNetConv,
    SchNetEdgeUpdate,
    NodeMultiTaskReadOut
)

from torch import nn 

from nff.nn.modules.painn import (DistanceEmbed, MessageBlock, UpdateBlock,
                                  EmbeddingBlock, 
                                  to_module, norm, InvariantMessage, InvariantDense)

from nff.nn.layers import Dense
from nff.utils.tools import make_directed

from torch_scatter import scatter_mean
from nff.utils.scatter import scatter_add


from torch import nn



def preprocess_r(r_ij):

    dist = ((vec ** 2 + EPS).sum(-1)) ** 0.5
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