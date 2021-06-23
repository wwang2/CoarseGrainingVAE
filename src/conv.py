import torch
from torch import nn 

from modules import * 

from torch_scatter import scatter_mean, scatter_add
from torch import nn

#from nff.nn.modules.painn import (MessageBlock, UpdateBlock,InvariantMessage, preprocess_r, InvariantDense)
                                  
def make_directed(nbr_list):

    gtr_ij = (nbr_list[:, 0] > nbr_list[:, 1]).any().item()
    gtr_ji = (nbr_list[:, 1] > nbr_list[:, 0]).any().item()
    directed = gtr_ij and gtr_ji

    if directed:
        return nbr_list, directed

    new_nbrs = torch.cat([nbr_list, nbr_list.flip(1)], dim=0)
    return new_nbrs, directed

def to_module(activation):
    return layer_types[activation]()

def preprocess_r(r_ij):
    dist = ((r_ij ** 2 + 1e-8).sum(-1)) ** 0.5
    unit = r_ij / dist.reshape(-1, 1)

    return dist, unit

class InvariantMessage(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 learnable_k,
                 dropout):
        super().__init__()

        self.inv_dense = nn.Sequential(Dense(in_features=feat_dim,
                                          out_features=feat_dim,
                                          bias=True,
                                          dropout_rate=dropout,
                                          activation=to_module(activation)),
                                    Dense(in_features=feat_dim,
                                          out_features=3 * feat_dim,
                                          bias=True,
                                          dropout_rate=dropout))

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
                s_j,
                dist,
                nbrs):

        phi = self.inv_dense(s_j)[nbrs[:, 1]]
        w_s = self.dist_embed(dist)
        output = phi * w_s
        # split into three components, so the tensor now has
        # shape n_atoms x 3 x feat_dim
        out_reshape = output.reshape(output.shape[0], 3, -1)

        return out_reshape


class MessageBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 learnable_k,
                 dropout):
        super().__init__()
        self.inv_message = InvariantMessage(feat_dim=feat_dim,
                                            activation=activation,
                                            n_rbf=n_rbf,
                                            cutoff=cutoff,
                                            learnable_k=learnable_k,
                                            dropout=dropout)

    def forward(self,
                s_j,
                v_j,
                r_ij,
                nbrs):

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_ij = unit_add + split_0 * v_j[nbrs[:, 1]]
        delta_s_ij = split_1

        # add results from neighbors of each node

        graph_size = s_j.shape[0]
        delta_v_i = scatter_add(src=delta_v_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        delta_s_i = scatter_add(src=delta_s_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        return delta_s_i, delta_v_i


class UpdateBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 dropout):
        super().__init__()
        self.u_mat = Dense(in_features=feat_dim,
                           out_features=feat_dim,
                           bias=False)
        self.v_mat = Dense(in_features=feat_dim,
                           out_features=feat_dim,
                           bias=False)
        self.s_dense = nn.Sequential(Dense(in_features=2*feat_dim,
                                           out_features=feat_dim,
                                           bias=True,
                                           dropout_rate=dropout,
                                           activation=to_module(activation)),
                                     Dense(in_features=feat_dim,
                                           out_features=3*feat_dim,
                                           bias=True,
                                           dropout_rate=dropout))

    def forward(self,
                s_i,
                v_i):

        # v_i = (num_atoms, num_feats, 3)
        # v_i.transpose(1, 2).reshape(-1, v_i.shape[1])
        # = (num_atoms, 3, num_feats).reshape(-1, num_feats)
        # = (num_atoms * 3, num_feats)
        # -> So the same u gets applied to each atom
        # and for each of the three dimensions, but differently
        # for the different feature dimensions

        v_tranpose = v_i.transpose(1, 2).reshape(-1, v_i.shape[1])

        # now reshape it to (num_atoms, 3, num_feats) and transpose
        # to get (num_atoms, num_feats, 3)

        num_feats = v_i.shape[1]
        u_v = (self.u_mat(v_tranpose).reshape(-1, 3, num_feats)
               .transpose(1, 2))
        v_v = (self.v_mat(v_tranpose).reshape(-1, 3, num_feats)
               .transpose(1, 2))

        v_v_norm = ((v_v ** 2 + 1e-10).sum(-1)) ** 0.5
        s_stack = torch.cat([s_i, v_v_norm], dim=-1)

        split = (self.s_dense(s_stack)
                 .reshape(s_i.shape[0], 3, -1))

        # delta v update
        a_vv = split[:, 0, :].unsqueeze(-1)
        delta_v_i = u_v * a_vv

        # delta s update
        a_sv = split[:, 1, :]
        a_ss = split[:, 2, :]

        inner = (u_v * v_v).sum(-1)
        delta_s_i = inner * a_sv + a_ss

        return delta_s_i, delta_v_i


class ContractiveMessageBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 learnable_k,
                 dropout):
        super().__init__()

        
        self.inv_dense = nn.Sequential(Dense(in_features=feat_dim,
                                          out_features=feat_dim,
                                          bias=True,
                                          dropout_rate=dropout,
                                          activation=to_module(activation)),
                                    Dense(in_features=feat_dim,
                                          out_features=3 * feat_dim,
                                          bias=True,
                                          dropout_rate=dropout))

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


########################### new models #################################


class InvariantFilter(nn.Module):
      def __init__(self,
                 feat_dim):
          super().__init__()
          self.inv2equi_filters = Dense(in_features=feat_dim,
                                    out_features=3 * feat_dim,
                                    bias=True)

      def forward(self, m_ij):
          
          filtered_msg = self.inv2equi_filters(m_ij).reshape(m_ij.shape[0], 3, -1)

          filter1 = filtered_msg[:, 0, :]
          filter2 = filtered_msg[:, 1, :]
          filter3 = filtered_msg[:, 2, :]

          return filter1, filter2, filter3


class EquivariantMPlayer(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 dropout):
        super().__init__()
        #self.dist_embed = SchNetEdgeFilter(n_gaussians=n_rbf,
        #                                cutoff=cutoff,
        #                                n_filters=feat_dim)

        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                        cutoff=cutoff,
                                        feat_dim=feat_dim,
                                        learnable_k=False,
                                        dropout=dropout)

        self.layers = nn.Sequential(Dense(in_features=feat_dim,
                                                  out_features=feat_dim,
                                                  bias=True,
                                                  dropout_rate=dropout,
                                                  activation=to_module(activation)),
                                            Dense(in_features=feat_dim,
                                                  out_features=3 * feat_dim,
                                                  bias=True,
                                                  dropout_rate=dropout))

        #self.edgefilters = InvariantFilter(feat_dim=feat_dim)        
        # self.inv2equi_filters = nn.ModuleList([nn.Sequential(
        #                                     Dense(in_features=feat_dim,
        #                                           out_features=feat_dim,
        #                                           bias=True,
        #                                           dropout_rate=dropout)) for _ in range(3)]
        #                                     ) # todo: use an MLP instead 
        
    def forward(self, h_i, v_i, d_ij, unit_r_ij, nbrs):
        
        phi = self.layers(h_i)
        edge_inv = phi[nbrs[:, 1]] * self.dist_embed(d_ij)
        #filter1, filter2, filter3 = self.edgefilters(edge_inv)
        
        edge_inv = edge_inv.reshape(edge_inv.shape[0], 3, -1)

        import ipdb; ipdb.set_trace()

        filter1 = edge_inv[:, 0 ]
        filter2 = edge_inv[:, 1 ]
        filter3 = edge_inv[:, 2 ]

        dv = filter1.unsqueeze(-1) * unit_r_ij.unsqueeze(1) + \
            filter2.unsqueeze(-1) * v_i[nbrs[:, 1]]

        dh = filter3
           
        # perform aggregation 
        dh_i = scatter_add(dh, nbrs[:,0], dim=0, dim_size=len(h_i))
        dv_i = scatter_add(dv, nbrs[:,0], dim=0, dim_size=len(h_i))
        
        return dh_i, dv_i 


class ContractiveEquivariantMPlayer(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 dropout):
        super().__init__()

        # self.dist_embed = SchNetEdgeFilter(n_gaussians=n_rbf,
        #                                 cutoff=cutoff,
        #                                 n_filters=feat_dim)

        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                        cutoff=cutoff,
                                        feat_dim=feat_dim,
                                        learnable_k=False,
                                        dropout=dropout)

        self.layers = nn.Sequential(Dense(in_features=feat_dim,
                                                  out_features=feat_dim,
                                                  bias=True,
                                                  dropout_rate=dropout,
                                                  activation=to_module(activation)),
                                            Dense(in_features=feat_dim,
                                                  out_features=3 * feat_dim,
                                                  bias=True,
                                                  dropout_rate=dropout))
      
        
    def forward(self, h_i, v_i, d_iI, unit_r_iI, mapping):

        # phi = self.layers(h_i)
        # edge_inv = h_i * self.dist_embed(d_iI)
        
        # dv = self.inv2equi_filters[0](edge_inv).unsqueeze(-1) * unit_r_iI.unsqueeze(1) + \
        #     self.inv2equi_filters[1](edge_inv).unsqueeze(-1) * v_i

        # dh = self.inv2equi_filters[2](edge_inv)
           
        phi = self.layers(h_i)
        edge_inv = phi * self.dist_embed(d_iI)
        #filter1, filter2, filter3 = self.edgefilters(edge_inv)
        edge_inv = edge_inv.reshape(edge_inv.shape[0], 3, -1)

        filter1 = edge_inv[:, 0, :]
        filter2 = edge_inv[:, 1, :]
        filter3 = edge_inv[:, 2, :]

        dv = filter1.unsqueeze(-1) * unit_r_iI.unsqueeze(1) + \
            filter2.unsqueeze(-1) * v_i

        dh = filter3

        # perform aggregation 
        dh_i = scatter_mean(dh, mapping, dim=0)
        dv_i = scatter_mean(dv, mapping, dim=0)
        
        return dh_i, dv_i 
    
    
