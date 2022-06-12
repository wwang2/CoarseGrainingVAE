import torch
from torch import nn 

from modules import * 

from torch_scatter import scatter_mean, scatter_add
from torch import nn
                                  
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
                 in_feat_dim,
                 out_feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 dropout):
        super().__init__()

        self.inv_dense = nn.Sequential(Dense(in_features=in_feat_dim,
                                          out_features=in_feat_dim,
                                          bias=True,
                                          dropout_rate=dropout,
                                          activation=to_module(activation)),
                                    Dense(in_features=in_feat_dim,
                                          out_features=out_feat_dim,
                                          bias=True,
                                          dropout_rate=dropout))

        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                        cutoff=cutoff,
                                        feat_dim=out_feat_dim,
                                        dropout=dropout)

        self.dist_filter = Dense(in_features=in_feat_dim,
              out_features=out_feat_dim,
              bias=True,
              dropout_rate=0.0)

        self.offset = torch.linspace(0.0, cutoff, in_feat_dim)

    def forward(self,
                s_j,
                dist,
                nbrs):

        phi = self.inv_dense(s_j)[nbrs[:, 1].cpu()]
        w_s = self.dist_embed(dist)
        # expanded_dist = (-(dist.unsqueeze(-1) - self.offset.to(phi.device)).pow(2)).exp()
        # w_s = self.dist_filter(expanded_dist)

        output = phi * w_s

        return output

class ENMessageBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 dropout):
        super().__init__()
        self.inv_message = InvariantMessage(in_feat_dim=feat_dim,
                                            out_feat_dim=2 * feat_dim, 
                                            activation=activation,
                                            n_rbf=n_rbf,
                                            cutoff=cutoff,
                                            dropout=dropout)

    def forward(self,
                s_j,
                v_j,
                r_ij,
                nbrs,
                edge_wgt=None):

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)

        inv_out = inv_out.reshape(inv_out.shape[0], 2, -1)
        equi_filter = inv_out[:, 0, :].unsqueeze(-1)
        inv_filter = inv_out[:, 1, :]

        delta_v_ij = equi_filter * unit.unsqueeze(1)
        delta_s_ij = inv_filter

        if edge_wgt is not None: 
            v_edge_wgt = edge_wgt[..., None, None]
            h_edge_wgt = edge_wgt[..., None]
        else:
            v_edge_wgt = 1 
            h_edge_wgt = 1

        graph_size = s_j.shape[0]
        delta_v_i = scatter_add(src=delta_v_ij * v_edge_wgt,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        delta_s_i = scatter_add(src=delta_s_ij * h_edge_wgt,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        return delta_s_i, delta_v_i


class InvMessageBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 dropout):
        super().__init__()
        self.inv_message = InvariantMessage(in_feat_dim=feat_dim,
                                            out_feat_dim=feat_dim, 
                                            activation=activation,
                                            n_rbf=n_rbf,
                                            cutoff=cutoff,
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

        delta_s_i = scatter_add(src=inv_out,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=s_j.shape[0])

        return delta_s_i, v_j

import sys
sys.path.append("../")
sys.path.append("../src")

from conv import * 

class EquiMessagePsuedo(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 dropout):
        super().__init__()
        self.inv_message = InvariantMessage(in_feat_dim=feat_dim,
                                            out_feat_dim=feat_dim * 9, 
                                            activation=activation,
                                            n_rbf=n_rbf,
                                            cutoff=cutoff,
                                            dropout=dropout)

    def forward(self,
                s_j,
                sbar_j,
                v_j,
                vbar_j, 
                r_ij, 
                nbrs, # directed edges that has both directions 
                edge_wgt=None):
        # edge_wgt the same size as 

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)

        inv_out = inv_out.reshape(inv_out.shape[0], 9, s_j.shape[1])

        split_0 = inv_out[:, 0, :] # update v
        split_1 = inv_out[:, 1, :].unsqueeze(-1)
        split_2 = inv_out[:, 2, :].unsqueeze(-1) # update e_ij
        split_3 = inv_out[:, 3, :].unsqueeze(-1) # update v cross vbar 
        split_4 = inv_out[:, 4, :].unsqueeze(-1) # update v cross v
        split_5 = inv_out[:, 5, :].unsqueeze(-1) # update vbar cross vbar 
        split_6 = inv_out[:, 6, :].unsqueeze(-1) # update vbar 
        split_7 = inv_out[:, 7, :].unsqueeze(-1) # update vbar cross vbar 
        split_8 = inv_out[:, 8, :].unsqueeze(-1) # update vbar 
        
        d_s_ij = split_0 * s_j[nbrs[:, 0]]
        d_sbar_ij = (v_j[nbrs[:, 0]] * vbar_j[nbrs[:, 1]]).sum(-1) # dot product between vbar and v 

        
        d_v_ij = split_1 * unit.unsqueeze(1) + \
                 split_2 * v_j[nbrs[:, 1]] + \
                 split_3 * torch.cross( v_j[nbrs[:, 0]],  vbar_j[nbrs[:, 1]]) + \
                 split_4 * sbar_j[nbrs[:, 0]].unsqueeze(-1) * vbar_j[nbrs[:,1]]

        d_vbar_ij = split_5 * vbar_j[nbrs[:, 1]] + \
                    split_6 * sbar_j[nbrs[:, 0]].unsqueeze(-1) * v_j[nbrs[:,1]]  + \
                    split_7 * torch.cross( v_j[nbrs[:, 0]],  v_j[nbrs[:, 1]]) + \
                    split_8 * torch.cross( vbar_j[nbrs[:, 0]],  vbar_j[nbrs[:, 1]]) 
        # add results from neighbors of each node
        
        graph_size = s_j.shape[0]
        dv = scatter_add(src=d_v_ij,
                                index=nbrs[:, 0],  
                                dim=0,
                                dim_size=graph_size)
        dvbar = scatter_add(src=d_vbar_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)
        
        dh = scatter_add(src=d_s_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)
        
        dhbar = scatter_add(src=d_sbar_ij,
                        index=nbrs[:, 0],
                        dim=0,
                        dim_size=graph_size)

        return dh, dhbar, dv, dvbar



class EquiMessagePsuedo2(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 dropout):
        super().__init__()
        self.inv_message = InvariantMessage(in_feat_dim=feat_dim,
                                            out_feat_dim=feat_dim * 4, 
                                            activation=activation,
                                            n_rbf=n_rbf,
                                            cutoff=cutoff,
                                            dropout=dropout)
        
        self.mix_dvbar = nn.Linear(feat_dim * 4, feat_dim, bias=False)
        self.mix_dv = nn.Linear(feat_dim * 3, feat_dim, bias=False)

    def forward(self,
                s_j,
                sbar_j,
                v_j,
                vbar_j, 
                r_ij, 
                nbrs, # directed edges that has both directions 
                edge_wgt=None):
        # edge_wgt the same size as 

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)

        inv_out = inv_out.reshape(inv_out.shape[0], 4, s_j.shape[1])

        # split invariant messages for different updates
        split_0 = inv_out[:, 0, :].unsqueeze(-1) # update v
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1) # update e_ij
        split_3 = inv_out[:, 3, :] # update e_ij
        
        # update ds using invariant message 
        d_s_ij = split_1
        
        # update dsbar 1 interactions 
        d_sbar_ij = split_3 * (v_j[nbrs[:, 0]] * vbar_j[nbrs[:, 1]]).sum(-1) # dot product between vbar and v 

        
        # update dvbar 4 interactions 
        dvbar_new1 = s_j.unsqueeze(-1)[nbrs[:, 0]] * vbar_j[nbrs[:, 1]]
        dvbar_new2 = sbar_j.unsqueeze(-1)[nbrs[:, 0]] * v_j[nbrs[:, 1]]
        dvbar_new3 = torch.cross( v_j[nbrs[:, 0]],  v_j[nbrs[:, 1]])
        dvbar_new4 = torch.cross( vbar_j[nbrs[:, 0]],  vbar_j[nbrs[:, 1]])
        dvbar_cat = torch.cat([dvbar_new1, dvbar_new2, dvbar_new3, dvbar_new4], dim=-2)
        
        # transform dvar
        dvbar = self.mix_dvbar(torch.transpose(dvbar_cat, -1, -2))
        dvbar_ij = torch.transpose(dvbar, -1, -2)

        # update dv_ij 3 interactions 
        dv_new1 = s_j[nbrs[:,0]].unsqueeze(-1) * v_j[nbrs[:,1]]
        dv_new2 = sbar_j[nbrs[:,0]].unsqueeze(-1) * vbar_j[nbrs[:,1]]
        dv_new3 = torch.cross(v_j[nbrs[:, 0]],  vbar_j[nbrs[:, 1]])
        dv_cat = torch.cat([dv_new1, dv_new2, dv_new3], dim=-2)
        
        dv_ij = self.mix_dv(torch.transpose(dv_cat, -1, -2))
        dvij = torch.transpose(dv_ij, -1, -2)
        
        d_v_ij = split_2 * unit.unsqueeze(1) + \
                 split_0 * v_j[nbrs[:, 1]] + dvij


        # add results from neighbors of each node
        
        graph_size = s_j.shape[0]
        dv = scatter_add(src=d_v_ij,
                                index=nbrs[:, 0],  
                                dim=0,
                                dim_size=graph_size)
        dvbar = scatter_add(src=dvbar_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)
        
        dh = scatter_add(src=d_s_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)
        
        dhbar = scatter_add(src=d_sbar_ij,
                        index=nbrs[:, 0],
                        dim=0,
                        dim_size=graph_size)

        return dh, dhbar, dv, dvbar


class EquiMessageCross(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 dropout):
        super().__init__()
        self.inv_message = InvariantMessage(in_feat_dim=feat_dim,
                                            out_feat_dim=feat_dim * 4, 
                                            activation=activation,
                                            n_rbf=n_rbf,
                                            cutoff=cutoff,
                                            dropout=dropout)

    def forward(self,
                s_j,
                v_j,
                r_ij,
                nbrs,
                edge_wgt=None):
        # edge_wgt the same size as 

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)

        inv_out = inv_out.reshape(inv_out.shape[0], 4, s_j.shape[1])

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)
        split_3 = inv_out[:, 3, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_ij = unit_add + split_0 * v_j[nbrs[:, 1].cpu()] + split_3 * torch.cross( v_j[nbrs[:, 0]],  v_j[nbrs[:, 1].cpu()])
        delta_s_ij = split_1

        # add results from neighbors of each node
        
        if edge_wgt is not None: 
            v_edge_wgt = edge_wgt[..., None, None]
            h_edge_wgt = edge_wgt[..., None]
        else:
            v_edge_wgt = 1 
            h_edge_wgt = 1

        graph_size = s_j.shape[0]
        dv = scatter_add(src=delta_v_ij * v_edge_wgt,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        dh = scatter_add(src=delta_s_ij * h_edge_wgt,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        return dh, dv


class DenseEquiMessageBlock(nn.Module):
    def __init__(self,
                 num_features,
                 activation,
                 n_rbf,
                 cutoff):
        super().__init__()
        
        self.inv_dense = nn.Sequential(Dense(in_features=num_features,
                                                  out_features=num_features,
                                                  bias=True,
                                                  dropout_rate=0.0,
                                                  activation=to_module(activation)),
                                            Dense(in_features=num_features,
                                                  out_features=num_features,
                                                  bias=True,
                                                  dropout_rate=0.0))

        self.dist_filter = nn.Sequential( Dense(in_features=num_features,
                                                  out_features=num_features,
                                                  bias=True,
                                                  dropout_rate=0.0,
                                                  activation=to_module(activation)),
                                            Dense(in_features=num_features,
                                                  out_features=num_features,
                                                  bias=True,
                                                  dropout_rate=0.0))
        self.update_filter = nn.Sequential( Dense(in_features=num_features,
                                          out_features=num_features,
                                          bias=True,
                                          dropout_rate=0.0,
                                          activation=to_module(activation)),
                                    Dense(in_features=num_features,
                                          out_features=3 * num_features,
                                          bias=True,
                                          dropout_rate=0.0))

        self.layer_norm = nn.LayerNorm(num_features)
        self.offset = torch.linspace(0.0, cutoff, num_features)
        
    def forward(self, h, v, adj, xyz):
        
        R_IJ = xyz[:, None, :, :] - xyz[:, :, None, :] 
        d_IJ = R_IJ.pow(2).sum(-1).sqrt()
        unit_IJ = R_IJ / (d_IJ + torch.diag(torch.ones(h.shape[1]))[None, ...]).unsqueeze(-1)
        
        phi = self.inv_dense(h)

        expanded_dist = (-(d_IJ.unsqueeze(-1) - self.offset).pow(2)).exp()
        w = self.dist_filter(expanded_dist)
        shape = list(w.shape[:-1])

        #print("phi_wegiht" ,phi.abs().mean())

        #print("w_wegiht" ,w.abs().mean())
        
        filter_w = self.update_filter((w * phi.unsqueeze(1) * phi.unsqueeze(2))).reshape(shape + [h.shape[-1], 3])

        #print("filter_w_wegiht" ,filter_w.abs().mean())

        filter_r = filter_w[..., 0] * adj.unsqueeze(-1)
        filter_v = filter_w[..., 1] * adj.unsqueeze(-1)
        filter_h = filter_w[..., 2] * adj.unsqueeze(-1)

        #import ipdb;ipdb.set_trace()

        dv_rcontribution = (filter_r.unsqueeze(-1) * unit_IJ.unsqueeze(-2)).sum(1)
        #dv_rcontribution = (filter_r.unsqueeze(-1) * unit_IJ.unsqueeze(-2)).sum(1) + (filter_r.unsqueeze(-1) * unit_IJ.unsqueeze(-2)).sum(2)
        dv_vcontribution = torch.einsum("bijf,bifv->bjfv", filter_v.squeeze() ,v) #+ torch.einsum("bijf,bjfv->bifv", filter_v.squeeze() ,v) 
        
        # todo add cross product contribution 
        dv = dv_rcontribution + dv_vcontribution
        dh = torch.einsum('bijf,bjf->bjf', filter_h, h)
        
        #print("h_wegiht" ,filter_h.abs().mean())
        #import ipdb ;ipdb.set_trace()

        dh = self.layer_norm(dh)

        return dh, dv 


class EquiMessageBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 dropout):
        super().__init__()
        self.inv_message = InvariantMessage(in_feat_dim=feat_dim,
                                            out_feat_dim=feat_dim * 3, 
                                            activation=activation,
                                            n_rbf=n_rbf,
                                            cutoff=cutoff,
                                            dropout=dropout)

        self.h_att = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim))
        self.v_att = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim))

    def forward(self,
                s_j,
                v_j,
                r_ij,
                nbrs,
                edge_wgt=None):

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)
        graph_size = s_j.shape[0]

        inv_out = inv_out.reshape(inv_out.shape[0], 3, s_j.shape[-1])

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_ij = unit_add + split_0 * v_j[nbrs[:, 1].cpu()]
        delta_s_ij = split_1

        if edge_wgt is not None: 
            v_edge_wgt = edge_wgt[..., None, None]
            h_edge_wgt = edge_wgt[..., None]
        else:
            v_edge_wgt = 1 
            h_edge_wgt = 1

        # # add results from neighbors of each node

        # h_att_wgt = torch.exp(-self.h_att(s_j[nbrs[:, 1].cpu()]) )
        # h_att_norm = scatter_add(h_att_wgt, 
        #                         index=nbrs[:, 0],
        #                          dim=0,
        #                         dim_size=graph_size)

        # h_att_norm_wgt = h_att_wgt / h_att_norm[nbrs[:, 0]]

        # v_att_wgt = torch.exp(-self.v_att(s_j[nbrs[:, 1].cpu()]) )
        # v_att_norm = scatter_add(v_att_wgt, 
        #                         index=nbrs[:, 0],
        #                          dim=0,
        #                         dim_size=graph_size)

        # v_att_norm_wgt = v_att_wgt / v_att_norm[nbrs[:, 0]]

        delta_v_i = scatter_add(src=delta_v_ij * v_edge_wgt, #* v_att_norm_wgt.unsqueeze(-1),
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        delta_s_i = scatter_add(src=delta_s_ij * h_edge_wgt, #* h_att_norm_wgt,
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

        v_tranpose = v_i.transpose(1, 2).reshape(-1, v_i.shape[1])

        num_feats = v_i.shape[1]
        u_v = (self.u_mat(v_tranpose).reshape(-1, 3, num_feats)
               .transpose(1, 2))
        v_v = (self.v_mat(v_tranpose).reshape(-1, 3, num_feats)
               .transpose(1, 2))

        v_v_norm = ((v_v ** 2 + 1e-10).sum(-1)) ** 0.5
        s_stack = torch.cat([s_i, v_v_norm], dim=-1)

        split = self.s_dense(s_stack).reshape(s_i.shape[0], 3, -1)

        # delta v update
        a_vv = split[:, 0, :].unsqueeze(-1)
        delta_v_i = u_v * a_vv

        # delta s update
        a_sv = split[:, 1, :]
        a_ss = split[:, 2, :]

        inner = (u_v * v_v).sum(-1)
        delta_s_i = inner * a_sv + a_ss

        return delta_s_i, delta_v_i


class PseudoUpdateBlock(nn.Module):
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
        self.s_dense = nn.Sequential(Dense(in_features=2 * feat_dim,
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

        v_tranpose = v_i.transpose(1, 2).reshape(-1, v_i.shape[1])

        num_feats = v_i.shape[1]
        u_v = (self.u_mat(v_tranpose).reshape(-1, 3, num_feats)
               .transpose(1, 2))
        v_v = (self.v_mat(v_tranpose).reshape(-1, 3, num_feats)
               .transpose(1, 2))

        v_v_norm = ((v_v ** 2 + 1e-10).sum(-1)) ** 0.5

        vector = (s_i.unsqueeze(-1) * v_i ) # product of pseudovector and pseudoscalar is vector 
        vector_norm = ((vector ** 2 + 1e-10).sum(-1)) ** 0.5

        s_stack = torch.cat([vector_norm, v_v_norm], dim=-1)

        split = self.s_dense(s_stack).reshape(s_i.shape[0], 3, -1)

        # pseduovector update 
        a_vv = split[:, 0, :].unsqueeze(-1)
        delta_v_i = u_v * a_vv

        # pseduoscalar update 
        a_sv = split[:, 1, :] * s_i
        a_ss = split[:, 2, :] * s_i 

        inner = (u_v * v_v).sum(-1)
        delta_s_i = inner * a_sv + a_ss

        return delta_s_i, delta_v_i



class ContractiveMessageBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
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
                                        feat_dim=3 * feat_dim,
                                        dropout=dropout)

    def forward(self,
                s_i,
                v_i,
                r_iI,
                mapping):

        dist, unit = preprocess_r(r_iI)
        phi = self.inv_dense(s_i)
        
        w_s = self.dist_embed(dist)
            
        inv_out = phi * w_s
        inv_out = inv_out.reshape(inv_out.shape[0], 3, -1)

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_iI = unit_add + split_0 * v_i
        delta_s_iI = split_1

        delta_v_I = scatter_add(src=delta_v_iI,
                                index=mapping,
                                dim=0)

        delta_s_I = scatter_add(src=delta_s_iI,
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
                                        feat_dim=3 * feat_dim,
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
        edge_inv = (phi[nbrs[:, 1].cpu()] *  phi[nbrs[:, 0]]) * self.dist_embed(d_ij)
        #filter1, filter2, filter3 = self.edgefilters(edge_inv)
        
        edge_inv = edge_inv.reshape(edge_inv.shape[0], 3, -1)

        import ipdb; ipdb.set_trace()

        filter1 = edge_inv[:, 0 ]
        filter2 = edge_inv[:, 1 ]
        filter3 = edge_inv[:, 2 ]

        dv = filter1.unsqueeze(-1) * unit_r_ij.unsqueeze(1) + \
            filter2.unsqueeze(-1) * v_i[nbrs[:, 1].cpu()]

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
                                        feat_dim=3 * feat_dim,
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
    
    
