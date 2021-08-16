import torch
from torch import nn
from conv import * 
from torch_scatter import scatter_mean, scatter_add

class CGpool(nn.Module):
    
    def __init__(self,
             n_conv,
             n_atom_basis,
             n_cgs):
        super().__init__()

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)

        self.update = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_atom_basis, n_atom_basis), 
                           nn.Tanh(), 
                           nn.Linear(n_atom_basis, n_atom_basis))
             for _ in range(n_conv)
            ]
        )
        
        self.cg_network = nn.Sequential(nn.Linear(n_atom_basis, n_atom_basis), 
                                       nn.Tanh(), 
                                       nn.Linear(n_atom_basis, n_cgs))

    def forward(self, atoms_nodes, xyz, bonds, tau):  

        h = self.atom_embed(atoms_nodes.to(torch.long))

        adj = torch.zeros(h.shape[0], h.shape[1], h.shape[1])
        adj[bonds[:, 0], bonds[:,1], bonds[:,2]] = 1
        adj[bonds[:, 0], bonds[:,2], bonds[:,1]] = 1

        for conv in self.update:
            dh = torch.einsum('bif,bij->bjf', conv(h), adj)
            h = h + dh 

        assign_logits = self.cg_network(h)
        assign = F.gumbel_softmax(assign_logits, tau=tau, dim=-1, hard=False)

        assign_norm = assign / assign.sum(1).unsqueeze(-2) 

        H = torch.einsum('bnj,bnf->bjf', assign_norm, h)
        # get coordinates 
        cg_xyz = torch.einsum("bin,bij->bjn", xyz, assign_norm)

        Ncg = H.shape[1]

        # compute weighted adjacency 
        cg_adj = assign.transpose(1,2).matmul(adj).matmul(assign)

        cg_adj = cg_adj * (1 - torch.eye(Ncg, Ncg)).unsqueeze(0)

        return assign, assign_logits, h, H, cg_xyz, cg_adj


class Enc(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 cutoff):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.dist_filter = Dense(in_features=feat_dim,
                      out_features=feat_dim * 3,
                      bias=True,
                      dropout_rate=0.0)
        
        self.inv_dense = nn.Sequential(Dense(in_features=feat_dim,
                                  out_features=feat_dim,
                                  bias=True,
                                  dropout_rate=0.0,
                                  activation=to_module(activation)),
                            Dense(in_features=feat_dim,
                                  out_features=3 * feat_dim,
                                  bias=True,
                                  dropout_rate=0.0))
        
        self.cutoff = cutoff
        
    def forward(self, assign, h, H, cg_xyz, xyz, cg_adj):
        
        V_I = torch.zeros(H.shape[0], H.shape[1], H.shape[2], 3).to(h.device)
        v_i = torch.zeros(h.shape[0], h.shape[1], H.shape[2], 3).to(h.device)
        
        r_iI = (xyz.unsqueeze(1) - cg_xyz.unsqueeze(2))
        d_iI = r_iI.pow(2).sum(-1).sqrt()
        unit = r_iI / d_iI.unsqueeze(-1)
        offset = torch.linspace(0.0, self.cutoff, self.feat_dim)
        
        phi = self.inv_dense(h)
        expanded_dist = (-(d_iI.unsqueeze(-1) - offset).pow(2)).exp()
        w_s = self.dist_filter(expanded_dist)

        shape = list(w_s.shape[:-1])
        
        # is this correct? 
        filter_w = (w_s * phi.unsqueeze(1)).reshape(shape + [h.shape[-1], 3])
        
        split_0 = filter_w[..., 0].unsqueeze(-1)
        split_1 = filter_w[..., 1]
        split_2 = filter_w[..., 2].unsqueeze(-1)
        
        unit_add = split_2 * unit.unsqueeze(-2)
        dv_iI = unit_add + split_0 * v_i.unsqueeze(1)
        ds_iI = split_1

        dV = dv_iI.sum(2)
        dH = ds_iI.sum(2)

        return H + dH, V_I+dV

class DiffPoolDecoder(nn.Module):
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv, activation):   
        
        nn.Module.__init__(self)
        # distance transform
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                  cutoff=cutoff,
                                  feat_dim=n_atom_basis,
                                  dropout=0.0)

        self.message_blocks = nn.ModuleList(
            [DiffpoolMessageBlock(feat_dim=n_atom_basis,
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
    
    def forward(self, cg_xyz, H, cg_adj):
        
        CG_nbr_list = cg_adj.nonzero()
        #CG_nbr_list, _ = make_directed(CG_nbr_list)
        r_ij = cg_xyz[CG_nbr_list[:, 1]] - cg_xyz[CG_nbr_list[:, 0]]
        
        V = torch.zeros(H.shape[0], H.shape[1], 3 ).to(H.device)

        for i, message_block in enumerate(self.message_blocks):
            
            # message block
            dH_message, dV_message = message_block(s_j=H,
                                                   v_j=V,
                                                   r_ij=r_ij,
                                                   nbrs=CG_nbr_list,
                                                   cg_adj=cg_adj # contains the weighted edges
                                                   )
            H = H + dH_message
            V = V + dV_message

            # update block
            dH_update, dV_update = self.update_blocks[i](s_i=H,
                                                v_i=V)
            H = H + dH_update
            V = V + dV_update

        return H, V 


class DiffpoolMessageBlock(nn.Module):
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
                cg_adj):

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)

        inv_out = inv_out.reshape(inv_out.shape[0], 4, -1)

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)
        split_3 = inv_out[:, 3, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        dv_ij = unit_add + split_0 * v_j[nbrs[:, 1]] + split_3 * torch.cross(v_j[nbrs[:, 0]], 
                                                                                  v_j[nbrs[:, 1]])
        ds_ij = split_1
        
        graph_size = s_j.shape[0]
        dv_i = scatter_add(src=dv_ij * cg_adj[nbrs[:,0], nbrs[:,1]][..., None, None],
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        ds_i = scatter_add(src=ds_ij * cg_adj[nbrs[:,0], nbrs[:,1]][..., None],
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        return ds_i, dv_i
