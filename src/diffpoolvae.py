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

    def forward(self, atoms_nodes, xyz, bond_edges):  

        bond_edges, _ = make_directed(bond_edges)
        h = self.atom_embed(atoms_nodes)

        for conv in self.update:
            dh = scatter_add(conv(h[bond_edges[:, 1]]), bond_edges[:, 0], dim=0)
            h = h + dh 
        h = self.cg_network(h)

        a = F.softmax(h, dim=-1)
        assignment_pad = torch.block_diag( *torch.split(a, batch['num_atoms'].tolist()) )
        cg_adj = assignment_pad[edge[:, 0]].t().matmul(assignment_pad[edge[:, 1]])
        atom2cgmap = assignment_pad.nonzero()

        assignment_pad = assignment_pad / assignment_pad.sum(0)[None, :]
        ciI_flat = assignment_pad[atom2cgmap[:,0], atom2cgmap[:,1]]
        cg_xyz = scatter_add(ciI_flat[..., None] * xyz[atom2cgmap[:, 0]], atom2cgmap[:, 1], dim=0)
    
        return h, cg_xyz, assignment_pad, cg_adj


 class DiffCGContracMessageBlock(nn.Module):
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
                h_i,
                v_i,
                r_iI,
                assignment_pad):

        
        mask_iI = assignment_pad.nonzero()

        
        d_iI = r_iI.pow(2).sum(-1).sqrt()
        unit = r_iI / d_iI.unsqueeze(-1)
        
        phi = contraction.inv_dense(h)
        w_s = contraction.dist_embed(d_iI)

        inv_out = phi[mask_iI[:, 1]] * w_s
        inv_out = inv_out.reshape(inv_out.shape[0], 3, -1)

        ciI_flat = assignment_pad[mask_iI[:,0], mask_iI[:,1]]
        
        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_iI = unit_add + split_0 * v_i[mask_iI[:,0]]
        delta_s_iI = split_1

        # scatter_mean weighed by assignment weights 
        dV = scatter_add(src=ciI_flat[..., None, None] *  delta_v_iI,
                                index=mask_iI[:,1],
                                dim=0)

        dH = scatter_add(src=ciI_flat[..., None] * delta_s_iI,
                                index=mask_iI[:,1],
                                dim=0)

        return dH, dV


class Enc(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_cg,
                 n_rbf,
                 cutoff):
        super().__init__()
        
        self.feat_dim = feat_dim
        
        self.h_embed = nn.Linear(n_cg, num_features)
        self.H_embed = nn.Linear(n_cg, num_features)

        self.DiffCGContracMessageBlock = DiffCGContracMessageBlock(feat_dim=feat_dim,
                                                                 activation=activation,
                                                                 n_rbf=n_rbf,
                                                                 cutoff=cutoff,
                                                                 dropout=0.0)
        self.update = conv.UpdateBlock(feat_dim=num_features,
                         activation=activation,
                         dropout=0.0)
        # update block
        
    def forward(self, h, H, cg_xyz, assignment_pad, cg_adj):
        
        h = self.h_embed(h)
        H = self.H_embed(H)
        
        V_I = torch.zeros(H.shape[0], self.feat_dim, 3).to(h.device)
        v_i = torch.zeros(z.shape[0], self.feat_dim, 3).to(h.device)
        
        mask_iI = assignment_pad.nonzero()
        r_iI = xyz[mask_iI[:, 0]] - cg_xyz[mask_iI[:, 1]]
        
        dH, dV = self.DiffCGContracMessageBlock(h, v_i, r_iI, assignment_pad)
        
        return H + dH, V_I+dV
