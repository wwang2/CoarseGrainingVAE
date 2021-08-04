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
