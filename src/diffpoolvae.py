import torch
from torch import nn
from conv import * 
from torch_scatter import scatter_mean, scatter_add

class DiffPoolVAE(nn.Module):
    def __init__(self, encoder, decoder, pooler):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.pooler = pooler 
        
    def forward(self, batch, tau):
    
        xyz = batch['xyz']        
        device = xyz.device
        
        z = batch['z'] # torch.ones_like( batch['z'] ) 
        nbr_list = batch['nbr_list']

        assign, assign_logits, h, H, adj, cg_xyz, soft_cg_adj = self.pooler(z, 
                                                                   batch['xyz'], 
                                                                   batch['bonds'], 
                                                                   tau=tau,
                                                                   gumbel=True)

        cg_adj = (soft_cg_adj > 0.01).to(torch.float).to(device)

        H, V = self.encoder(h, H, xyz, cg_xyz, assign, nbr_list, cg_adj)
        H, V = self.decoder(H, cg_adj, cg_xyz)

        dx = torch.einsum('bcfe,bac->bcfe', V[:, :, :z.shape[1], :], assign).sum(1)

        x_recon = torch.einsum('bce,bac->bae', cg_xyz, assign) + dx
        
        return xyz, x_recon, assign, adj

class CGpool(nn.Module):
    
    def __init__(self,
             n_conv,
             n_atom_basis,
             n_cgs,
             assign_logits=None):
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

        if assign_logits is not None: 
            self.assign_logits = nn.Parameter(assign_logits)
        else:
            self.assign_logits = None

    def forward(self, atoms_nodes, xyz, bonds, tau, gumbel=False):  

        h = self.atom_embed(atoms_nodes.to(torch.long))

        adj = torch.zeros(h.shape[0], h.shape[1], h.shape[1]).to(h.device)
        adj[bonds[:, 0], bonds[:,1], bonds[:,2]] = 1
        adj[bonds[:, 0], bonds[:,2], bonds[:,1]] = 1

        for conv in self.update:
            dh = torch.einsum('bif,bij->bjf', conv(h), adj)
            h = h + dh 

        if self.assign_logits is None:
            assign_logits = self.cg_network(h)

        else:
            nbatch = h.shape[0]
            assign_logits = torch.stack( [ self.assign_logits] * nbatch )

        if gumbel:
            assign = F.gumbel_softmax(assign_logits, tau=tau, dim=-1, hard=False)
        else:
            assign = F.softmax(assign_logits * (1/tau) , dim=-1) 

        assign_norm = assign / assign.sum(1).unsqueeze(-2) 

        H = torch.einsum('bnj,bnf->bjf', assign_norm, h)
        # get coordinates 
        cg_xyz = torch.einsum("bin,bij->bjn", xyz, assign_norm)

        Ncg = H.shape[1]

        # compute weighted adjacency 
        cg_adj = assign.transpose(1,2).matmul(adj).matmul(assign)

        cg_adj = cg_adj * (1 - torch.eye(Ncg, Ncg).to(h.device)).unsqueeze(0)

        return assign, assign_logits, h, H, adj, cg_xyz, cg_adj


class DenseContract(nn.Module):
    
    def __init__(self,
             feat_dim,
             n_rbf,
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
        self.offset = torch.linspace(0.0, self.cutoff, self.feat_dim)
        
    def forward(self, assign, h, v, cg_xyz, xyz, cg_adj):
        # pool atomwise message to CG bead 
        
        # xyz: b x n x 3
        # cg_xyz: b x N x 3 
        # h: b x n x f
        # v: b x n x f 
        # assign: b x n x N 
          
        r_iI = (xyz.unsqueeze(1) - cg_xyz.unsqueeze(2))
        d_iI = r_iI.pow(2).sum(-1).sqrt()
        unit_iI = r_iI / d_iI.unsqueeze(-1)
        
        
        #V_I = torch.zeros(H.shape[0], H.shape[1], H.shape[2], 3).to(h.device)
        
        r_iI = (xyz.unsqueeze(1) - cg_xyz.unsqueeze(2))
        d_iI = r_iI.pow(2).sum(-1).sqrt()
        unit = r_iI / d_iI.unsqueeze(-1)
        
        phi = self.inv_dense(h)
        expanded_dist = (-(d_iI.unsqueeze(-1) - self.offset.to(h.device)).pow(2)).exp()
        w_s = self.dist_filter(expanded_dist)

        shape = list(w_s.shape[:-1])
        
        # need to assign with assignment vector 
        filter_w = (w_s * phi.unsqueeze(1)).reshape(shape + [h.shape[-1], 3])
        
        split_0 = filter_w[..., 0].unsqueeze(-1)
        split_1 = filter_w[..., 1]
        split_2 = filter_w[..., 2].unsqueeze(-1)
        unit_add = split_2 * unit.unsqueeze(-2)
        
        dv_iI = unit_add + split_0 * v.unsqueeze(1)
        ds_iI = split_1

        dV = torch.einsum('bcafe,bac->bcfe', dv_iI, assign)
        dH = torch.einsum('bcaf,bac->bcf', ds_iI, assign)
        
        return dH, dV
    
    
class DenseEquiEncoder(nn.Module):
    
    def __init__(self,
             n_conv,
             n_atom_basis,
             n_rbf,
             activation,
             cutoff):
        super().__init__()

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
        
        self.contract = nn.ModuleList(
            [DenseContract(feat_dim=n_atom_basis,
                           activation=activation,
                           n_rbf=n_rbf,
                           cutoff=cutoff,
                           )
             for _ in range(n_conv)]
            )
        
        self.n_conv = n_conv

    def forward(self, h, H, xyz, cg_xyz, assign, nbr_list, cg_adj):
        
        # prepare inputs 
        
        h_shape = h.shape
        v_shape = list(h.shape) + [3]
        
        h_stack = h.reshape(-1, h.shape[-1])
        
        v_stack_shape = list(h_stack.shape) + [3]
        h_stack_shape = h_stack.shape 
        
        v_stack = torch.zeros(*v_stack_shape).to(h.device)
        
        pad_nbr_list = (nbr_list[:, 0] * h.shape[1]).unsqueeze(1) + nbr_list[:, 1:]

        r_ij = xyz[nbr_list[:, 0], nbr_list[:, 1]] - xyz[nbr_list[:, 0], nbr_list[:, 2]]
        
        
        # intialize H, V
        V = torch.zeros(list(H.shape) + [3]).to(H.device)
    
        for i in range(self.n_conv):
            ds_message, dv_message = self.message_blocks[i](s_j=h_stack,
                                                   v_j=v_stack,
                                                   r_ij=r_ij,
                                                   nbrs=pad_nbr_list)
            h_stack = h_stack + 0.5 * ds_message
            v_stack = v_stack + 0.5 * dv_message

            # # update block
            ds_update, dv_update = self.update_blocks[i](s_i=h_stack, v_i=v_stack)
            h_stack = h_stack + 0.5 *  ds_update # atom message 
            v_stack = v_stack + 0.5 *  dv_update
                    
            h_pad = h_stack.reshape(h_shape)
            v_pad = v_stack.reshape(v_shape)
            
            dH, dV = self.contract[i](assign, h_pad, v_pad, cg_xyz, xyz, cg_adj)
            
            V = V + dV 
            H = H + dH
            
        return H, V

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

class DenseEquivariantDecoder(nn.Module):
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv, activation, cross_flag=True, atomwise_z=False):   
        
        nn.Module.__init__(self)
        # distance transform
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                  cutoff=cutoff,
                                  feat_dim=n_atom_basis,
                                  dropout=0.0)

        if cross_flag:
            self.message_blocks = nn.ModuleList(
                [EquiMessageCross(feat_dim=n_atom_basis,
                              activation=activation,
                              n_rbf=n_rbf,
                              cutoff=cutoff,
                              dropout=0.0)
                 for _ in range(num_conv)]
            )
        else: 
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

    
    def forward(self, H, cg_adj, cg_xyz):
        
        H_stack = H.view(-1, H.shape[2])
        
        
        deg = cg_adj.sum(-1)
        deg_stack = deg.view(-1)
        deg_inv_sqrt = deg_stack.reciprocal().sqrt()
        
        cg_nbr_list = (cg_adj > 0.0).nonzero()
        pad_nbr_list = (cg_nbr_list[:, 0] * H.shape[1]).unsqueeze(1) + cg_nbr_list[:, 1:]
        edge_weights = cg_adj[cg_nbr_list[:,0], cg_nbr_list[:,1], cg_nbr_list[:,2]]
    
        r_ij = cg_xyz[cg_nbr_list[:, 0], cg_nbr_list[:, 2]] - cg_xyz[cg_nbr_list[:, 0], cg_nbr_list[:, 1]] 
        
        
        V_stack = torch.zeros(list(H_stack.shape) + [3]).to(H.device)
        
        # compute node degress 

        for i, message_block in enumerate(self.message_blocks):
            
            # message block
            dH_message, dV_message = message_block(s_j=H_stack,
                                                   v_j=V_stack,
                                                   r_ij=r_ij,
                                                   nbrs=pad_nbr_list,
                                                   # normalize by node degree
                                                   edge_wgt=deg_inv_sqrt[pad_nbr_list[:,0]] * deg_inv_sqrt[pad_nbr_list[:,1]]
                                                   )
            H_stack = H_stack + dH_message
            V_stack = V_stack + dV_message

            # update block
            dH_update, dV_update = self.update_blocks[i](s_i=H_stack,
                                                v_i=V_stack)
            H_stack = H_stack + dH_update
            V_stack = V_stack + dV_update
            
            H_unpack = H_stack.reshape(H.shape[0], H.shape[1], -1) 
            V_unpack = V_stack.reshape(H.shape[0], H.shape[1], H.shape[2], 3)

        return H_unpack, V_unpack 

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
