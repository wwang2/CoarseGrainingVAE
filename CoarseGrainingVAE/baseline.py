import torch
from torch import nn
from .conv import * 
from .modules import * 
from torch_scatter import scatter_mean, scatter_add


class Baseline(nn.Module):
    def __init__(self, pooler, n_cgs, n_atoms):
        nn.Module.__init__(self)
        self.pooler = pooler 
        self.B = nn.Parameter(0.01 * torch.randn(n_cgs, n_atoms))
        
    def forward(self, batch):
    
        xyz = batch['xyz']        
        device = xyz.device
        
        z = batch['z'] # torch.ones_like( batch['z'] ) 
        nbr_list = batch['nbr_list']

        soft_assign, assign_norm, h, H, adj, cg_xyz, soft_cg_adj, knbrs = self.pooler(z, 
                                                                   batch['xyz'], 
                                                                   batch['bonds'], 
                                                                   tau=0.0,
                                                                   gumbel=True)

        # recenter xyz and CG xyz 
        shift = xyz.mean(1).unsqueeze(1)

        xyz = xyz - shift 
        cg_xyz = cg_xyz - shift

        x_recon = torch.einsum("bce,ca->bae", cg_xyz, self.B)
        
        return soft_assign, xyz, x_recon


class settransform(nn.Module):
    def __init__(self, K, activation):
        nn.Module.__init__(self)
        self.mlp = torch.nn.Sequential(nn.Linear(K, K), to_module(activation), nn.Linear(K, K))
        self.update = torch.nn.Sequential(nn.Linear(K, K), to_module(activation), nn.Linear(K, K))
  
    def forward(self, edgeset):
        update = self.mlp(edgeset)
        CGcontract = update.mean(-2)
        output = self.update(update + CGcontract.unsqueeze(-2))
        # this is permutational equivariant 
        return output 
        
class edgesetMLP(nn.Module):
    def __init__(self, pooler, n_cgs, n_atoms, knn, depth, feature_dim, cutoff, activation):
        nn.Module.__init__(self)
        self.smear = GaussianSmearing(0, cutoff, feature_dim)      
        self.layers = nn.ModuleList(
            [settransform(K=feature_dim, activation=activation) for _ in range(depth)])
        
        self.pooler = pooler
        self.knn = knn
        self.n_cgs = n_cgs
        self.n_atoms = n_atoms
        self.decode = nn.Sequential(nn.Linear(feature_dim, feature_dim),
                                    to_module(activation),
                                    nn.Linear(feature_dim, n_atoms ))
        
    def forward(self, batch):
        
        xyz = batch['xyz']        
        device = xyz.device
        
        z = batch['z'] # torch.ones_like( batch['z'] ) 
        nbr_list = batch['nbr_list']

        soft_assign, assign_norm, h, H, adj, cg_xyz, soft_cg_adj, knbrs = self.pooler(z, 
                                                                   batch['xyz'], 
                                                                   batch['bonds'], 
                                                                   tau=0.0,
                                                                   gumbel=True)
        # get 
        dist = (cg_xyz.unsqueeze(-2) - cg_xyz.unsqueeze(-3)).pow(2).sum(-1).sqrt()
        value, knbrs = dist.sort(dim=-1, descending=False)
        knbrs = knbrs.cpu()
        value = value.cpu()

        value[:,:, self.knn+1:] = 0.0
        cg_nbr_list = value.nonzero()
        pad_cg_xyz = cg_xyz.reshape(-1, 3)
        pad_cg_nbr_list = cg_nbr_list[:, 1:] + (cg_nbr_list[:, 0] * cg_xyz.shape[1]).unsqueeze(-1)
        dist_vec = pad_cg_xyz[pad_cg_nbr_list[:,1]] - pad_cg_xyz[pad_cg_nbr_list[:,0]]
        dist_vec = dist_vec.reshape(cg_xyz.shape[0],  self.n_cgs * self.knn, 3)
        dist = dist_vec.pow(2).sum(-1).sqrt().reshape(cg_xyz.shape[0], self.n_cgs, self.knn, 1)
        
        # perm. equivariant update 
        output= self.smear(dist)
        for module in self.layers:
            output = module(output)
        coeffs = self.decode(output).reshape(-1, self.n_cgs * self.knn, self.n_atoms)
        
        dx_recon = torch.einsum("bio,bin->bon", coeffs, dist_vec)
        cg_offset = torch.einsum("bin,bij->bjn", dx_recon, assign_norm)
        cg_offset_lift = cg_offset[:, self.pooler.assign_idx, :]

        xyz_recon = cg_xyz[:, self.pooler.assign_idx, :] - cg_offset_lift + dx_recon

        return soft_assign, xyz, xyz_recon


class MLP(nn.Module):
    def __init__(self, pooler, n_cgs, n_atoms, width=1, depth=1, activation='ReLU'):
        nn.Module.__init__(self)
        self.pooler = pooler 

        self.input_dim = n_cgs * 3 
        self.output_dim = n_atoms * 3 
        self.layer_width = self.output_dim * width


        self.n_cgs = n_cgs 
        self.n_atoms = n_atoms


        layer_list = [ torch.nn.Linear(self.input_dim, self.layer_width ) ] + \
                    [to_module(activation), torch.nn.Linear(self.layer_width, self.layer_width)] * depth + \
                    [to_module(activation), torch.nn.Linear(self.layer_width, self.output_dim)]

        self.mlp = torch.nn.Sequential(*layer_list)
        
    def forward(self, batch):
    
        xyz = batch['xyz']        
        device = xyz.device
        
        z = batch['z'] # torch.ones_like( batch['z'] ) 
        nbr_list = batch['nbr_list']

        soft_assign, assign_norm, h, H, adj, cg_xyz, soft_cg_adj, knbrs = self.pooler(z, 
                                                                   batch['xyz'], 
                                                                   batch['bonds'], 
                                                                   tau=0.0,
                                                                   gumbel=True)

        x_recon = self.mlp(cg_xyz.reshape(-1, self.n_cgs * 3))

        x_recon = x_recon.reshape(-1, self.n_atoms, 3)
        
        return soft_assign, xyz, x_recon


class EquiMLP(nn.Module):
    def __init__(self, pooler, n_cgs, n_atoms, width=1, depth=1, activation='ReLU', knn=3):
        nn.Module.__init__(self)
        self.pooler = pooler 

        self.knn = knn 

        self.input_dim = 1
        self.output_dim = n_atoms 
        self.layer_width = self.output_dim * width

        self.n_cgs = n_cgs 
        self.n_atoms = n_atoms

        layer_list = [ torch.nn.Linear(self.input_dim, self.layer_width ) ] + \
                    [to_module(activation), torch.nn.Linear(self.layer_width, self.layer_width)] * depth + \
                    [to_module(activation), torch.nn.Linear(self.layer_width, self.output_dim)]

        self.mlp = torch.nn.Sequential(*layer_list)
        
    def forward(self, batch):
    
        xyz = batch['xyz']        
        device = xyz.device
        
        z = batch['z'] # torch.ones_like( batch['z'] ) 
        nbr_list = batch['nbr_list']

        soft_assign, assign_norm, h, H, adj, cg_xyz, soft_cg_adj, knbrs = self.pooler(z, 
                                                                   batch['xyz'], 
                                                                   batch['bonds'], 
                                                                   tau=0.0,
                                                                   gumbel=True)

        
        # get 
        dist = (cg_xyz.unsqueeze(-2) - cg_xyz.unsqueeze(-3)).pow(2).sum(-1).sqrt()
        value, knbrs = dist.sort(dim=-1, descending=False)
        knbrs = knbrs.cpu()
        value = value.cpu()

        value[:,:, self.knn+1:] = 0.0
        cg_nbr_list = value.nonzero()
        pad_cg_xyz = cg_xyz.reshape(-1, 3)
        pad_cg_nbr_list = cg_nbr_list[:, 1:] + (cg_nbr_list[:, 0] * cg_xyz.shape[1]).unsqueeze(-1)
        dist_vec = pad_cg_xyz[pad_cg_nbr_list[:,1]] - pad_cg_xyz[pad_cg_nbr_list[:,0]]
        dist_vec = dist_vec.reshape(cg_xyz.shape[0],  self.n_cgs * self.knn, 3)

        dist = dist_vec.pow(2).sum(-1).sqrt().reshape(cg_xyz.shape[0], self.n_cgs * self.knn, 1)

        coeffs = self.mlp(dist).reshape(cg_xyz.shape[0], self.n_cgs * self.knn, self.n_atoms)

        dx_recon = torch.einsum("bio,bin->bon", coeffs, dist_vec)

       # coeffs = self.mlp(dist).reshape(cg_xyz.shape[0], self.n_cgs, self.knn, self.k)

        # recentering 
        cg_offset = torch.einsum("bin,bij->bjn", dx_recon, assign_norm)
        cg_offset_lift = cg_offset[:, self.pooler.assign_idx, :]

        xyz_recon = cg_xyz[:, self.pooler.assign_idx, :] - cg_offset_lift + dx_recon
        
        return soft_assign, xyz, xyz_recon

class EquiMLP2(nn.Module):
    def __init__(self, pooler, n_cgs, n_atoms, width=1, depth=1, activation='ReLU', knn=3):
        nn.Module.__init__(self)
        self.pooler = pooler 

        self.knn = knn 

        self.input_dim = int( n_cgs  * self.knn)
        self.output_dim = self.input_dim * n_atoms
        self.layer_width = self.output_dim * width
        
        print("input dim: {}".format(self.input_dim)) 
        print("output dim: {}".format(self.output_dim))

        self.n_cgs = n_cgs 
        self.n_atoms = n_atoms


        layer_list = [ torch.nn.Linear(self.input_dim, self.layer_width ) ] + \
                    [to_module(activation), torch.nn.Linear(self.layer_width, self.layer_width)] * depth + \
                    [to_module(activation), torch.nn.Linear(self.layer_width, self.output_dim)]

        self.mlp = torch.nn.Sequential(*layer_list)
        
    def forward(self, batch):
    
        xyz = batch['xyz']        
        device = xyz.device
        
        z = batch['z'] # torch.ones_like( batch['z'] ) 
        nbr_list = batch['nbr_list']

        soft_assign, assign_norm, h, H, adj, cg_xyz, soft_cg_adj, knbrs = self.pooler(z, 
                                                                   batch['xyz'], 
                                                                   batch['bonds'], 
                                                                   tau=0.0,
                                                                   gumbel=True)

        
        # get 
        dist = (cg_xyz.unsqueeze(-2) - cg_xyz.unsqueeze(-3)).pow(2).sum(-1).sqrt()
        value, knbrs = dist.sort(dim=-1, descending=False)
        knbrs = knbrs.cpu()
        value = value.cpu()

        value[:,:, self.knn+1:] = 0.0
        cg_nbr_list = value.nonzero()
        pad_cg_xyz = cg_xyz.reshape(-1, 3)
        pad_cg_nbr_list = cg_nbr_list[:, 1:] + (cg_nbr_list[:, 0] * cg_xyz.shape[1]).unsqueeze(-1)
        dist_vec = pad_cg_xyz[pad_cg_nbr_list[:,1]] - pad_cg_xyz[pad_cg_nbr_list[:,0]]
        dist_vec = dist_vec.reshape(cg_xyz.shape[0],  self.n_cgs * self.knn, 3)

        dist = dist_vec.pow(2).sum(-1).sqrt().reshape(cg_xyz.shape[0], self.knn * self.n_cgs)

        coeffs = self.mlp(dist).reshape(cg_xyz.shape[0], self.n_cgs * self.knn, self.n_atoms)

        dx_recon = torch.einsum("bio,bin->bon", coeffs, dist_vec)

       # coeffs = self.mlp(dist).reshape(cg_xyz.shape[0], self.n_cgs, self.knn, self.k)

        # recentering 
        cg_offset = torch.einsum("bin,bij->bjn", dx_recon, assign_norm)
        cg_offset_lift = cg_offset[:, self.pooler.assign_idx, :]

        xyz_recon = cg_xyz[:, self.pooler.assign_idx, :] - cg_offset_lift + dx_recon
        
        return soft_assign, xyz, xyz_recon


class SetEquiMLP(nn.Module):
    def __init__(self, pooler, n_cgs, n_atoms, width=1, depth=1, activation='ReLU', knn=3):
        nn.Module.__init__(self)
        self.pooler = pooler 

        self.n_cgs = n_cgs 
        self.n_atoms = n_atoms
        self.knn = knn 

        if self.knn > n_cgs - 1:
            self.knn = n_cgs - 1 

        # get max per n_cg 
        assign_idx = self.pooler.assign_idx
        mode_val = torch.mode(assign_idx)[0]
        self.k = (assign_idx == mode_val).sum().item()

        self.input_dim = n_cgs * self.knn
        self.output_dim = n_cgs * self.knn * self.k # self.n_atoms
        # self.input_dim = int( n_cgs  * (n_cgs - 1) / 2 )
        # self.output_dim = self.input_dim * n_atoms
        self.layer_width = self.output_dim * width

        print("input dim: {}".format(self.input_dim)) 
        print("output dim: {}".format(self.output_dim))        

        layer_list = [ torch.nn.Linear(self.input_dim, self.layer_width ) ] + \
                    [to_module(activation), torch.nn.Linear(self.layer_width, self.layer_width)] * depth + \
                    [to_module(activation), torch.nn.Linear(self.layer_width, self.output_dim)]

        self.mlp = torch.nn.Sequential(*layer_list)

    def CG2ChannelIdx(self, CG_mapping):

        CG2atomChannel = torch.zeros_like(CG_mapping).to("cpu")

        for cg_type in torch.unique(CG_mapping): 
            cg_filter = CG_mapping == cg_type
            num_contri_atoms = cg_filter.sum().item()
            CG2atomChannel[cg_filter] = torch.LongTensor(list(range(num_contri_atoms)))#.to(CG_mapping.device)
            
        return CG2atomChannel.detach()
        
    def forward(self, batch):
    
        xyz = batch['xyz']        
        device = xyz.device
        
        z = batch['z'] # torch.ones_like( batch['z'] ) 
        nbr_list = batch['nbr_list']

        soft_assign, assign_norm, h, H, adj, cg_xyz, soft_cg_adj, knbrs = self.pooler(z, 
                                                                   batch['xyz'], 
                                                                   batch['bonds'], 
                                                                   tau=0.0,
                                                                   gumbel=True)

        # get 
        dist = (cg_xyz.unsqueeze(-2) - cg_xyz.unsqueeze(-3)).pow(2).sum(-1).sqrt()
        value, knbrs = dist.sort(dim=-1, descending=False)
        knbrs = knbrs.cpu()
        value = value.cpu()

        value[:,:, self.knn+1:] = 0.0
        cg_nbr_list = value.nonzero()
        pad_cg_xyz = cg_xyz.reshape(-1, 3)
        pad_cg_nbr_list = cg_nbr_list[:, 1:] + (cg_nbr_list[:, 0] * cg_xyz.shape[1]).unsqueeze(-1)
        dist_vec = pad_cg_xyz[pad_cg_nbr_list[:,1]] - pad_cg_xyz[pad_cg_nbr_list[:,0]]
        dist_vec = dist_vec.reshape(cg_xyz.shape[0],  self.n_cgs, self.knn, 3)

        dist = dist_vec.pow(2).sum(-1).sqrt().reshape(cg_xyz.shape[0], self.knn * self.n_cgs)

        coeffs = self.mlp(dist).reshape(cg_xyz.shape[0], self.n_cgs, self.knn, self.k)

        #import ipdb;ipdb.set_trace()

        dx = torch.einsum("bnkj, bnki->bnji", coeffs, dist_vec) # get array of size B x N x k x 3

        dx_recon =  dx[:, self.pooler.assign_idx, self.CG2ChannelIdx(self.pooler.assign_idx), :] #dx.reshape(cg_xyz.shape[0], -1, 3)[:, :self.pooler.assign_idx.shape[0], :]

        #dx_recon = torch.einsum("bio,bin->bon", coeffs, dist_vec)
 

        # basis = cg_xyz.unsqueeze(1) - cg_xyz.unsqueeze(2)

        # dist = basis.pow(2).sum(-1).sqrt()
        # triu_indx = torch.ones_like(dist[0]).triu(diagonal=1).nonzero()

        # dist_input = dist[:, triu_indx[:,0 ], triu_indx[:,1]]
        # basis_triu = basis[:,triu_indx[:,0 ], triu_indx[:,1] ]

        # coeffs = self.mlp(dist_input).reshape(dist_input.shape[0], self.n_atoms, self.input_dim)

        # dx_recon = torch.einsum("ije,inj->ine", basis_triu, coeffs )

        # recentering 
        cg_offset = torch.einsum("bin,bij->bjn", dx_recon, assign_norm)
        cg_offset_lift = cg_offset[:, self.pooler.assign_idx, :]

        xyz_recon = cg_xyz[:, self.pooler.assign_idx, :] - cg_offset_lift + dx_recon
        
        return soft_assign, xyz, xyz_recon


class EquiLinear(nn.Module):
    def __init__(self, pooler, n_cgs, n_atoms, cross, knn):
        nn.Module.__init__(self)
        self.pooler = pooler 
        self.knn = knn
        # if cross:
        #     self.B = nn.Parameter(0.01 * torch.randn(n_atoms, n_cgs ** 2 + (n_cgs ** 2)**2 ) )
        # else:

        self.knn = knn
        self.n_cgs = n_cgs
        self.B = nn.Parameter(0.01 * torch.randn(n_atoms, n_cgs * knn ) )

        self.cross = cross
        
    def forward(self, batch):
    
        xyz = batch['xyz']        
        device = xyz.device
        
        z = batch['z'] # torch.ones_like( batch['z'] ) 
        nbr_list = batch['nbr_list']

        soft_assign, assign_norm, h, H, adj, cg_xyz, soft_cg_adj, knbrs = self.pooler(z, 
                                                                   batch['xyz'], 
                                                                   batch['bonds'], 
                                                                   tau=0.0,
                                                                   gumbel=True)





        # get 
        dist = (cg_xyz.unsqueeze(-2) - cg_xyz.unsqueeze(-3)).pow(2).sum(-1).sqrt()
        value, knbrs = dist.sort(dim=-1, descending=False)
        knbrs = knbrs.cpu()
        value = value.cpu()

        value[:,:, self.knn+1:] = 0.0
        cg_nbr_list = value.nonzero()
        pad_cg_xyz = cg_xyz.reshape(-1, 3)
        pad_cg_nbr_list = cg_nbr_list[:, 1:] + (cg_nbr_list[:, 0] * cg_xyz.shape[1]).unsqueeze(-1)
        dist_vec = pad_cg_xyz[pad_cg_nbr_list[:,1]] - pad_cg_xyz[pad_cg_nbr_list[:,0]]
        dist_vec = dist_vec.reshape(cg_xyz.shape[0],  self.n_cgs * self.knn, 3)
        
        #dx = xyz - cg_xyz[:, self.pooler.assign_idx, :]
        
        dx_recon = torch.einsum("ije,nj->ine", dist_vec,  self.B )

        # recentering 
        cg_offset = torch.einsum("bin,bij->bjn", dx_recon, assign_norm)
        cg_offset_lift = cg_offset[:, self.pooler.assign_idx, :]

        xyz_recon = cg_xyz[:, self.pooler.assign_idx, :] - cg_offset_lift + dx_recon
    
        return soft_assign, xyz, xyz_recon