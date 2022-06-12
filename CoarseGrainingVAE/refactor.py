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

            # update_block = self.update_blocks[i]
            # dH_I, dV_I = update_block(s_i=H_I, v_i=V_I)

            # H_I = H_I + dH_I 
            # V_I = V_I + dV_I
            
        return H_I, V_I 
