import os 
import sys
from tqdm import tqdm 
import torch
import numpy as np
from ase import Atoms, io 
import networkx as nx
from datetime import date
import torch.autograd.profiler as profiler

def annotate_job(task, job_name, N_cg):
    today = date.today().strftime('%m-%d')
    return "{}_{}_{}_N{}".format(job_name, today, task, N_cg)

def create_dir(name):
    if not os.path.isdir(name):
        os.mkdir(name)   

def check_CGgraph(dataset):
    frame_idx = np.random.randint(0, len(dataset), 20)

    for idx in frame_idx:
        a = dataset.props['CG_nbr_list'][idx]
        adj = [ tuple(pair.tolist()) for pair in a ]
        G = nx.Graph()
        G.add_edges_from(adj)
        connected = nx.is_connected(G)
        if not connected:
            print("One of the sampled CG graphs is not connected, training failed")
            return connected
        return True

class EarlyStopping():
    '''from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/'''
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def KL(mu1, std1, mu2, std2):
    if mu2 == None:
        return -0.5 * torch.sum(1 + torch.log(std1.pow(2)) - mu1.pow(2) - std1.pow(2), dim=-1).mean()
    else:
        return 0.5 * ( (std1.pow(2) / std2.pow(2)).sum(-1) + ((mu1 - mu2).pow(2) / std2).sum(-1) + \
            torch.log(std2.pow(2)).sum(-1) - torch.log(std1.pow(2)).sum(-1) - std1.shape[-1] ).mean()

def batch_to(batch, device):
    gpu_batch = dict()
    for key, val in batch.items():
        gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
    return gpu_batch

def loop(loader, optimizer, device, model, beta, epoch, 
        gamma, eta=0.0, kappa=0.0, train=True, looptext='', tqdm_flag=True):
    
    total_loss = []
    recon_loss = []
    orient_loss = []
    norm_loss = []
    graph_loss = []
    kl_loss = []
    
    if train:
        model.train()
        mode = '{} train'.format(looptext)
    else:
        model.train() # yes, still set to train when reconstructing
        mode = '{} valid'.format(looptext)

    if tqdm_flag:
        loader = tqdm(loader, position=0, file=sys.stdout,
                         leave=True, desc='({} epoch #{})'.format(mode, epoch))
    
    for i, batch in enumerate(loader):

        batch = batch_to(batch, device)

        S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon = model(batch)

        # loss
        loss_kl = KL(S_mu, S_sigma, H_prior_mu, H_prior_sigma) 
        loss_recon = (xyz_recon - xyz).pow(2).mean()

        # add graph loss 
        edge_list = batch['bond_edge_list'].to("cpu")
        xyz = batch['nxyz'][:, 1:].to("cpu")
        gen_dist = (xyz_recon[edge_list[:, 0 ]] - xyz_recon[edge_list[:, 1 ]]).pow(2).sum(-1).sqrt()
        data_dist = (xyz[edge_list[:, 0 ]] - xyz[edge_list[:, 1 ]]).pow(2).sum(-1).sqrt().to(xyz_recon.device)
        loss_graph = (gen_dist - data_dist).pow(2).mean()

        # add orientation loss 
        cg_xyz = batch['CG_nxyz'][:, 1:]
        mapping = batch['CG_mapping']

        loss_dx_orient = 0.0
        loss_dx_norm = 0.0

        loss = loss_kl * beta + loss_recon + loss_graph * gamma + eta * loss_dx_orient + kappa * loss_dx_norm
        
        # optimize 
        if train:
            optimizer.zero_grad()
            loss.backward()

            # perfrom gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

        else:
            loss.backward()

        # logging 
        recon_loss.append(loss_recon.item())
        kl_loss.append(loss_kl.item())
        # orient_loss.append(loss_dx_orient.item())
        # norm_loss.append(loss_dx_norm.item())
        graph_loss.append(loss_graph.item())
        total_loss.append(loss.item())
        
        mean_kl = np.array(kl_loss).mean()
        mean_recon = np.array(recon_loss).mean()
        # mean_orient = np.array(orient_loss).mean()
        # mean_norm = np.array(norm_loss).mean()
        mean_graph = np.array(graph_loss).mean()
        mean_total_loss = np.array(total_loss).mean()
        
        memory = torch.cuda.memory_allocated(device) / (1024 ** 2)

        postfix = ['total={:.3f}'.format(mean_total_loss),
                    'KL={:.4f}'.format(mean_kl) , 
                   'recon={:.4f}'.format(mean_recon),
                   # 'norm={:.4f}'.format(mean_norm) , 
                   # 'orient={:.4f}'.format(mean_orient),
                   'graph={:.4f}'.format(mean_graph) , 
                   'memory ={:.4f} Mb'.format(memory) 
                   ]
        
        if tqdm_flag:
            loader.set_postfix_str(' '.join(postfix))

        del loss, loss_graph, loss_kl, loss_recon, loss_dx_orient, loss_dx_norm

    for result in postfix:
        print(result)
    
    return mean_total_loss, mean_kl, mean_recon, mean_graph, xyz, xyz_recon 

def get_all_true_reconstructed_structures(loader, device, model, atomic_nums, n_cg, atomwise_z=False, tqdm_flag=True):

    model.eval()

    model = model.to(device)

    true_xyzs = []
    recon_xyzs = []
    cg_xyzs = []
    mus = []
    sigmas = []

    if atomwise_z == True:
        n_z = len(atomic_nums)
    else:
        n_z = n_cg 

    if tqdm_flag:
        loader = tqdm(loader, position=0, leave=True) 

    for batch in loader:
        batch = batch_to(batch, device)
        S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon = model(batch)

        true_xyzs.append(xyz.detach().cpu())
        recon_xyzs.append(xyz_recon.detach().cpu())
        cg_xyzs.append(batch['CG_nxyz'][:, 1:].detach().cpu())
        
        mus.append(S_mu.detach().cpu())
        sigmas.append(S_sigma.detach().cpu())

        num_features = S_mu.shape[-1]

        del S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon

        memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
        postfix = ['memory ={:.4f} Mb'.format(memory)]
        
        if tqdm_flag:
            loader.set_postfix_str(' '.join(postfix))

    true_xyzs = torch.cat(true_xyzs).reshape(-1, len(atomic_nums), 3).numpy()
    recon_xyzs = torch.cat(recon_xyzs).reshape(-1, len(atomic_nums), 3).numpy()
    cg_xyzs = torch.cat(cg_xyzs).reshape(-1, n_cg, 3).numpy()
    
    mu = torch.cat(mus).reshape(-1, n_z, num_features).mean(0)
    sigma = torch.cat(sigmas).reshape(-1, n_z, num_features).mean(0)
    
    return true_xyzs, recon_xyzs, cg_xyzs, mu, sigma

def dump_numpy2xyz(xyzs, atomic_nums, path):
    trajs = [Atoms(positions=xyz, numbers=atomic_nums.ravel()) for xyz in xyzs]
    io.write(path, trajs)