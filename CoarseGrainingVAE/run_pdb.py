import sidechainnet as scn
import numpy as np
import matplotlib.pyplot as plt 
import re
import glob
from math import sqrt 
import sys
sys.path.append("../scripts/")
sys.path.append("../src/")

from run_ala import * 
from utils import * 
from sidechain import * 
from sampling import get_bond_graphs
from sidechainnet.structure.PdbBuilder import ATOM_MAP_14

def run_cv(params):
    working_dir = params['logdir']
    device  = params['device']
    n_cgs  = params['n_cgs']
    n_basis  = params['n_basis']
    n_rbf  = params['n_rbf']
    atom_cutoff = params['atom_cutoff']
    cg_cutoff = params['cg_cutoff']
    enc_nconv  = params['enc_nconv']
    dec_nconv  = params['dec_nconv']
    batch_size  = params['batch_size']
    beta  = params['beta']
    nsplits = params['nsplits']
    ndata = params['ndata']
    nsamples = params['nsamples']
    nepochs = params['nepochs']
    lr = params['lr']
    activation = params['activation']
    optim = optim_dict[params['optimizer']]
    dataset_label = params['dataset']
    shuffle_flag = params['shuffle']
    cg_mp_flag = params['cg_mp']
    atom_decode_flag = params['atom_decode']
    nevals = params['nevals']
    graph_eval = params['graph_eval']
    tqdm_flag = params['tqdm_flag']
    n_ensemble = params['n_ensemble']
    det = params['det']
    gamma = params['gamma']
    factor = params['factor']
    patience = params['patience']
    eta = params['eta']
    kappa = params['kappa']
    mapshuffle = params['mapshuffle']
    threshold = params['threshold']
    savemodel = params['savemodel']
    auxcutoff = params['auxcutoff']

    failed = False
    min_lr = 5e-8

    if det:
        beta = 0.0
        print("Recontruction Task")
    else:
        print("Sampling Task")


    if params['dataset'] == 'debug':
        data = scn.load("debug")
        train_props = get_sidechainet_props(data['train'])
        val_props = get_sidechainet_props(data['valid-10'])

        traindata = CGDataset(train_props.copy())
        valdata = CGDataset(val_props.copy())

        traindata.generate_neighbor_list(atom_cutoff=params['atom_cutoff'], 
                                       cg_cutoff=None, device="cpu", undirected=True)
        valdata.generate_neighbor_list(atom_cutoff=params['atom_cutoff'], 
                                       cg_cutoff=None, device="cpu", undirected=True)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-logdir", type=str)
    parser.add_argument("-device", type=int)
    parser.add_argument("-n_cgs", type=int)
    parser.add_argument("-lr", type=float, default=2e-4)
    parser.add_argument("-dataset", type=str, default='debug')
    parser.add_argument("-n_basis", type=int, default=512)
    parser.add_argument("-n_rbf", type=int, default=10)
    parser.add_argument("-activation", type=str, default='swish')
    parser.add_argument("-atom_cutoff", type=float, default=3.5)
    parser.add_argument("-optimizer", type=str, default='adam')
    parser.add_argument("-cg_cutoff", type=float, default=4.0)
    parser.add_argument("-enc_nconv", type=int, default=3)
    parser.add_argument("-dec_nconv", type=int, default=3)
    parser.add_argument("-batch_size", type=int, default=2)
    parser.add_argument("-nepochs", type=int, default=2)
    parser.add_argument("-ndata", type=int, default=200)
    parser.add_argument("-edgeorder", type=int, default=2)
    parser.add_argument("-beta", type=float, default=0.001)
    parser.add_argument("-gamma", type=float, default=0.01)
    parser.add_argument("-eta", type=float, default=0.01)
    parser.add_argument("-kappa", type=float, default=0.01)
    parser.add_argument("-threshold", type=float, default=1e-3)
    parser.add_argument("-nsplits", type=int, default=5)
    parser.add_argument("-patience", type=int, default=5)
    parser.add_argument("-factor", type=float, default=0.6)
    parser.add_argument("--cross", action='store_true', default=False)
    parser.add_argument("--graph_eval", action='store_true', default=False)
    parser.add_argument("--shuffle", action='store_true', default=False)
    parser.add_argument("--cg_mp", action='store_true', default=False)
    parser.add_argument("--tqdm_flag", action='store_true', default=False)
    parser.add_argument("--det", action='store_true', default=False)

    params = vars(parser.parse_args())
    params['savemodel'] = True

    # add more info about this job 
    if params['det']:
        task = 'recon'
    else:
        task = 'sample'
    
    params['logdir'] = annotate_job(params['cg_method'] +  task + '_ndata{}'.format(params['ndata']), params['logdir'], params['n_cgs'])
 
    run_cv(params)