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
    nepochs = params['nepochs']
    lr = params['lr']
    activation = params['activation']
    optim = optim_dict[params['optimizer']]
    dataset_label = params['dataset']
    shuffle_flag = params['shuffle']
    cg_mp_flag = params['cg_mp']
    tqdm_flag = params['tqdm_flag']
    det = params['det']
    gamma = params['gamma']
    factor = params['factor']
    patience = params['patience']
    eta = params['eta']
    kappa = params['kappa']
    threshold = params['threshold']

    failed = False
    min_lr = 5e-8

    split_dir = working_dir
    create_dir(split_dir)


    if params['dataset'] == 'debug':
        data = scn.load( params['dataset'] )
    elif params['dataset'] == 'casp12':
        data = scn.load(casp_version=12, thinning=30)
    elif params['dataset'] == 'casp14':
        data = scn.load(casp_version=14, thinning=30)


    train_props = get_sidechainet_props(data['train'], n_data=params['n_data'])
    val_props = get_sidechainet_props(data['valid-10'], n_data=params['n_data'])
    test_props = get_sidechainet_props(data['test'], n_data=params['n_data'])

    traindata = CGDataset(train_props.copy())
    valdata = CGDataset(val_props.copy())
    testdata = CGDataset(test_props.copy())

    traindata.generate_neighbor_list(atom_cutoff=params['atom_cutoff'], 
                                   cg_cutoff=None, device="cpu", undirected=True)
    valdata.generate_neighbor_list(atom_cutoff=params['atom_cutoff'], 
                                   cg_cutoff=None, device="cpu", undirected=True)
    testdata.generate_neighbor_list(atom_cutoff=params['atom_cutoff'], 
                                   cg_cutoff=None, device="cpu", undirected=True)

    trainloader = DataLoader(traindata, batch_size=1, collate_fn=CG_collate, shuffle=False)
    valloader = DataLoader(valdata, batch_size=1, collate_fn=CG_collate, shuffle=False)
    testloader = DataLoader(testdata, batch_size=1, collate_fn=CG_collate, shuffle=False)

    # initialize model 
    atom_mu = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))
    atom_sigma = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))

    # register encoder 

    decoder = EquivariantDecoder(n_atom_basis=n_basis, n_rbf = n_rbf, 
                                  cutoff=params['cg_cutoff'], num_conv = dec_nconv, activation=activation, cross_flag=True,
                                  atomwise_z=False)

    encoder = EquiEncoder(n_conv=enc_nconv, n_atom_basis=n_basis, 
                                   n_rbf=n_rbf, cutoff=cg_cutoff, activation=activation,
                                    cg_mp=cg_mp_flag, dir_mp=False, atomwise_z=False)

    # define prior 
    cgPrior = CGprior(n_conv=enc_nconv, n_atom_basis=n_basis, 
                                   n_rbf=n_rbf, cutoff=cg_cutoff, activation=activation,
                                     dir_mp=False)

    model = CGequiVAE(encoder, decoder, atom_mu, atom_sigma, n_cgs, feature_dim=n_basis, prior_net=cgPrior,
                        atomwise_z=False, offset=False, det=True).to(device)

    optimizer = optim(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=2, 
                                                            factor=factor, verbose=True, 
                                                            threshold=threshold,  min_lr=min_lr)

    early_stopping = EarlyStopping(patience=patience)
    model.train()
    # dump model hyperparams 
    with open(os.path.join(split_dir, 'modelparams.json'), "w") as outfile: 
        json.dump(params, outfile, indent=4)

    # intialize training log 
    train_log = pd.DataFrame({'epoch': [], 'lr': [], 'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': [],
               'train_KL': [], 'val_KL': [], 'train_graph': [], 'val_graph': []})


    for epoch in range(params['nepochs']):
        train_loss, mean_kl_train, mean_recon_train, mean_graph_train, xyz_train, xyz_train_recon = loop(trainloader, optimizer, device,
                                                   model, beta, epoch, 
                                                   train=True,
                                                    gamma=params['gamma'],
                                                    eta=params['eta'],
                                                    kappa=params['kappa'],
                                                    looptext='Ncg {} Fold {} train'.format(n_cgs, epoch),
                                                    tqdm_flag=True)


        val_loss, mean_kl_val, mean_recon_val, mean_graph_val, xyz_val, xyz_val_recon = loop(valloader, optimizer, device,
                                           model, beta, epoch, 
                                           train=False, 
                                            gamma=gamma,
                                            eta=eta,
                                            kappa=kappa,
                                            looptext='Ncg {} Fold {} val'.format(n_cgs, epoch),
                                            tqdm_flag=True)

        stats = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 
                'train_loss': train_loss, 'val_loss': val_loss, 
                'train_recon': mean_recon_train, 'val_recon': mean_recon_val,
               'train_KL': mean_kl_train, 'val_KL': mean_kl_val, 
               'train_graph': mean_graph_train, 'val_graph': mean_graph_val}

        train_log = train_log.append(stats, ignore_index=True)

        # smoothen the validation curve 
        smooth = sm.nonparametric.lowess(train_log['val_loss'].values,  # y
                                        train_log['epoch'].values, # x
                                        frac=0.2)

        smoothed_valloss = smooth[-1, 1]

        scheduler.step(smoothed_valloss)

        if optimizer.param_groups[0]['lr'] <= min_lr * 1.5:
            print('converged')
            break

        early_stopping(smoothed_valloss)
        if early_stopping.early_stop:
            break

        # check NaN
        if np.isnan(mean_recon_val):
            print("NaN encoutered, exiting...")
            failed = True
            break 

        # dump training curve 
        train_log.to_csv(os.path.join(split_dir, 'train_log.csv'),  index=False)


    for i, batch in enumerate(testloader):
        if i <= 4: 
            batch = batch_to(batch, device)
            mu, sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon = model(batch)
            
            # only get the first protein 
            xyz_recon_first = xyz_recon[: batch['num_atoms'].item()]
            seq = batch['seq'][0]
            msk = batch['msk'][0]

            pad_xyz = dense2pad_crd(xyz_recon_first, batch['num_CGs'][0].item(),  batch['CG_mapping'])
            save_pdb(msk, seq, pad_xyz.reshape(-1, 3), '{}/{}.pdb'.format(working_dir, seq))
            
        

    # dump selected pdbs 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-logdir", type=str)
    parser.add_argument("-device", type=int)
    parser.add_argument("-n_cgs", type=int, default=10)
    parser.add_argument("-lr", type=float, default=1e-5)
    parser.add_argument("-dataset", type=str, default='debug')
    parser.add_argument("-n_data", type=int, default=1000)
    parser.add_argument("-cg_method", type=str, default='alpha')
    parser.add_argument("-n_rbf", type=int, default=8)
    parser.add_argument("-n_basis", type=int, default=600)
    parser.add_argument("-activation", type=str, default='swish')
    parser.add_argument("-atom_cutoff", type=float, default=3.5)
    parser.add_argument("-optimizer", type=str, default='adam')
    parser.add_argument("-cg_cutoff", type=float, default=9.5)
    parser.add_argument("-enc_nconv", type=int, default=3)
    parser.add_argument("-dec_nconv", type=int, default=3)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-nepochs", type=int, default=2)
    parser.add_argument("-ndata", type=int, default=200)
    parser.add_argument("-edgeorder", type=int, default=1)
    parser.add_argument("-beta", type=float, default=0.0)
    parser.add_argument("-gamma", type=float, default=0.0)
    parser.add_argument("-eta", type=float, default=0.0)
    parser.add_argument("-kappa", type=float, default=0.0)
    parser.add_argument("-threshold", type=float, default=1e-3)
    parser.add_argument("-nsplits", type=int, default=5)
    parser.add_argument("-patience", type=int, default=15)
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