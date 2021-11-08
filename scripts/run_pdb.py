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

def save_selected_recon(loader, model, device, path):

    test_results = {'id':[], 'rmsd':[], 'len': [], 'ged':[], 'time':[]}
    df = pd.DataFrame(test_results)

    for i, batch in enumerate(loader):

        batch = batch_to(batch, device)
        start_time = time.time()
        mu, sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon = model(batch)
        dt = time.time() - start_time

        # only get the first protein 
        xyz_recon_first = xyz_recon[: batch['num_atoms'][0].item()]
        seq = batch['seq'][0]
        msk = batch['msk'][0]
        id = batch['id'][0]
        z = batch['nxyz'][:, 0].detach().cpu().numpy()

        # compute rmsd: 
        rmsd = torch.sqrt(((xyz_recon - xyz)**2).sum(axis=1).mean()).item()

        # compute graph ged 
        ref_atoms = Atoms(numbers=z, positions=xyz.detach().cpu().numpy())
        recon_atoms =  Atoms(numbers=z, positions=xyz_recon.detach().cpu().numpy())
        all_rmsds, heavy_rmsds, valid_ratio, valid_hh_ratio, graph_val_ratio, graph_hh_val_ratio = eval_sample_qualities(ref_atoms, [recon_atoms])

        # compute bond deviation
        bondpairs = get_bond_graphs(ref_atoms).triu().nonzero()
        bond_ref = (xyz[bondpairs[:, 0]] - xyz[bondpairs[:, 1]]).pow(2).sum(-1).sqrt()
        bond_recon = (xyz_recon[bondpairs[:, 0]] - xyz_recon[bondpairs[:, 1]]).pow(2).sum(-1).sqrt()
        bond_diff = (bond_ref - bond_recon).abs().mean()

        result = {'id': id , 'rmsd': rmsd, 'ged': graph_val_ratio[0],  'time': dt , 'len': len(seq), 'bond_diff': bond_diff.item()}
        df = df.append(result, ignore_index=True)

        pad_xyz = dense2pad_crd(xyz_recon_first, batch['num_CGs'][0].item(),  batch['CG_mapping'][: batch['num_atoms'][0].item()])    
        save_pdb(msk, seq, pad_xyz.reshape(-1, 3), '{}/{}_backmap.pdb'.format(path, id))

    df.to_csv('{}/egnn_test_results.csv'.format(path))
        # save results 


def run_cv(params):
    working_dir = params['logdir']
    device  = params['device']
    n_basis  = params['n_basis']
    n_rbf  = params['n_rbf']
    cg_cutoff = params['cg_cutoff']
    dec_nconv  = params['dec_nconv']
    batch_size  = params['batch_size']
    beta  = params['beta']
    ndata = params['ndata']
    nepochs = params['nepochs']
    lr = params['lr']
    activation = params['activation']
    optim = optim_dict[params['optimizer']]
    dataset_label = params['dataset']
    tqdm_flag = params['tqdm_flag']
    gamma = params['gamma']
    factor = params['factor']
    patience = params['patience']
    threshold = params['threshold']

    failed = False
    min_lr = 5e-8

    split_dir = working_dir
    create_dir(split_dir)

    if params['dataset'] == 'debug':
        data = scn.load( params['dataset'] )
        train_props = get_sidechainet_props(data['train'], params, n_data=params['n_data'], split='train', thinning=params['thinning'])
        val_props = get_sidechainet_props(data['valid-10'], params, n_data=params['n_data'], split='valid-10', thinning=params['thinning'])
        test_props = get_sidechainet_props(data['test'], params, n_data=params['n_data'], split='test', thinning=params['thinning'])
    elif params['dataset'] == 'casp12':
        data_path = '../data/casp12_{}_' + str(params['thinning']) + '_all.pkl'
        train_props = pickle.load(open(data_path.format('train'), "rb" ) )
        val_props = pickle.load(open(data_path.format('val'), "rb" ) )
        test_props = pickle.load(open(data_path.format('test'), "rb" ) )

    #train_props = get_sidechainet_props(data['train'], params, n_data=params['n_data'], split='train', thinning=params['thinning'])
    #val_props = get_sidechainet_props(data['valid-10'], params, n_data=params['n_data'], split='valid-10', thinning=params['thinning'])
    #test_props = get_sidechainet_props(data['test'], params, n_data=params['n_data'], split='test', thinning=params['thinning'])


    traindata = CGDataset(train_props.copy())
    valdata = CGDataset(val_props.copy())
    testdata = CGDataset(test_props.copy())


    # remove problemic structures 
    valid_ids = []
    for i, edge in enumerate(train_props['bond_edge_list']):
        natoms = train_props['nxyz'][i].shape[0]
        nedges = edge.shape[0]
        ratio = nedges/ natoms 
        if ratio <= 3.0 and ratio != 0.0:
            valid_ids.append(i)

    traindata = get_subset_by_indices(valid_ids, traindata)

    all_idx = list( range(len(traindata.props['nxyz'])) )
    random.shuffle(all_idx)
    traindata = get_subset_by_indices(all_idx[:params['n_data']], traindata)

    traindata.generate_neighbor_list(atom_cutoff=0.5,
                                   cg_cutoff=None, device="cpu", undirected=True, use_bond=True)
    valdata.generate_neighbor_list(atom_cutoff=0.5,
                                   cg_cutoff=None, device="cpu", undirected=True, use_bond=True)
    testdata.generate_neighbor_list(atom_cutoff=0.5,
                                   cg_cutoff=None, device="cpu", undirected=True, use_bond=True)

    trainloader = DataLoader(traindata, batch_size=batch_size, collate_fn=CG_collate, shuffle=False)
    valloader = DataLoader(valdata, batch_size=batch_size, collate_fn=CG_collate, shuffle=False)
    testloader = DataLoader(testdata, batch_size=1, collate_fn=CG_collate, shuffle=False)

    # initialize model 
    atom_mu = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))
    atom_sigma = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis))

    # register encoder 

    decoder = EquivariantDecoder(n_atom_basis=n_basis, n_rbf = n_rbf, 
                                  cutoff=params['cg_cutoff'], num_conv = dec_nconv, activation=activation, cross_flag=True)

    model = PCN(decoder,  feature_dim=n_basis,  offset=False).to(device)

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


    cv_stats_pd = pd.DataFrame( { 'train_heavy_recon': [],
                'test_heavy_recon': [],
                'train_graph': [],  'test_graph': [],
                'recon_all_ged': [], 'recon_heavy_ged': [], 
                'recon_all_valid_ratio': [], 
                'recon_heavy_valid_ratio': [] })


    for epoch in range(params['nepochs']):

        train_loss, mean_kl_train, mean_recon_train, mean_graph_train, xyz_train, xyz_train_recon = loop(trainloader, optimizer, device,
                                                   model, beta, epoch, 
                                                   train=True,
                                                    gamma=gamma,
                                                    looptext='dataset {} Fold {} train'.format(params['dataset'], epoch),
                                                    tqdm_flag=True)


        val_loss, mean_kl_val, mean_recon_val, mean_graph_val, xyz_val, xyz_val_recon = loop(valloader, optimizer, device,
                                           model, beta, epoch, 
                                           train=False, 
                                            gamma=gamma,
                                            looptext='dataset {} Fold {} val'.format(params['dataset'], epoch),
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

    if not failed: 
        test_true_xyzs, test_recon_xyzs, test_cg_xyzs,test_all_valid_ratio, test_heavy_valid_ratio, test_all_ged, test_heavy_ged= get_all_true_reconstructed_structures(testloader, 
                                                                                         device,
                                                                                         model,
                                                                                         None,
                                                                                         n_cg=None,
                                                                                         tqdm_flag=tqdm_flag, reflection=False)
        test_loss, mean_kl_test, mean_recon_test, mean_graph_test, xyz_test, xyz_test_recon = loop(testloader, optimizer, device,
                                   model, beta, epoch, 
                                   train=False, 
                                    gamma=gamma,
                                    looptext='dataset {} Fold {} test'.format(params['dataset'], epoch),
                                    tqdm_flag=True)


        train_loss, mean_kl_train, mean_recon_train, mean_graph_train, xyz_train, xyz_train_recon = loop(trainloader, optimizer, device,
                           model, beta, epoch, 
                           train=False, 
                            gamma=gamma,
                            looptext='dataset {} Fold {} train'.format(params['dataset'], epoch),
                            tqdm_flag=True) 


        test_stats = {} 
        test_stats['train_heavy_recon'] = mean_recon_train
        test_stats['test_heavy_recon'] = mean_recon_test
        test_stats['train_graph'] = mean_graph_train
        test_stats['test_graph'] = mean_graph_test
        test_stats['recon_heavy_ged'] = test_heavy_ged
        test_stats['recon_heavy_valid_ratio'] = test_heavy_valid_ratio

        cv_stats_pd = cv_stats_pd.append(test_stats, ignore_index=True)
        cv_stats_pd.to_csv(os.path.join(split_dir, 'cv_stats.csv'),  index=False)

        # record test results one by one 
        save_selected_recon(testloader, model, device ,split_dir )

        # save model 
        model = model.to('cpu')
        torch.save(model.state_dict(), os.path.join(split_dir, 'model.pt'))

        return cv_stats_pd['test_heavy_recon'].mean(), cv_stats_pd['recon_heavy_ged'].mean(), failed
    else:
        return 0.0, 0.0, True 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-logdir", type=str)
    parser.add_argument("-device", type=int)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-dataset", type=str, default='debug')
    parser.add_argument("-thinning", type=int, default=30)
    parser.add_argument("-n_data", type=int, default=1000)
    parser.add_argument("-cg_method", type=str, default='alpha')
    parser.add_argument("-edgeorder", type=int, default=2)
    parser.add_argument("-n_rbf", type=int, default=8)
    parser.add_argument("-n_basis", type=int, default=512)
    parser.add_argument("-activation", type=str, default='swish')
    parser.add_argument("-optimizer", type=str, default='adam')
    parser.add_argument("-cg_cutoff", type=float, default=15.5)
    parser.add_argument("-dec_nconv", type=int, default=9)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-nepochs", type=int, default=2)
    parser.add_argument("-ndata", type=int, default=200)
    parser.add_argument("-beta", type=float, default=0.0)
    parser.add_argument("-gamma", type=float, default=0.0)
    parser.add_argument("-threshold", type=float, default=1e-3)
    parser.add_argument("-patience", type=int, default=15)
    parser.add_argument("-factor", type=float, default=0.6)
    parser.add_argument("--tqdm_flag", action='store_true', default=False)
    params = vars(parser.parse_args())
    params['savemodel'] = True
    
    params['logdir'] = annotate_job(params['cg_method'] + '_ndata{}'.format(params['ndata']), params['logdir'], params['dataset'])
 
    run_cv(params)