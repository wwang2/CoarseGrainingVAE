import pickle 
import torch
from run_ala import * 
from utils import * 
from sidechain import * 
from sampling import get_bond_graphs


params = {'edgeorder': 2}

thinning = 30 

for caspversion in range(7, 13, 1):

    data = scn.load(casp_version=caspversion, thinning=params['thinning'])

    params['dataset'] = 'casp{}'.format(caspversion)


    train_path = os.path.join("../data/",  params['dataset'] + '_{}_{}_all.pkl'.format("train", thinning))
    val_path = os.path.join("../data/",  params['dataset'] + '_{}_{}_all.pkl'.format("val", thinning))
    test_path = os.path.join("../data/",  params['dataset'] + '_{}_{}_all.pkl'.format("test", thinning))

    train_props = get_sidechainet_props(data['train'], params, n_data=params['n_data'], split='train', thinning=params['thinning'])
    val_props = get_sidechainet_props(data['valid-10'], params, n_data=params['n_data'], split='valid-10', thinning=params['thinning'])
    test_props = get_sidechainet_props(data['test'], params, n_data=params['n_data'], split='test', thinning=params['thinning'])


    pickle.dump( train_props, open( train_path, "wb" ) )
    pickle.dump( val_props, open( val_path, "wb" ) )
    pickle.dump( test_props, open( test_path, "wb" ) )
        