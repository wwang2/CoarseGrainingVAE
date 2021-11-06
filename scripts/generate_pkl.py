import pickle 
import torch
from run_ala import * 
from utils import * 
from sidechain import * 
from sampling import get_bond_graphs
import sidechainnet as scn




def generate_data(params):

    params['edgeorder'] = 2 
    params['n_data'] = 1000000

    thinning = params['thinning']
    caspversion = params['caspversion']

    print("generating CASP{}".format(caspversion))
    data = scn.load(casp_version=caspversion, thinning=thinning)

    params['dataset'] = 'casp{}'.format(caspversion)

    train_path = os.path.join("../data/",  params['dataset'] + '_{}_{}_all.pkl'.format("train", thinning))
    val_path = os.path.join("../data/",  params['dataset'] + '_{}_{}_all.pkl'.format("val", thinning))
    test_path = os.path.join("../data/",  params['dataset'] + '_{}_{}_all.pkl'.format("test", thinning))

    train_props = get_sidechainet_props(data['train'], params, n_data=params['n_data'], split='train', thinning=thinning)
    val_props = get_sidechainet_props(data['valid-10'], params, n_data=params['n_data'], split='valid-10', thinning=thinning)
    test_props = get_sidechainet_props(data['test'], params, n_data=params['n_data'], split='test', thinning=thinning)

    pickle.dump( train_props, open( train_path, "wb" ) )
    pickle.dump( val_props, open( val_path, "wb" ) )
    pickle.dump( test_props, open( test_path, "wb" ) )
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-caspversion", type=int, default=12)
    parser.add_argument("-thinning", type=str, default=30)
    params = vars(parser.parse_args())

    generate_data(params)