import argparse
from sigopt import Connection
import sys
import os
from run_baseline import * 
import shutil
import copy 

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-id", type=int, default=None)
parser.add_argument('-dataset', type=str, default='dipeptide')
parser.add_argument('-ndata', type=int, default= 2000)
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-n_cgs", type=int)
parser.add_argument("-mapshuffle", type=float, default=0.0)
parser.add_argument("-n_epochs", type=int, default=60)
parser.add_argument("-cg_method", type=str, default='newman')
parser.add_argument("--dry_run", action='store_true', default=False)
parser.add_argument("--tqdm_flag", action='store_true', default=False)
parser.add_argument("--cross", action='store_true', default=False)
parser.add_argument('-model', type=str, default='equilinear')
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'GMBSZWXFWPHHUCXSDYLLCBBCBTKZUBVCBQRMCMXEFNEYGCFY'
    n_epochs = 2
    n_obs = 2
else:
    token = 'JGTKFUYDJMOKBMDFXICMGNEFBXOOSIPAVSGUWPSMJCVDWYMA'
    n_epochs = params['n_epochs'] 
    n_obs = 1000


paramsrange = [
            dict(name='cutoff', type='double', bounds=dict(min=2.0, max=9.0)),
            dict(name='edgeorder', type='int', bounds=dict(min=1, max=3)),
            dict(name='beta', type='double', bounds=dict(min=0.0001, max=5.0), transformation="log"),
            dict(name='gamma', type='double', bounds=dict(min=0.0001, max=5.0), transformation="log"),
            dict(name='kappa', type='double', bounds=dict(min=0.0001, max=5.0), transformation="log"),
            dict(name='lr', type='double', bounds=dict(min=0.00002, max=0.0002), transformation="log")]

if 'mlp' in params['model']:
    paramsrange += [dict(name='depth', type='int', bounds=dict(min=1, max=3)), 
                    dict(name='width', type='int', bounds=dict(min=1, max=3)),
                    dict(name='activation', type='categorical', 
                        categorical_values=["ReLU", "shifted_softplus", "LeakyReLU", "swish", "ELU"])
                    ]

create_dir(params['logdir'])
conn = Connection(client_token=token)

if params['id'] == 0:
    experiment = conn.experiments().create(
        name=params['logdir'],
        metrics=[dict(name='recon', objective='minimize')],
        parameters=paramsrange ,
        observation_budget=n_obs, # how many iterations to run for the optimization
        parallel_bandwidth=10,
    )

elif type(params['id']) != 0:
    experiment = conn.experiments(params['id']).fetch()


i = 0
while experiment.progress.observation_count < experiment.observation_budget:

    suggestion = conn.experiments(experiment.id).suggestions().create()
    trial =  suggestion.assignments

    trial['logdir'] = os.path.join(params['logdir'], suggestion.id)
    trial['n_epochs'] = params['n_epochs']
    trial['N_cg'] = params['n_cgs'] 
    trial['batch_size'] = params['batch_size']
    trial['device'] = params['device']
    trial['tqdm_flag'] = params['tqdm_flag']
    trial['cg_method'] = params['cg_method']
    trial['ndata'] = params['ndata']
    trial['dataset'] = params['dataset']
    trial['mapshuffle'] = params['mapshuffle']
    trial['model'] = params['model']
    trial['cross'] = params['cross']
    trial['n_splits'] = 2

    print("Suggestion ID: {}".format(suggestion.id))

    exp_param = copy.deepcopy(trial)
    baseline_param = copy.deepcopy(trial) 
    exp_param['logdir'] = os.path.join(trial['logdir'], 'exp')

    for key in exp_param.keys():
        print("{}: {}".format(key, exp_param[key]))

    create_dir(trial['logdir'])
    print("run experiments")
    exp_recon, exp_geds, failed, assign = run(exp_param)
    print("run baseline")

    value = exp_geds.mean()

    if not failed:
        conn.experiments(experiment.id).observations().create(
          suggestion=suggestion.id,
          value=value,
        )
    elif failed:
        conn.experiments(experiment.id).observations().create(
          suggestion=suggestion.id,
          failed=failed
        )

    experiment = conn.experiments(experiment.id).fetch()