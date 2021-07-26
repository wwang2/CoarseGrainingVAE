import argparse
from sigopt import Connection
import sys
import os
from run_ala import * 
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-id", type=int, default=None)
parser.add_argument("-dataset", type=str)
parser.add_argument("-min_cgcutoff", type=float)
parser.add_argument("-batch_size", type=int)
parser.add_argument("-ndata", type=int, default=3000)
parser.add_argument("-nevals", type=int, default=24)
parser.add_argument("-n_cgs", type=int)
parser.add_argument("-cg_method", type=str)
parser.add_argument("-n_epochs", type=int, default=60)
parser.add_argument("--dry_run", action='store_true', default=False)
parser.add_argument("--graph_opt", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'GMBSZWXFWPHHUCXSDYLLCBBCBTKZUBVCBQRMCMXEFNEYGCFY'
    n_epochs = 2
    n_obs = 2
    ndata = 200
    nevals = 24
else:
    token = 'JGTKFUYDJMOKBMDFXICMGNEFBXOOSIPAVSGUWPSMJCVDWYMA'
    n_epochs = params['n_epochs'] 
    n_obs = 1000
    ndata = params['ndata']
    nevals = params['nevals']

create_dir(params['logdir'])

#Intiailize connections 
conn = Connection(client_token=token)

if params['id'] == None:
    experiment = conn.experiments().create(
        name=params['logdir'],
        metrics=[dict(name='cv rmsd', objective='minimize')],
        parameters=[
            dict(name='n_basis', type='int', bounds=dict(min=32, max=600)),
            dict(name='n_rbf', type='int', bounds=dict(min=5, max=16)),
            dict(name='activation', type='categorical', categorical_values=["ReLU", "shifted_softplus", "LeakyReLU", "swish", "ELU"]),
            dict(name='cg_cutoff', type='double', bounds=dict(min=params['min_cgcutoff'], max=params['min_cgcutoff'] + 10.0)),
            dict(name='atom_cutoff', type='double', bounds=dict(min=4.0, max=8.5)),
            dict(name='enc_nconv', type='int', bounds=dict(min=2, max=5)),
            dict(name='dec_nconv', type='int', bounds=dict(min=2, max=7)),
            dict(name='beta', type='double', bounds=dict(min=0.0001, max=0.01), transformation="log"),
            dict(name='lr', type='double', bounds=dict(min=0.0001, max=0.001), transformation="log"),
            #dict(name='dir_mp', type='categorical', categorical_values=["True", "False"]),
            #dict(name='dec_type', type='categorical', categorical_values=["EquivariantDecoder", "ENDecoder"]),
        ],
        observation_budget=n_obs, # how many iterations to run for the optimization
        parallel_bandwidth=10,
    )

elif type(params['id']) == int:
    experiment = conn.experiments(params['id']).fetch()

i = 0
while experiment.progress.observation_count < experiment.observation_budget:

    suggestion = conn.experiments(experiment.id).suggestions().create()
    trial =  suggestion.assignments

    dir_mp_flag = True

    trial['logdir'] = os.path.join(params['logdir'], suggestion.id)
    trial['device'] = params['device']
    trial['dataset'] = params['dataset']
    trial['batch_size'] = params['batch_size']
    trial['nepochs'] = n_epochs
    trial['ndata'] = ndata
    trial['nsamples'] = 20
    trial['n_cgs'] = params['n_cgs']
    trial['nsplits'] = 3
    trial['randommap'] = False
    trial['shuffle'] = False
    trial['optimizer'] = 'adam'
    trial['dir_mp'] = dir_mp_flag
    trial['cg_mp'] = False
    trial['atom_decode'] = False
    trial['cg_method'] = params['cg_method']
    trial['nevals'] = 300
    trial['graph_eval'] = True
    trial['tqdm_flag'] = False
    trial['n_ensemble'] = 1

    cv_mean, cv_std, cv_ged_mean, cv_ged_std, failed = run_cv(trial)

    if params['graph_opt']:
        target_mean = cv_mean
        target_std = cv_std
    else:
        target_mean = cv_ged_mean
        target_std = cv_ged_std

    if np.isnan(cv_mean):
        failed = True

    if not failed:
        conn.experiments(experiment.id).observations().create(
          suggestion=suggestion.id,
          value=target_mean,
          value_stddev=target_std
        )
    elif failed:
        conn.experiments(experiment.id).observations().create(
          suggestion=suggestion.id,
          failed=failed
        )

    experiment = conn.experiments(experiment.id).fetch()