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
parser.add_argument("-n_cgs", type=int)
parser.add_argument("--dry_run", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'GMBSZWXFWPHHUCXSDYLLCBBCBTKZUBVCBQRMCMXEFNEYGCFY'
    n_epochs = 2
    n_obs = 2
    ndata = 200
else:
    token = 'JGTKFUYDJMOKBMDFXICMGNEFBXOOSIPAVSGUWPSMJCVDWYMA'
    n_epochs = 60 
    n_obs = 1000
    ndata = 10000

create_dir(params['logdir'])

#Intiailize connections 
conn = Connection(client_token=token)

if params['id'] == None:
    experiment = conn.experiments().create(
        name=params['logdir'],
        metrics=[dict(name='cv rmsd', objective='minimize')],
        parameters=[
            dict(name='n_basis', type='int', bounds=dict(min=32, max=256)),
            dict(name='n_rbf', type='int', bounds=dict(min=5, max=16)),
            dict(name='activation', type='categorical', categorical_values=["ReLU", "shifted_softplus", "LeakyReLU", "swish", "ELU"]),
            dict(name='cg_cutoff', type='double', bounds=dict(min=params['min_cgcutoff'], max=params['min_cgcutoff'] + 10.0)),
            dict(name='atom_cutoff', type='double', bounds=dict(min=3.0, max=4.5)),
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

    # if trial['dir_mp'] == 'True':
    #     dir_mp_flag = True 
    # else:
    #     dir_mp_flag = True

    # if trial['dir_mp'] == 'True':
    #     dir_mp_flag = True 
    # else:
    #     dir_mp_flag = False

    dir_mp_flag = True

    trial['logdir'] = os.path.join(params['logdir'], suggestion.id)
    trial['device'] = params['device']
    trial['dataset'] = params['dataset']
    trial['batch_size'] = params['batch_size']
    trial['nepochs'] = n_epochs
    trial['ndata'] = ndata
    trial['nsamples'] = 200
    trial['n_cgs'] = params['n_cgs']
    trial['nsplits'] = 3
    trial['randommap'] = False
    trial['shuffle'] = False
    trial['optimizer'] = 'adam'
    trial['dir_mp'] = dir_mp_flag
    trial['cg_mp'] = False
    trial['atom_decode'] = False

    cv_mean, cv_std = run_cv(trial)
    if np.isnan(cv_mean):
        fail_flag = True
    else:
        fail_flag = False 

    conn.experiments(experiment.id).observations().create(
      suggestion=suggestion.id,
      value=cv_mean,
      value_stddev=cv_std,
      failed=fail_flag
    )

    experiment = conn.experiments(experiment.id).fetch()