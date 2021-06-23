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
parser.add_argument("--dry_run", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'GMBSZWXFWPHHUCXSDYLLCBBCBTKZUBVCBQRMCMXEFNEYGCFY'
    n_epochs = 2
    n_obs = 2
else:
    token = 'JGTKFUYDJMOKBMDFXICMGNEFBXOOSIPAVSGUWPSMJCVDWYMA'
    n_epochs = 30 
    n_obs = 1000

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
            dict(name='cg_cutoff', type='int', bounds=dict(min=4.0, max=5.0)),
            dict(name='atom_cutoff', type='double', bounds=dict(min=3.0, max=4.5)),
            dict(name='enc_nconv', type='int', bounds=dict(min=2, max=5)),
            dict(name='dec_nconv', type='int', bounds=dict(min=2, max=5)),
            dict(name='beta', type='double', bounds=dict(min=0.0001, max=0.01), transformation="log"),
            dict(name='lr', type='double', bounds=dict(min=0.00001, max=0.001), transformation="log"),
        ],
        observation_budget=n_obs, # how many iterations to run for the optimization
        parallel_bandwidth=10,
    )

i = 0
while experiment.progress.observation_count < experiment.observation_budget:

    suggestion = conn.experiments(experiment.id).suggestions().create()
    trial =  suggestion.assignments
    trial['logdir'] = os.path.join(params['logdir'], suggestion.id)
    trial['device'] = params['device']
    trial['dataset'] = 'dipeptide'
    trial['batch_size'] = 64
    trial['nepochs'] = n_epochs
    trial['ndata'] = 5000
    trial['nsamples'] = 200
    trial['n_cgs'] = 7
    trial['nsplits'] = 3
    trial['randommap'] = False
    trial['shuffle'] = False
    trial['optimizer'] = 'adam'

    cv_score = run_cv(trial)

    conn.experiments(experiment.id).observations().create(
      suggestion=suggestion.id,
      value=cv_score,
    )

    experiment = conn.experiments(experiment.id).fetch()