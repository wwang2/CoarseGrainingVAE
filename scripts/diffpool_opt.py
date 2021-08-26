import argparse
from sigopt import Connection
import sys
import os
from run_diffpool import * 
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-id", type=int, default=None)
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-n_cgs", type=int)
parser.add_argument("-n_epochs", type=int, default=60)
parser.add_argument("--dry_run", action='store_true', default=False)
parser.add_argument("--tqdm_flag", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'GMBSZWXFWPHHUCXSDYLLCBBCBTKZUBVCBQRMCMXEFNEYGCFY'
    n_epochs = 2
    n_obs = 2
else:
    token = 'JGTKFUYDJMOKBMDFXICMGNEFBXOOSIPAVSGUWPSMJCVDWYMA'
    n_epochs = params['n_epochs'] 
    n_obs = 1000

create_dir(params['logdir'])
conn = Connection(client_token=token)

if params['id'] == None:
    experiment = conn.experiments().create(
        name=params['logdir'],
        metrics=[dict(name='recon', objective='minimize')],
        parameters=[
            dict(name='num_features', type='int', bounds=dict(min=128, max=600)),
            dict(name='n_rbf', type='int', bounds=dict(min=5, max=20)),
            dict(name='activation', type='categorical', categorical_values=["ReLU", "shifted_softplus", "LeakyReLU", "swish", "ELU"]),
            dict(name='cutoff', type='double', bounds=dict(min=4.0, max=9.0)),
            dict(name='nconv_pool', type='int', bounds=dict(min=2, max=7)),
            dict(name='enc_nconv', type='int', bounds=dict(min=2, max=7)),
            dict(name='dec_nconv', type='int', bounds=dict(min=2, max=7)),
            dict(name='gamma', type='double', bounds=dict(min=0.0001, max=1.0), transformation="log"),
            dict(name='kappa', type='double', bounds=dict(min=0.0001, max=1.0), transformation="log"),
            dict(name='lr', type='double', bounds=dict(min=0.00001, max=0.001), transformation="log"),
            dict(name='tau_rate', type='double', bounds=dict(min=0.0001, max=0.1), transformation="log"),
            dict(name='tau_0', type='double', bounds=dict(min=1.0, max=5.0)),
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

    trial['logdir'] = os.path.join(params['logdir'], suggestion.id)
    trial['n_epochs'] = params['n_epochs']
    trial['N_cg'] = params['n_cgs'] 
    trial['batch_size'] = params['batch_size']
    trial['device'] = params['device']
    trial['tqdm_flag'] = params['tqdm_flag']

    print("Suggestion ID: {}".format(suggestion.id))

    test_recon, failed, assign = run(trial)

    if not failed:
        conn.experiments(experiment.id).observations().create(
          suggestion=suggestion.id,
          value=test_recon,
        )
    elif failed:
        conn.experiments(experiment.id).observations().create(
          suggestion=suggestion.id,
          failed=failed
        )

    experiment = conn.experiments(experiment.id).fetch()