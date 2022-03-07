
import argparse
from sigopt import Connection
import sys
import os
from run_pdb import * 
import shutil



parser = argparse.ArgumentParser()
parser.add_argument("-id", type=int, default=0)
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int)
#parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("-dataset", type=str, default='debug')
parser.add_argument("-thinning", type=int, default=30)
parser.add_argument("-n_data", type=int, default=1000)
parser.add_argument("-nepochs", type=int, default=50)
parser.add_argument("-edgeorder", type=int, default=2)
#parser.add_argument("-n_rbf", type=int, default=8)
#parser.add_argument("-n_basis", type=int, default=512)
#parser.add_argument("-activation", type=str, default='swish')
parser.add_argument("-optimizer", type=str, default='adam')
#parser.add_argument("-cg_cutoff", type=float, default=12.5)
parser.add_argument("-dec_nconv", type=int, default=6)
parser.add_argument("-batch_size", type=int, default=4)
parser.add_argument("-n_epochs", type=int, default=2)
parser.add_argument("-ndata", type=int, default=200)
parser.add_argument("-beta", type=float, default=0.0)
#parser.add_argument("-gamma", type=float, default=0.0)
parser.add_argument("-threshold", type=float, default=1e-3)
# parser.add_argument("-patience", type=int, default=15)
# parser.add_argument("-factor", type=float, default=0.6)
parser.add_argument("--tqdm_flag", action='store_true', default=False)
parser.add_argument("--dry_run", action='store_true', default=False)
params = vars(parser.parse_args())
params['savemodel'] = True



if params['dry_run']:
    token = 'GMBSZWXFWPHHUCXSDYLLCBBCBTKZUBVCBQRMCMXEFNEYGCFY'
    params['nepochs'] = 1
    n_obs = 2
    ndata = 200
else:
    token = 'JGTKFUYDJMOKBMDFXICMGNEFBXOOSIPAVSGUWPSMJCVDWYMA'
    n_epochs = params['n_epochs'] 
    n_obs = 1000
    ndata = params['ndata']


params['logdir'] = params['logdir']


create_dir(params['logdir'])

#Intiailize connections 
conn = Connection(client_token=token)
#Intiailize connections 

if params['id'] == 0:
    experiment = conn.experiments().create(
        name=params['logdir'],
        metrics=[dict(name='test recon', objective='minimize')],
        parameters=[
            dict(name='n_basis', type='int', bounds=dict(min=128, max=700)),
            dict(name='n_rbf', type='int', bounds=dict(min=5, max=10)),
            dict(name='activation', type='categorical', categorical_values=["ReLU", "shifted_softplus", "LeakyReLU", "swish", "ELU"]),
            dict(name='cg_cutoff', type='double', bounds=dict(min=5.0, max=9.5)),
            #dict(name='edgeorder', type='int', bounds=dict(min=1, max=3)),
            dict(name='dec_nconv', type='int', bounds=dict(min=2, max=8)),
            dict(name='kappa', type='double', bounds=dict(min=0.0001, max=10.0), transformation="log"),
            dict(name='gamma', type='double', bounds=dict(min=1.0, max=30.0), transformation="log"),
            dict(name='lr', type='double', bounds=dict(min=0.0001, max=0.0005), transformation="log"),
            dict(name='factor', type='double', bounds=dict(min=0.1, max=0.9), transformation="log"),
            dict(name='patience', type='int', bounds=dict(min=1, max=10))
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


    run_params = dict(params,**trial)
    run_params['logdir'] = os.path.join(params['logdir'], suggestion.id)

    print("Suggestion ID: {}".format(suggestion.id))

    test_recon, test_ged, failed = run_cv(run_params)


    if np.isnan(test_recon):
        failed = True

    if not failed:
        conn.experiments(experiment.id).observations().create(
          suggestion=suggestion.id,
          value=test_ged
        )
    elif failed:
        conn.experiments(experiment.id).observations().create(
          suggestion=suggestion.id,
          failed=failed
        )

    experiment = conn.experiments(experiment.id).fetch()
