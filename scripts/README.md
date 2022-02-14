# to run the alanine dipeptide 3-bead example 

python run_ala.py -logdir dipep_run -dataset dipeptide -device 0 -n_cgs 3  -batch_size 32 -nsamples 20 -ndata 500 -nepochs 600 -nevals 5 -atom_cutoff 8.5 -cg_cutoff 9.5 -nsplits 5 -beta 0.05 -activation swish -dec_nconv 5 -enc_nconv 4 -lr 0.00008 -n_basis 600 -n_rbf 8 --graph_eval -gamma 25.0 -eta 0.0 -kappa 0.0 -patience 15 -cg_method newman -edgeorder 2 --tqdm_flag
