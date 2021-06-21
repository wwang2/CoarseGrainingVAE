#!/bin/sh
conda activate nff 
for var in 4 5 6 7 8 9 10 11 12
do 
    python run_ala.py -logdir ../exp/ala_cg{$var}_ng -device 3 -n_cgs $var -batch_size 64 -nsamples 200 -ndata 40000 -nepochs 30 -atom_cutoff 4.5 -cg_cutoff 5.0
done