#!/bin/sh
conda activate nff 
for var in 4 5 6 7 8 9 10 11 
do 
    python run_ala.py -logdir ../exp/ala_cg{$var}_gn -device 3 -n_cgs $var -batch_size 64 -nsamples 100 -ndata 40000 -nepochs 30 -cutoff 4.5
done