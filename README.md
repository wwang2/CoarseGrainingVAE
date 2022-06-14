# Coarse-Graining Variational Autoencoders(CGVAE)

This is a repo accompanying our ICML 2022 paper "Generative Coarse-graining of Molecualr Conformations" ([arxiv link](https://arxiv.org/abs/2201.12176)). We propose a geometric generative model for backmmaping fine-grained coordinates from coarse-grained coordinates. 

Highlights:
* :heavy_check_mark: Our model works for arbitrary coarse-graining mapping protocols
* :heavy_check_mark: Our model can backmap from very representations
* :heavy_check_mark: Our model incoporates necessary geometric constraints 
* :heavy_check_mark: Our model is generative 

### Install packages 

I highly recommend creating a dedicated conda environment via: 
```
conda create -n cgvae python=3.8
```

Download and install 
```
git clone https://github.com/wwang2/CoarseGrainingVAE.git
cd CoarseGrainingVAE
conda activate cgvae
pip install -r requirement.txt # I have tested this, it should work 
pip install -e . # -e is useful if you want to edit the source code
```
#

### Download data 

#### Alanine Dipeptide trajectories 

trajectories will be automatically downloaded when running the script. The dataset is provided via `mdshare`


#### Chignolin trajectories 
  
Before downloading, make sure you have at least 1.3G diskspace.

```
cd CoarseGrainingVAE
mkdir data
wget http://pub.htmd.org/chignolin_trajectories.tar.gz -P data
cd data 
tar -xf chignolin_trajectories.tar.gz
```

### Run experiments 

Navigate to scripts directory

To run experiment for alanine dipeptide trajecotries: 

```
python run_ala.py -logdir ./dipep_exp -dataset dipeptide -device 0 -n_cgs 3 -batch_size 32 -nsamples 20 -ndata 20000 -nepochs 600 -nevals 5 -atom_cutoff 8.5 -cg_cutoff 9.5 -nsplits 5 -beta 0.05 -activation swish -dec_nconv 5 -enc_nconv 4 -lr 0.00008 -n_basis 600 -n_rbf 8 --graph_eval -gamma 25.0 -eta 0.0 -kappa 0.0 -patience 15 -cg_method cgae -edgeorder 2
```

to run experiment for chignolin trajectories:

```
python run_ala.py -logdir ./chig_exp  -dataset chignolin -device 0 -n_cgs 6 -batch_size 2 -nsamples 35 -ndata 5000 -nepochs 100 -atom_cutoff 12.0 -cg_cutoff 25.0 -nsplits 5 -beta 0.05 -gamma 50.0 -eta 0.0 -kappa 0.0 -activation swish -dec_nconv 9 -enc_nconv 2 -lr 0.0001 -n_basis 600 -n_rbf 10 -cg_method cgae --graph_eval -n_ensemble 8 -factor 0.3 -patience 14
```

A set of coarse-graining mapping is first determined before the model training. The available mapping choices include the following (specified with `-cg_method`):
```
-cg_method cgae # coarse-graining autoencoders 
-cg_method newman # girvan-newman alrogithm 
-cg_method minimal # coarse-graining by keeping heavy atoms only 
-cg_method alpha # coarse-graining based on alpha carbon
-cg_method random # coarse-graining randomly (yes, you will find it also works!)
-cg_method backbonepartition # randomly generate parition based on backbone topology so that contiguity is ensured 

```

Feel free to change the desired CG resolution via `-n_cg`, it should work for a minimal of 3-bead coarse-graining representations. (My dream is to be able to backmap any molecular geometries from representations of only 3-beads, 3 beads are all you need?) 

Evaluation stats is generated with 5-fold cross-validation. Feel free to change hyperparameters, it should not be very sensitive to hyperparameter choice. Generated samples also dumped for visualization. 

The model work for larger systems and proteins, I have tested it on a covid spike protein before, but the training might take a while. 

# 

### Future plans

There are many things I want to do with this tool for CG modeling, but I might not have the time to do it. Here are a few things:

- [ ] provide flexible user input options for handling different coordinates format and coarse-graining mapping 
- [ ] create a web-service for generating backmapping models 
- [ ] better code structures 

This repo is a reference implementation, and it might need some more work to incorporate this into a pipeline. I will try to maintain it even though I am planning on leaving academia. It is very likely that this project will be better packaged in another project for community use, depending how this field develops. If you want to contribute, please get in touch!

# 

### Citation info

```
@inproceedings{
  wang2022generativecg,
  title={Generative Coarse-Graining of Molecular Conformations},
  author={Wang, Wujie and Xu, Minkai and Cai, Chen and Miller, Benjamin Kurt and Smidt, Tess and Wang, Yusu and Tang, Jian and G{\'o}mez-Bombarelli, Rafael},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```
