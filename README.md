# Coarse-Graining Variational Autoencoders

### Install packages 

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
