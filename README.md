# Coarse-Graining Variational Autoencoders

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
