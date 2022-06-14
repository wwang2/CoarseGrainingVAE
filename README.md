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


* ### citation info

```
@inproceedings{
  wang2022generativecg,
  title={Generative Coarse-Graining of Molecular Conformations},
  author={Wang, Wujie and Xu, Minkai and Cai, Chen and Miller, Benjamin Kurt and Smidt, Tess and Wang, Yusu and Tang, Jian and G{\'o}mez-Bombarelli, Rafael},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```
