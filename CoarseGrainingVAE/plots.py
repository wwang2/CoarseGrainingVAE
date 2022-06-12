from ase import io
from ase import Atoms
from scipy.stats import gaussian_kde
import numpy as np
import torch

import pyemma
import mdshare
import mdtraj as md

import matplotlib.pyplot as plt 
import matplotlib

import numpy.ma as ma
import itertools
import re
import pandas as pd
import glob

import math
def get_cv_stats( csvs, column):
    
    cg_res = []
    
    all_mean = []
    all_error = []

    for path in csvs:
        res = int(re.search('(?<=N)[0-9]+', path).group(0))
        cv_data = pd.read_csv(path)
        cg_res.append(res)
        mean = cv_data[column].mean()
        error = cv_data[column].std() / math.sqrt(cv_data[column].shape[0])
        
        all_mean.append(mean)
        all_error.append(error)
    
    reorder = np.argsort(cg_res)
    cg_res = np.array(cg_res)[reorder]

    all_mean = np.array(all_mean)[reorder]
    all_error = np.array(all_error)[reorder]
    
    return cg_res, all_mean, all_error

def retrieve_frames(exp_dir, variable_names):
    cvs = glob.glob("{}/{}/cv_stats.csv".format(working_dir, exp_dir))

    data_list = []
    data_frame = []
    for variable in variable_names:
        res, mean, se  = get_cv_stats( cvs, variable) 
        data_frame.append( mean ) 
        data_frame.append( se ) 

    mux = pd.MultiIndex.from_product([variable_names, ['mean','se']])
    frame = pd.DataFrame(np.array(data_frame).transpose(), columns=mux, index=res)

    return frame 

def kernel_density_plot(data, xlabel, ylabel, label='kT', figsize=(6,6)):
    k = gaussian_kde(np.vstack([data[:,0], data[:,1]]))

    density, x_edge, y_edge = np.histogram2d(data[:,0], data[:,1], 
                                             density=True, bins=300)

    x = (x_edge[1:] + x_edge[:-1]) * 0.5
    y = (y_edge[1:] + y_edge[:-1]) * 0.5

    xi, yi = np.meshgrid(x, y)
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    plt.figure(figsize=figsize)
    plt.pcolormesh(xi, yi, np.log(zi.reshape(xi.shape) + 1e-3), alpha=0.8, #, norm=matplotlib.colors.LogNorm() 
                   shading='nearest')

    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    plt.ylabel(xlabel, fontsize=25)
    plt.xlabel(ylabel, fontsize=25)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label=label, size=20)



def ramachandran_plot(xyzs, atomic_nums):

    traj = []
    for xyz in xyzs:
        traj.append( Atoms(positions=xyz, numbers=atomic_nums.ravel()) )

    io.write('tmp.xyz', traj)

    pdb = mdshare.fetch('alanine-dipeptide-nowater.pdb', working_directory='data')
    feat = pyemma.coordinates.featurizer(pdb)
    feat.add_backbone_torsions() 
    data = pyemma.coordinates.load('tmp.xyz', features=feat)

    return data

def get_bond_name(bond):
    atom1_res = bond.atom1.residue.name
    atom2_res = bond.atom2.residue.name

    atom1_el = bond.atom1.name
    atom2_el = bond.atom2.name

    bond_name = '{}-{} - {}-{}'.format(atom1_res, atom1_el, atom2_res, atom2_el)
    
    return bond_name

def is_backbone_bond(bond):
    
    return bond.atom1.is_backbone and bond.atom2.is_backbone



def get_bonds(traj, pdb, backbone = True):

    bond_indices = []
    bond_names = []

    # get backbond distances 
    for bond in peptide.top.bonds:

        if is_backbone_bond(bond):
            bond_indices.append([bond.atom1.index ,bond.atom2.index])
            bond_names.append(get_bond_name(bond))

    bond_indices = np.array(bond_indices)
    bonds = md.compute_distances(traj, bond_indices)
    
    bonds = bonds * 10.0 # to Angstroms 
    return bonds, bond_names

def get_sample_rmsd(rmsd_samples, axis):
    merged = list(itertools.chain(*rmsd_samples))

    cg_res = [int(re.search('(?<=N)[0-9]+', path).group(0)) for path in merged]

    sample_gen = []
    rmsd_vals = {}


    for path in merged:
        res = int(re.search('(?<=N)[0-9]+', path).group(0))
        rmsd_data = np.loadtxt(path)

        if res not in rmsd_vals.keys():
            rmsd_data = np.loadtxt(path)
            if len(rmsd_data.shape) != 1:
                rmsd_vals[res] = [rmsd_data[:, axis].mean()]
            else:
                rmsd_vals[res] = []
        else:
            rmsd_vals[res] += [rmsd_data[:, axis].mean()]
            
    rmsds = [np.array(val) for val in list( rmsd_vals.values() )]
    #rmsd_std = [np.array(val).std() for val in list( rmsd_vals.values() )]
    cg_res = list( rmsd_vals.keys() )

    reorder = np.argsort(cg_res)
    cg_res = np.array(cg_res)[reorder]

    rmsds = np.array(rmsds)[reorder]
            
    return cg_res, rmsds


def get_cv(files, mask=None):
    cg_res = []
    cv = []

    for exp in files:
        m = re.search('(?<=N)[0-9]+', exp)
        cg_res.append(int(m.group(0)))
        cv.append( np.loadtxt(exp) )
    cv = np.array(cv)
    
    # apply argsort 
    cg_res = np.array(cg_res)
    order = np.argsort(cg_res)
    
    cg_res = cg_res[order]
    cv = cv[order]
    
    if mask is not None: 
        mask_value = ma.array(cv, mask=mask).mean(axis=1)[:, None]        
        cv = np.where(mask, mask_value, cv) 
    # else:
    #     mask = mask = np.isnan(cv)
    #     mask_value = ma.array(cv, mask=mask).mean(axis=1)[:, None]        
    #     cv = np.where(mask, mask_value, cv)  

    return cg_res, cv, order