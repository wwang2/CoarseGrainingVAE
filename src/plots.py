from ase import io
from scipy.stats import gaussian_kde
import numpy as np
import torch

import pyemma
import mdshare

import matplotlib.pyplot as plt 
import matplotlib

def ramachandran_plot(xyzs, atomic_nums):

    traj = []
    for xyz in xyzs:
        traj.append( Atoms(positions=xyz, numbers=atomic_nums.ravel()) )

    io.write('tmp.xyz', traj)

    pdb = mdshare.fetch('alanine-dipeptide-nowater.pdb', working_directory='data')
    feat = pyemma.coordinates.featurizer(pdb)
    feat.add_backbone_torsions() 
    data = pyemma.coordinates.load('tmp.xyz', features=feat)

    k = gaussian_kde(np.vstack([data[:,0], data[:,1]]))

    density, x_edge, y_edge = np.histogram2d(data[:,0], data[:,1], 
                                             density=True, bins=300)

    x = (x_edge[1:] + x_edge[:-1]) * 0.5
    y = (y_edge[1:] + y_edge[:-1]) * 0.5

    xi, yi = np.meshgrid(x, y)
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    plt.figure(figsize=(6,6))
    plt.pcolormesh(xi, yi, np.log(zi.reshape(xi.shape) + 1e-3), alpha=0.8, #, norm=matplotlib.colors.LogNorm() 
                   shading='nearest')

    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    plt.ylabel('$\Psi$', fontsize=25)
    plt.xlabel('$\Phi$', fontsize=25)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label='density', size=20)
