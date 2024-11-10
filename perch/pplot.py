
import matplotlib.pyplot as plt
import numpy as np

'''
Plotting functions for persistent homology output.
'''

hcol = {0: 'palevioletred', 1: 'mediumpurple', 2: 'deepskyblue'}
hnames = {0: '$H_0$', 1: '$H_1$', 2: '$H_2$'}

def barcode(hom, ax=None):  # ,dimensions=None):

    '''
    Plot barcode.

    Parameters:
    -----------
    ax : matplotlib.pyplot.axis
        Axis object.

    '''

    '''if dimensions == None:
        dimensions = list(np.unique(self.generators[:,0]).astype('int'))
    if type(dimensions) != list:
        print('fixing')
        dimensions = list(dimensions)

    plotcol = [self.filter(dimension=d) for d in dimensions]
    ravpc = self.generators#np.array(plotcol).ravel()'''
    plotcol = hom.generators

    # plotcol = self.generators
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    for k in range(len(plotcol)):
        p_i = plotcol[k]
        ax.plot(np.array([p_i[1], p_i[2]]), [k, k], c=hcol[p_i[0]])
    ax.set_xlabel('Birth –-- Death')
    ax.set_ylabel('Structure Number')
    # ax.set_xscale('linear')
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in
               list(hcol.values())[0:hom.max_Hi + 1]]
    ax.legend(markers, hnames.values(), numpoints=1, fontsize=14)
    fig.tight_layout()


def pers_diagram(hom, ax=None, dimensions=None):
    '''
    Plot persistence diagram.

    Parameters:
    -----------
    ax : matplotlib.pyplot.axis
        Axis object.
    dimensions : list
        Homology dimensions.

    '''

    if dimensions == None:
        dimensions = list(np.unique(hom.generators[:, 0]).astype('int'))
    if type(dimensions) != list:
        print('fixing')
        dimensions = list(dimensions)

    plotcol = [hom.filter(dimension=d) for d in dimensions]
    ravpc = hom.generators  # np.array(plotcol).ravel()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for d in range(len(dimensions)):
        ax.scatter(plotcol[d][:, 1], plotcol[d][:, 2], s=3, alpha=0.5, c=hcol[dimensions[d]])  # ,c=hcol[plotcol[:,0]])
    ax.plot([-np.nanmax(np.abs(ravpc[:, 1:3])), np.nanmax(np.abs(ravpc[:, 1:3]))],
            [-np.nanmax(np.abs(ravpc[:, 1:3])), np.nanmax(np.abs(ravpc[:, 1:3]))], c='grey', ls='--', alpha=0.5, lw=0.5)
    ax.set_xlim(1.1 * np.nanmin(ravpc[:, 1]), 1.1 * np.nanmax(ravpc[:, 1]))
    ax.set_ylim(1.1 * np.nanmin(ravpc[:, 2]), 1.1 * np.nanmax(ravpc[:, 2]))
    ax.set_xlabel('Birth', fontsize=14)
    ax.set_ylabel('Death', fontsize=14)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in
               list(hcol.values())[0:hom.max_Hi + 1]]
    ax.legend(markers, hnames.values(), numpoints=1, fontsize=14)
    fig.tight_layout()


def lifetime_diagram(hom, ax=None, dimensions=None):
    '''
    Plot lifetime diagram.

    Parameters:
    -----------
    ax : matplotlib.pyplot.axis
        Axis object.
    dimensions : list
        Homology dimensions.

    '''

    if dimensions == None:
        dimensions = list(np.unique(hom.generators[:, 0]).astype('int'))
    if type(dimensions) != list:
        print('fixing')
        dimensions = list(dimensions)

    plotcol = [hom.filter(dimension=d) for d in dimensions]
    # ravpc = hom.generators#np.array(plotcol).ravel()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for d in range(len(dimensions)):
        ax.scatter(plotcol[d][:, 1], np.abs(plotcol[d][:, 2] - plotcol[d][:, 1]), s=3, alpha=0.5,
                   c=hcol[dimensions[d]])  # ,c=hcol[plotcol[:,0]])
    # ax.set_xlim(1.1*np.min(plotcol[:,1]),1.1*np.max(plotcol[:,1]))
    # ax.set_ylim(1.1*np.min(plotcol[:,2]),1.1*np.max(plotcol[:,2]))
    ax.set_xlabel('Birth', fontsize=14)
    ax.set_ylabel('Lifetime', fontsize=14)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in
               list(hcol.values())[0:hom.max_Hi + 1]]
    ax.legend(markers, hnames.values(), numpoints=1, fontsize=14)
    fig.tight_layout()