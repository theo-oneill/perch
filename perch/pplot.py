'''
Plotting functions for persistent homology output.
'''

import matplotlib.pyplot as plt
import numpy as np


hcol = {0: 'palevioletred', 1: 'mediumpurple', 2: 'deepskyblue'}
hnames = {0: '$H_0$', 1: '$H_1$', 2: '$H_2$'}


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
    sentinel = np.finfo(ravpc.dtype).min if np.issubdtype(ravpc.dtype, np.floating) else None
    bd = ravpc[:, 1:3].copy()
    if sentinel is not None:
        bd[bd == sentinel] = np.nan
        bd[bd == -sentinel] = np.nan

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for d in range(len(dimensions)):
        ax.scatter(plotcol[d].birth, plotcol[d].death, s=3, alpha=0.5, c=hcol[dimensions[d]])
    extent = np.nanmax(np.abs(bd))
    if np.isfinite(extent):
        ax.plot([-extent, extent], [-extent, extent],
                c='grey', ls='--', alpha=0.5, lw=0.5)

    def _set_lim(setter, lo, hi):
        if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
            setter(1.1 * lo, 1.1 * hi)

    _set_lim(ax.set_xlim, np.nanmin(bd[:, 0]), np.nanmax(bd[:, 0]))
    _set_lim(ax.set_ylim, np.nanmin(bd[:, 1]), np.nanmax(bd[:, 1]))
    ax.set_xlabel('Birth', fontsize=14)
    ax.set_ylabel('Death', fontsize=14)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in
               list(hcol.values())[0:hom.max_Hi + 1]]
    ax.legend(markers, hnames.values(), numpoints=1, fontsize=14)
    if created_fig:
        fig.tight_layout()
