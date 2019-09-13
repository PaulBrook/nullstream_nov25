"""
Created on Fri Sep 13 15:06:34 2019

@author: albertos (shamelessly stolen)
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def seaborn_corner(params, truths, labels, ranges=None, param_names=None,
                   bins=20, cmap=None, lower_true_color='k', true_in_lim=True,
                   diag_color=(0.362553, 0.003243, 0.649245),
                   savefile='', height=2.5):
    """
    Corner plot w/Seaborn. Options for diagonal plot are 'hist' or 'kde'.
    Options for the lower triangle are 'hexbin' or 'contour'.  Params should be
    a pandas dataframe; true_vals and labels are dicts with keys which match the
    columns of params.  Order and/or use of a subset of columns can be specified by
    passing a list to param_names.  To set the x/ylim such that the true value
    is always displayed, set true_in_lim to True; set it to False to use only
    the histogram for the limits
    """
    g = sns.PairGrid(params, vars=param_names, height=height, diag_sharey=False)

    g.map_diag(sns.distplot, bins=bins, kde=False, color=diag_color,
               hist=True, norm_hist=True)

    g.map_lower(plt.hexbin, gridsize=bins, cmap=cmap, mincnt=1,
                edgecolors=None, linewidths=0, rasterized=True)

    # make upper triangle invisible
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)

    if ranges is None:
        ax_titles = labels
    else:
        ax_titles = {var: '{0}\n[{1:0.5g}, {2:0.5g}]'.format(labels[var],
                                                             ranges[var][0],
                                                             ranges[var][1]) for var in g.x_vars}

    # Fix upper left corner plot
    g.axes[0, 0].set_ylabel('')
    g.axes[0, 0].yaxis.set_ticklabels('')

    # fiddle with lower triangle plots
    x_vars, y_vars = np.meshgrid(g.x_vars, g.y_vars)
    for ax, xvar, yvar in zip(g.axes.flatten(), x_vars.flatten(), y_vars.flatten()):
        # change labels
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        if xlabel in labels.keys():
            ax.set_xlabel(labels[xlabel])
        if ylabel in labels.keys():
            ax.set_ylabel(labels[ylabel])

        # plot true values for off-diagonals
        if xvar != yvar:
            xlim = [np.min(params[xvar]), np.max(params[xvar])]
            ylim = [np.min(params[yvar]), np.max(params[yvar])]
            if true_in_lim:
                xlim = [np.fmin(xlim[0], truths[xvar]),
                        np.fmax(xlim[1], truths[xvar])]
                ylim = [np.fmin(ylim[0], truths[yvar]),
                        np.fmax(ylim[1], truths[yvar])]
            ax.vlines(truths[xvar], ylim[0], ylim[1], linestyles='dashed',
                      colors=lower_true_color)
            ax.hlines(truths[yvar], xlim[0], xlim[1], linestyles='dashed',
                      colors=lower_true_color)
            ax.set_ylim(ylim + [-0.1, 0.1]*np.diff(ylim))
            ax.set_xlim(xlim + [-0.1, 0.1]*np.diff(xlim))

    # plot true values for diagonals
    for ax, var in zip(g.diag_axes, g.x_vars):
        ylim = ax.get_ylim()
        ax.vlines(truths[var], ylim[0], ylim[1], linestyles='dashed',
                  colors=diag_color)
        ax.set_ylim(ylim)
        ax.set_title(ax_titles[var])

    plt.savefig(savefile, bbox_inches='tight')

