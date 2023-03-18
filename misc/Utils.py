import numpy as np
import matplotlib.pyplot as plt


def nans(shape_of_arr):
    if type(shape_of_arr) == int:
        shape_of_arr = int(shape_of_arr)
    else:
        shape_of_arr = [int(s) for s in shape_of_arr]
    out = np.zeros(shape_of_arr)
    out.fill(np.nan)
    return out


def p(n1, n2=1, tit=None, sharey=True, hline=True, wspace=None, hspace=None, returnfig=False, sharex=False, figsize=None):
    f = plt.figure(figsize=figsize, dpi=200)
    ax= f.subplots(n1, n2, sharey=sharey, sharex=sharex)
    if n1*n2>1 and ax.ndim>1:ax=ax.flatten()
    if n1*n2>1:
        if hline: [a.axhline(0, c='black',lw=0.5, zorder=-1) for a in ax]
        [a.spines['right'].set_visible(False) for a in ax]
        [a.spines['top'].set_visible(False) for a in ax]
    else:
        if hline: ax.axhline(0, c='black',lw=0.5, zorder=-1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax =[ax]
    if tit is not None:
        plt.suptitle(tit)
    plt.subplots_adjust(hspace =hspace, wspace=wspace)
    if returnfig:
        return f, ax
    return ax

def getpmarker(p, label_ns=False):
    lab = ''
    if p < 0.05:
        lab += '*'
        if p < 0.01: lab += '*'
        if p < 0.001: lab += '*'
    elif label_ns:
        lab='n.s.'
    return lab