import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import seaborn as sb
from misc.ImportDm import epochs
from misc import Utils, ImportDm
from functools import partial
import multiprocessing

timestamps = {'optionMade':100, 'optionsOn':130}


def func(area, cell):
    for epoch in epochs:

        # Load in the data

        (y,             # firing rate in 10 ms increments +/- 1 second from trial epoch
         distChange,    # For this step, what is the new distance to target compared to the previous step (+1 = moved towards target)
         currAngle,     # Angle between current location and target
         hd,            # What direction did they move (north south east west)
         numsteps,      # In this trial, how many steps have they taken?
         perfTrials,    # Was this trial perfect? I.e. all steps were towards the target
         startAngle,    # Starting angle to the target
         currDist,      # Current distance to the target
         from_x,         # X-coordinate of state they just moved from
         from_y,         # Y-coordinate of ....
         to_x,           # X-coordinate of state they have just chosen
         to_y,           # Y-coordinate of ...
         ) = ImportDm.getData(area, cell, epoch)

        # Only use trials where they moved towards the target
        m = distChange == 1
        y = y[m]
        state_x = np.array(from_x[m], dtype=int, copy=True)
        state_y = np.array(from_y[m], dtype=int, copy=True)

        # We count the number of state visits in each half of the session in order to perform a median split
        frs = np.zeros((3, np.max(state_x) + 1, np.max(state_y) + 1, len(y[0])))
        counts = np.zeros((3, np.max(state_x) + 1, np.max(state_y) + 1, len(y[0])))
        n = len(state_x)
        for i, (xx, yy) in enumerate(zip(state_x, state_y)):
            counts[int(i < (n // 2)), xx, yy] += 1
            frs[int(i < (n // 2)), xx, yy] += y[i]

            counts[2, xx, yy] += 1
            frs[2, xx, yy] += y[i]

        # We smooth with a different weighting on diagonal vs manhattan distances
        diag_scaling = 0.25
        manhattan_scaling = 0.5
        origcounts, origfrs = np.copy(counts), np.copy(frs)
        for k in range(3):
            for i in range(len(counts[k])):
                for j in range(len(counts[k][0])):
                    if i>0 and j>0:
                        counts[k][i,j] += origcounts[k][i-1,j-1] * diag_scaling
                        frs[k][i,j] += origfrs[k][i-1,j-1] * diag_scaling
                    if i>0:
                        counts[k][i,j] += origcounts[k][i-1,j] * manhattan_scaling
                        frs[k][i,j] += origfrs[k][i-1,j] * manhattan_scaling
                    if i>0 and j<len(counts[k][0])-1:
                        counts[k][i,j] += origcounts[k][i-1,j+1] * diag_scaling
                        frs[k][i,j] += origfrs[k][i-1,j+1] * diag_scaling
                    if i<len(counts[k])-1 and j<len(counts[k][0])-1:
                        counts[k][i,j] += origcounts[k][i+1,j+1] * diag_scaling
                        frs[k][i,j] += origfrs[k][i+1,j+1] * diag_scaling
                    if i<len(counts[k])-1:
                        counts[k][i,j] += origcounts[k][i+1,j] * manhattan_scaling
                        frs[k][i,j] += origfrs[k][i+1,j] * manhattan_scaling
                    if i<len(counts[k])-1 and j>0:
                        counts[k][i,j] += origcounts[k][i+1,j-1] * diag_scaling
                        frs[k][i,j] += origfrs[k][i+1,j-1] * diag_scaling
                    if j<len(counts[k][0])-1:
                        counts[k][i,j] += origcounts[k][i,j+1] * manhattan_scaling
                        frs[k][i,j] += origfrs[k][i,j+1] * manhattan_scaling
                    if j>0:
                        counts[k][i,j] += origcounts[k][i,j-1] * manhattan_scaling
                        frs[k][i,j] += origfrs[k][i,j-1] * manhattan_scaling

        frs /= counts

        #%% Cross correlation
        try:
            ac = scipy.signal.correlate2d(frs[2, ..., timestamps[epoch]], frs[2, ..., timestamps[epoch]])
            # delete central points
            centre = np.where(ac == np.nanmax(ac))
            ac[centre[0][0]-1:centre[0][0]+2, centre[1][0]] = np.nan
            ac[centre[0][0]-1:centre[0][0]+2, centre[1][0]-1] = np.nan
            ac[centre[0][0]-1:centre[0][0]+2, centre[1][0]+1] = np.nan
        except IndexError:
            print('error numero dos')

        # Plot our data
        axes = Utils.p(2, 3, sharey=False, figsize=(10, 5))
        [ax.axis('off') for ax in axes[[0, 1, 2, 3, 5]]]

        # Rate maps
        try:
            sb.heatmap(frs[2, ..., timestamps[epoch]].T, ax=axes[2], annot=False, cmap='rainbow', square=True, cbar=False);
            sb.heatmap(ac.T, ax=axes[-1], cmap='rainbow', square=True)
            sb.heatmap(origcounts[2, ..., timestamps[epoch]].T, ax=axes[3], annot=True, cmap='rainbow', cbar=False, square=True)
            axes[2].set_title('FR (all trials)')
            axes[1].set_title('Autocorr')
            axes[3].set_title('Samples')
        except ValueError:
            print('beep boop im an error')

        # Rate maps from median split
        sb.heatmap(frs[0, ..., timestamps[epoch]].T, ax=axes[0], annot=False, cmap='rainbow', square=True, cbar=False);
        sb.heatmap(frs[1, ..., timestamps[epoch]].T, ax=axes[1], annot=False, cmap='rainbow', square=True, cbar=False);
        axes[0].set_title('FR (first half)')
        axes[1].set_title('FR (second half)')
        axes[-1].set_title('Autocorr')
        axes[4].set_xticklabels([])
        axes[4].set_yticklabels([])

        # Correlate across states which were visited in both splits
        a, b = frs[0, ..., timestamps[epoch]].flatten(), frs[1, ..., timestamps[epoch]].flatten()
        no_nans = ~np.isnan(a) & ~np.isnan(b)
        r, p = scipy.stats.pearsonr(a[no_nans], b[no_nans])

        # Plot the correlation across states
        sb.regplot(a, b, ax=axes[4], label=f'r={np.round(r, 2)} {Utils.getpmarker(p, True)}\nn={no_nans.sum()}/108')
        axes[4].set_title('Split1 vs Split2')
        axes[4].legend()

        # Save our plot
        plt.suptitle(f'Area: {area}, Cell: ' + str(cell))
        plt.tight_layout()
        plt.savefig(f'plots/plot_{epoch}_{area}_{cell}')
        plt.close('all')


if __name__ == '__main__':
    parallel = F
    if parallel:
        for area in ImportDm.areas[::-1]:
            f = partial(func, area)
            pool = multiprocessing.Pool()
            pool.map(f, range(ImportDm.n[area]))
    else:
        for cell in range(ImportDm.n[area]):
            func(area, cell)
