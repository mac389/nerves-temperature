import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.gridspec as gridspec

from matplotlib import rcParams
rcParams['text.usetex'] = True

format = lambda text: r'\Large \textbf{\textsc{%s}}'%text

def adjust_spines(ax,spines=['left','bottom']):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',10)) # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def raster(activity, ax, color='k'):
    
    units,duration = activity.shape
    for ith, trial in enumerate(activity.T):
        ax.vlines(np.where(trial==1)[0], ith + .5, ith + 1.5, color=color)
    ax.set_ylim(ymin=0,ymax=units)
    ax.set_xlim(xmin=0,xmax=duration)
    return ax

def network_activity(overlap,r,moniker='x',hopfield=False):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    cax = ax.imshow(overlap,interpolation='nearest',aspect='auto')
    adjust_spines(ax)
    ax.set_xlabel(format('Time'))
    ax.set_ylabel(format('Memory'))
    cbar = plt.colorbar(cax)
    cbar.set_label(format('Overlap'))

    rax = fig.add_subplot(122)        
    if not hopfield:
        rcax = rax.imshow(r,interpolation='nearest',aspect='auto')
        adjust_spines(rax)
        rax.set_ylabel('Time')
        rax.set_xlabel('Neuron')
        rcbar = plt.colorbar(rcax)
        rcbar.set_label(format('Rate'))
    else:
        raster(r,rax)
        adjust_spines(rax)
        rax.spines['bottom'].set_smart_bounds(False)
        rax.set_ylabel('Time')
        rax.set_xlabel('Neuron')
    plt.tight_layout()
    plt.savefig('%s-network-activity.png'%moniker)

def connection_matrix_and_memory(memories,M,moniker='x'):
    
    #show connection matrix,memories
    fig = plt.figure()
    ax = plt.subplot2grid((2,2),(0,0),rowspan=2)
    cax = ax.imshow(memories,interpolation='nearest',aspect='auto',cmap=plt.cm.bone_r)
    adjust_spines(ax)
    ax.set_ylabel(format('Unit'))
    ax.set_xlabel(format('Memory'))

    kax = plt.subplot2grid((2,2),(0,1))
    vmax = max(np.absolute(M).ravel())
    kcax = kax.imshow(M,interpolation='nearest',aspect='equal',cmap=plt.cm.bwr,vmin=-vmax,vmax=vmax)
    adjust_spines(kax)
    cbar = plt.colorbar(kcax)
    cbar.set_ticks([-vmax,0,vmax])
    kax.set_ylabel(format('From'))
    kax.set_xlabel(format('To'))

    dist_weights = plt.subplot2grid((2,2),(1,1))
    Mpos = M[M>0]
    Mneg = M[M<0]
    dist_weights.hist(Mpos.ravel(),color='r',alpha=0.8)
    dist_weights.hist(Mneg.ravel(),color='b',alpha=0.8)
    dist_weights.axvline(x=Mpos.ravel().mean(),linewidth=2,color='r',linestyle='--')
    dist_weights.axvline(x=Mneg.ravel().mean(),linewidth=2,color='b',linestyle='--')
    dist_weights.set_yticks([0,1000,2000,3000,4000,5000])
    dist_weights.set_xticks([-vmax,0,vmax])
    adjust_spines(dist_weights)
    dist_weights.set_xlabel(format('Synaptic Weight'))
    dist_weights.set_ylabel(format('Count'))
    dist_weights.annotate('Median: %.02f'%np.median(Mpos.ravel()), xy=(.75, .7), xycoords='axes fraction',
                    horizontalalignment='center', verticalalignment='center',color='r')
    dist_weights.annotate('Median: %.02f'%np.median(Mneg.ravel()), xy=(.2, .7), xycoords='axes fraction',
                    horizontalalignment='center', verticalalignment='center',color='b')
    plt.tight_layout()
    plt.savefig('%s-weights-memories.png'%moniker,dpi=300)
    plt.savefig('%s-weights-memories.tiff'%moniker,dpi=300)
    