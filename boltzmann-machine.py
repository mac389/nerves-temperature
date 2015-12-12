import numpy as np 

#What connectivity patterns between NEURONS to make UNITS connected like a Bolztmann machine

import random
import Graphics as artist
import matplotlib.pyplot as plt 

from scipy.constants import k
from matplotlib import rcParams

rcParams['text.usetex'] = True 

def make_memories(nmem=20,nunits=100,sparsity=0.25):
	memories = np.random.random_sample(size=(nmem,nunits))
	memories[memories<sparsity] = 1
	memories[memories!=1] = -1
	return memories.astype(int)

def make_connection_matrix(memories):
    M = np.array([np.outer(memory,memory) for memory in memories]).sum(axis=0)
    M[np.diag_indices_from(M)] = 0 
    return M.astype(int)

N = {'units':100,'memories':10}
memories = make_memories(nmem = N['memories'],nunits = N['units'])
connection_matrix = make_connection_matrix(memories)
duration = 1000
T = 310.15
idxs = np.random.choice(N['units'],size=duration).astype(int)
threshold = np.random.random_sample(size=(duration,))
#Initial conditions 
#Mixing fraction FOR LATER

#Bipolar units

T =[4,1,.25]
data = {}

for temperature in T:
	v = np.zeros((N['units'],duration))
	v[:,0] = 2*np.random.randint(2,size=N['units'])-1

	for t in xrange(1,duration):
		v[:,t] = v[:,t-1]

		inputs = float(connection_matrix[idxs[t],:].dot(v[:,t]))
		chance_of_firing = 1./(1+np.exp(-inputs/temperature))
		v[idxs[t],t] = 1 if chance_of_firing > threshold[t] else -1

	data[temperature] = v

row_idx,col_idx = np.tril_indices(3,)
heatmap_args = {'aspect':'auto','interpolation':'nearest','cmap':plt.cm.bone_r}
fig,axes = plt.subplots(nrows=len(T),ncols=len(T))
for row in xrange(len(data)):
	for col in xrange(len(data)):
		if row == col:
			ax = axes[row,col]
			cax = ax.imshow(data[T[row]],**heatmap_args)
			artist.adjust_spines(ax)
			if row == 1 and col == 0:
				ax.set_ylabel(artist.format('Neuron'))
			if row == 2 and col == 1:
				ax.set_xlabel(artist.format('Time'))
		elif col > row:
			axes[row,col].axis('off')
		else:
			cax = axes[row,col].imshow(data[T[row]]-data[T[col]],**heatmap_args)
			artist.adjust_spines(axes[row,col])

plt.tight_layout()
plt.show()
