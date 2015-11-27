import numpy as np 
import Graphics as artist
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

from matplotlib import rcParams
rcParams['text.usetex'] = True 

def make_memories(nmem=20,nunits=100,sparsity=0.25):
	memories = np.random.random_sample(size=(nmem,nunits))
	memories[memories<sparsity] = 1
	memories[memories!=1] == 0
	return memories.astype(int)

def make_M(memories,lam=1.25,sparsity=0.25):
	nmem,nunits = memories.shape
	gain = lam/(sparsity*nunits*(1-sparsity))
	matrix = np.array([np.outer(memory-sparsity*np.ones_like(memory),memory-sparsity*np.ones_like(memory))
					for memory in memories]).sum(axis=0)
	return gain*matrix - (1./(sparsity*nunits) * np.ones((nunits,nunits)))

def F(alpha,current,background):
	ans = alpha*np.tanh((current-background)/alpha)
	ans[ans<0] = 0
	return ans 

def T(temperature,rmax=310.15):
	return np.cos(temperature-rmax)

sparsity = 0.25
nunits=100
memories = make_memories(sparsity=sparsity)

M = make_M(memories,sparsity=sparsity)

duration = 1000
alpha = 150.
background = -20 #negative shows spontaneous activity
r = np.zeros((nunits,duration))

#random activation 
r[:,0] = np.random.random_sample(size=(nunits,))
epsilon = 0.01
temperature = 310.15
for t in xrange(1,duration):
	r[:,t] = r[:,t-1] + epsilon*F(alpha,M.dot(r[:,t-1]),background)*T(temperature)

overlap = np.array([memory.dot(r)/nunits for memory in memories]) # is there a better/normalized measure?


'''
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(-100,101)
y = F(alpha,x,background)
ax.plot(x,y,'k',linewidth=1.5)
artist.adjust_spines(ax)
ax.set_xlabel('Input')
ax.set_ylabel('Firing Rate')
plt.savefig('i-o-single-neuron-more-rapid.png')
plt.savefig('i-o-single-neuron-more-rapid.tiff',dpi=300)
'''
'''
fig = plt.figure()
ax = fig.add_subplot(121)
cax = ax.imshow(overlap,interpolation='nearest',aspect='auto')
artist.adjust_spines(ax)
ax.set_xlabel(artist.format('Time'))
ax.set_ylabel(artist.format('Memory'))
cbar = plt.colorbar(cax)
cbar.set_label(artist.format('Overlap'))

rax = fig.add_subplot(122)
rcax = rax.imshow(r,interpolation='nearest',aspect='auto')
artist.adjust_spines(rax)
rax.set_ylabel('Time')
rax.set_xlabel('Neuron')
rcbar = plt.colorbar(rcax)
rcbar.set_label(artist.format('Rate'))
plt.tight_layout()
plt.show()
'''
#Analyze overlap

#-Where is overlap greatest?
#-Is overlap in many patterns explained by pattern similarity or by ambiguous activity?

'''
#show connection matrix,memories
fig = plt.figure()
ax = plt.subplot2grid((2,2),(0,0),rowspan=2)
cax = ax.imshow(memories,interpolation='nearest',aspect='auto',cmap=plt.cm.bone_r)
artist.adjust_spines(ax)
ax.set_ylabel(artist.format('Unit'))
ax.set_xlabel(artist.format('Memory'))

kax = plt.subplot2grid((2,2),(0,1))
vmax = max(np.absolute(M).ravel())
kcax = kax.imshow(M,interpolation='nearest',aspect='equal',cmap=plt.cm.bwr,vmin=-vmax,vmax=vmax)
artist.adjust_spines(kax)
cbar = plt.colorbar(kcax)
cbar.set_ticks([-vmax,0,vmax])
kax.set_ylabel(artist.format('From'))
kax.set_xlabel(artist.format('To'))

dist_weights = plt.subplot2grid((2,2),(1,1))
Mpos = M[M>0]
Mneg = M[M<0]
dist_weights.hist(Mpos.ravel(),color='r',alpha=0.8)
dist_weights.hist(Mneg.ravel(),color='b',alpha=0.8)
dist_weights.axvline(x=Mpos.ravel().mean(),linewidth=2,color='r',linestyle='--')
dist_weights.axvline(x=Mneg.ravel().mean(),linewidth=2,color='b',linestyle='--')
dist_weights.set_yticks([0,1000,2000,3000,4000,5000])
dist_weights.set_xticks([-vmax,0,vmax])
artist.adjust_spines(dist_weights)
dist_weights.set_xlabel('Synaptic Weight')
dist_weights.set_ylabel('Count')
dist_weights.annotate('Median: %.02f'%np.median(Mpos.ravel()), xy=(.75, .7), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',color='r')
dist_weights.annotate('Median: %.02f'%np.median(Mneg.ravel()), xy=(.2, .7), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',color='b')
plt.tight_layout()
plt.savefig('weights-memories.png',dpi=300)
plt.savefig('weights-memories.tiff',dpi=300)
'''
