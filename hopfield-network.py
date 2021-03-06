import numpy as np 
import random
import Graphics as artist

def make_memories(nmem=20,nunits=100,sparsity=0.25):
	memories = np.random.random_sample(size=(nmem,nunits))
	memories[memories<sparsity] = 1
	memories[memories!=1] = -1
	return memories.astype(int)

def sgn(arr):
	arr[arr>=0] = 1
	arr[arr<0] = -1
	return arr.astype(int)

def make_M(memories):
    M = np.array([np.outer(memory,memory) for memory in memories]).sum(axis=0)
    M[np.diag_indices_from(M)] = 0 
    return M

def hamming(u,v):
	#Assumes that u,v are binary vectors [-1,1] of the same length
	u = np.array(u)
	v = np.array(v)
	if u.shape == v.shape:
		return np.absolute(u-v).sum()/float(2*len(v))

def overlap(u,v):
	u = np.array(u)
	v = np.array(v)
	if u.shape == v.shape:
		return u.dot(v)/float(len(v))

duration = 1000
nunits = 100
nmem=10
memories = make_memories(nmem=nmem,nunits=nunits)
M = make_M(memories)
r = np.zeros((nunits,duration),dtype=int)

#artist.connection_matrix_and_memory(memories,M,moniker='hopfield')

#Initial conditions 
mixing_fraction = 0.5 
#r[:,0] = 2*np.round(np.random.random_sample(size=(nunits,))).astype(int)-1
r[:,0] = memories[1,:]
for t in xrange(1,duration):
	#Asynchronous updating
	r[:,t] = r[:,t-1]

	idx = random.choice(xrange(nunits))
	r[idx,t] = 1 if M[idx,:].dot(r[:,t-1]) >= 0 else -1

overlap = np.array([memory.dot(r)/float(nunits) for memory in memories])

artist.hopfield_network_activity(overlap,r,memories,moniker='hopfield')