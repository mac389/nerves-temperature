from brian2 import *
from brian2.units import *
from scipy.constants import physical_constants

import numpy as np 
taum = 20*ms
taue = 5*ms
taui = 10*ms
Vt = -30*mV
Vr = -60*mV

R,_,_ = physical_constants['molar gas constant']
#Also returns units, precision

F,_,_ = physical_constants['Faraday constant']

reversal_potentials = {'sodium':{'concentration':{'in': 18,'out':145},'Z': 1},
						'potassium': {'concentration':{'in': 135,'out':3},'Z': 1},
						'chloride': {'concentration':{'in': 10,'out':110},'Z':-1},
						'calcium': {'concentration':{'in': .0001,'out':1.2},'Z':2}}


def calculate_nernst_potential(temperature=298., ion='sodium'):
	#assume temperature in Kelvins 
	ion = reversal_potentials[ion]
	return 1000*R*temperature/(ion['Z']*F)*np.log(ion['concentration']['out']/float(ion['concentration']['in']))

for ion in reversal_potentials.keys():
	reversal_potentials[ion]['E'] = calculate_nernst_potential(ion=ion)


El = -40*mV
Ee = 0.5*(reversal_potentials['sodium']['E'] + reversal_potentials['calcium']['E'])*mV
Ei = 0.5*(reversal_potentials['potassium']['E'] + reversal_potentials['chloride']['E'])*mV
gl = 1

eqs = '''
dv/dt  = (ge*(v-Ee)+gi*(v-Ei)+gl*(v-El))/taum : volt (unless refractory)
dge/dt = -ge/taue : 1 
dgi/dt = -gi/taui : 1
'''

P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms)
P.v = 'Vr + rand() * (Vt - Vr)'
P.ge = 0
P.gi = 0

we = .02# excitatory synaptic weight (voltage)
wi = .1# inhibitory synaptic weight
Ce = Synapses(P, P, pre='ge += we')
Ci = Synapses(P, P, pre='gi += wi')
Ce.connect('i<3200', p=0.02)
Ci.connect('i>=3200', p=0.02)

s_mon = SpikeMonitor(P)

run(1 * second)

plot(s_mon.t/ms, s_mon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
show()