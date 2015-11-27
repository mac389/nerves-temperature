import matplotlib.pyplot as plt 
import numpy as np 
import Graphics as artist 

from scipy.constants import physical_constants
from matplotlib import rcParams


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

def calculate_v_steady(temperature=298):
	for ion in reversal_potentials.keys():
		reversal_potentials[ion]['E'] = calculate_nernst_potential(temperature=temperature,ion=ion)

	return np.average([reversal_potentials['chloride']['E']]+[-40])

#Calculate V steady at many different temperatures
V,T = zip(*[(calculate_v_steady(T),T) for T in range(273,310)])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(T,V,'.')

plt.show()