import numpy as np
from scipy.constants import physical_constants, epsilon_0

S=0.5 #Electron spin
gs = -physical_constants['electron g factor'][0]
muB=physical_constants['Bohr magneton'][0] 
kB=physical_constants['Boltzmann constant'][0] 
amu=physical_constants['atomic mass constant'][0] #An atomic mass unit in kg
e0=epsilon_0 #Permittivity of free space
a0=physical_constants['Bohr radius'][0]
kB = physical_constants['Boltzmann constant'][0]

def xyz_to_lrz(E_in):
	""" Convert from linear to circular bases """
	# create output array
	E_out = np.zeros_like(E_in,dtype='complex')
	
	# z-component doesn't change
	E_out[2] = E_in[2]
	
	## Following sign convention in 
	## 'Optically Polarised Atoms' by Auzinsh, Budker and Rochester, eq 6.32
	## OUP, 2010
	# L = 1./sqrt(2) * (x - iy)
	# R = 1./sqrt(2) * (x + iy) 
	E_out[0] = 1./np.sqrt(2) * (E_in[0] - 1.j*E_in[1])
	E_out[1] = 1./np.sqrt(2) * (E_in[0] + 1.j*E_in[1])
	
	return E_out
	
def lrz_to_xyz(E_in):
	""" Convert from circular to linear bases """

	# create output array
	E_out = np.zeros_like(E_in,dtype='complex')
	
	# z-component doesn't change
	E_out[2] = E_in[2]

	# x = 1. / sqrt(2) * [L + R]
	# y = 1.j / sqrt(2) * [L - R]
	E_out[0] = 1./np.sqrt(2) * (E_in[0] + E_in[1])
	E_out[1] = 1.j/np.sqrt(2) * (E_in[0] - E_in[1])
	
	return E_out
