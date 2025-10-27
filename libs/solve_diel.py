import numpy as np
from numpy import pi,sin,cos
from scipy.constants import physical_constants, epsilon_0
from sympy import Symbol, lambdify, solve
from sympy.matrices import det, Matrix
import scipy.linalg as la

S=0.5 #Electron spin
gs = -physical_constants['electron g factor'][0]
muB=physical_constants['Bohr magneton'][0] 
kB=physical_constants['Boltzmann constant'][0] 
amu=physical_constants['atomic mass constant'][0] #An atomic mass unit in kg
e0=epsilon_0 #Permittivity of free space
a0=physical_constants['Bohr radius'][0]
kB = physical_constants['Boltzmann constant'][0]

def solve_diel(chiL, chiR, chiZ, THETA, Bfield, verbose=False,force_numeric=False):
	''' 
	Solves the wave equation to find the two propagating normal modes of the system, 
	for a given magnetic field angle THETA. For the general case, use symbolic python to 
	solve for the roots of n-squared.
	(Escapes this slow approach for the two analytic cases for the Voigt and Faraday geometries)
	
	Returns the rotation matrix to transform the coordinate system into the normal mode basis,
	and returns the two refractive index arrays.
	
	Inputs:
	
		chiL, chiR, chiZ	:	1D lists or numpy arrays, of length N, that are the frequency-dependent electric susceptibilities
		THETA				:	Float, Magnetic field angle in radians
		Bfield				:	Float, Magnitude of applied magnetic field (skips slow approach if magnetic field is very close to zero)
		
	Options:
		
		verbose			:	Boolean to output more print statements (timing reports mostly)
		force_numeric	:	If True, forces all angles to go through the numeric approach, rather than escaping for the analytic cases (THETA=0, THETA=pi/2...)
	
	Outputs:
		RotMat	:	Rotation matrix to transform coordinate system, dimensions (3, 3, N)
		n1		:	First solution for refractive index, dimensions (N)
		n2		:	Second solution for refractive index, dimensions (N)
	
	'''


	if verbose: 
		print(('B-field angle (rad, pi rad): ',THETA, THETA/np.pi))
	
	# make chiL,R,Z arrays if not already
	chiL = np.array(chiL)
	chiR = np.array(chiR)
	chiZ = np.array(chiZ)
	
	#### Escape the slow loop for analytic (Faraday and Voigt) cases
	## For these analytic cases we can use array operations and it is therefore
	## much faster to compute
	if (abs(THETA%(2*np.pi) - np.pi/2) < 1e-4) and (not force_numeric):
		# ANALYTIC SOLNS FOR VOIGT
		if verbose: print('Voigt - analytic')
		
		# solutions for elements of the dielectric tensor:
		ex = 0.5 * (2. + chiL + chiR)
		exy = 0.5j * (chiR - chiL)
		ez = 1.0 + chiZ

		# refractive indices to propagate
		n1 = np.sqrt(ex + exy**2/ex)
		n2 = np.sqrt(ez)
		
		ev1 = [np.zeros(len(ex)),ex/exy,np.ones(len(ex))]
		ev2 = [np.ones(len(ex)),np.zeros(len(ex)),np.zeros(len(ex))]
		ev3 = [np.zeros(len(ex)),np.zeros(len(ex)),np.ones(len(ex))]
		
		RotMat = np.array([ev1,ev2,ev3])
		
		if verbose:
			print('Shortcut:')
			print((RotMat.shape))
			print((n1.shape))
			print((n2.shape))
		
	elif ((abs(THETA) < 1e-4) or ((abs(THETA - np.pi)) < 1e-4) or abs(Bfield)<1e-2)  and (not force_numeric): ## Use Faraday geometry if Bfield is very close to zero
		# ANALYTIC SOLNS FOR FARADAY
		#if verbose: 
		if verbose: print('Faraday - analytic TT')
		
		ex = 0.5 * (2. + chiL + chiR)
		exy = 0.5j * (chiR - chiL)
		e_z = 1.0 + chiZ
		
		n1 = np.sqrt(ex + 1.j*exy)
		n2 = np.sqrt(ex - 1.j*exy)

		ev1 = np.array([-1.j*np.ones(len(ex)),np.ones(len(ex)),np.zeros(len(ex))]) 
		ev2 = np.array([1.j*np.ones(len(ex)),np.ones(len(ex)),np.zeros(len(ex))])
		ev3 = [np.zeros(len(ex)),np.zeros(len(ex)),np.ones(len(ex))]

		if (abs(THETA) < 1e-4):
			RotMat = np.array([ev1,ev2,ev3])
		else:
			#if anti-aligned, swap the two eigenvectors
			RotMat = np.array([ev2,ev1,ev3])
			
		if verbose:
			print('Shortcut:')
			print((RotMat.shape))
			print((n1.shape))
			print((n2.shape))

	else:
		if verbose: print('Non-analytic angle.. This will take a while...')	##### THIS IS THE ONE THAT's WRONG....
		# set up sympy symbols
		theta = Symbol('theta',real=True)
		n_sq = Symbol('n_sq')
		e_x = Symbol('e_x')
		e_xy = Symbol('e_xy')
		e_z = Symbol('e_z')

		# General form of the dielectric tensor
		DielMat = Matrix (( 	[(e_x - n_sq)*cos(theta), e_xy, e_x*sin(theta)],
									[-e_xy * cos(theta), e_x - n_sq, -e_xy*sin(theta)],
									[(n_sq - e_z)*sin(theta), 0, e_z*cos(theta)] 			))
		
		# Substitute in angle
		DielMat_sub = DielMat.subs(theta, pi*THETA/np.pi)

		# Find solutions for complex indices for a given angle
		solns = solve(det(DielMat_sub), n_sq)
		
		# Find first refractive index
		DielMat_sub1 = DielMat_sub.subs(n_sq, solns[0])
		n1 = np.zeros(len(chiL),dtype='complex')
		n1old = np.zeros(len(chiL),dtype='complex')
		# Find second refractive index
		DielMat_sub2 = DielMat_sub.subs(n_sq, solns[1])
		n2 = np.zeros(len(chiL),dtype='complex')
		n2old = np.zeros(len(chiL),dtype='complex')
		
		Dsub1 = lambdify((e_x,e_xy,e_z), DielMat_sub1, 'numpy')
		Dsub2 = lambdify((e_x,e_xy,e_z), DielMat_sub2, 'numpy')
		
		nsub1 = lambdify((e_x,e_xy,e_z), solns[0], 'numpy')
		nsub2 = lambdify((e_x,e_xy,e_z), solns[1], 'numpy')
		
		# Initialise rotation matrix
		RotMat = np.zeros((3,3,len(chiL)),dtype='complex')
		
		# populate refractive index arrays
		n1 = np.sqrt(nsub1(0.5*(2.+chiL+chiR), 0.5j*(chiR-chiL), (1.0+chiZ)))
		n2 = np.sqrt(nsub2(0.5*(2.+chiL+chiR), 0.5j*(chiR-chiL), (1.0+chiZ)))
			
		# loop over all elements of chiL,R,Z to populate eigenvectors
		# time-limiting step for arrays of length >~ 5000
		for i, (cL, cR, cZ) in enumerate(zip(chiL,chiR,chiZ)):
			#if verbose: print 'Detuning point i: ',i
			
		
			# NEW method
			
			# Sub in values of susceptibility
			DMaNP = Dsub1(0.5*(2.+cL+cR), 0.5j*(cR-cL), (1.0+cZ))
			#print DMa
			ev1 = null(DMaNP).T
			# Populate the refractive index array
			#n1[i] = np.sqrt(nsub1(0.5*(2.+cL+cR), 0.5j*(cR-cL), (1.0+cZ)))
			
			
			#print '\n\n\n'
			
			#print 'scipy: ', ev1
			
			#
			## Now repeat the above for second eigenvector
			#
			
		## NEW
			# Sub in values of susceptibility
			DMaNP = Dsub2(0.5*(2.+cL+cR), 0.5j*(cR-cL), (1.0+cZ))
			# Find null eigenvector
			ev2 = null(DMaNP).T
			# Populate the refractive index array
			#n2[i] = np.sqrt(nsub2(0.5*(2.+cL+cR), 0.5j*(cR-cL), (1.0+cZ)))
			
			# Populate the rotation matrix
			RotMat[:,:,i] = [ev1, ev2, [0,0,1]]
			
	if verbose: print('SD done')
	return RotMat, n1, n2

def null(A,tol=1e-6):
	ee, ev = la.eig(A)
	
	#for E,V in zip(ee,ev.T):
	#	print 'Eigs:',abs(E), '\t', E#, '\t', V
	#print '\n'
	
	z = list(zip(ee,ev.T))
	zs = sorted(z, key=lambda f: abs(f[0])) # sort by absolute value of eigenvectors
	ees, evs = list(zip(*zs))
	
	#for E,V in zip(ee,ev):
	#	print abs(E), '\t', E, '::', V
	
	if abs(ees[0]<tol):
		return evs[0].T
	else:
		print('No null eigenvector found! List of eigenvalules:')
		for E,V in zip(ee,ev.T):
			print(('Eigs:',abs(E), '\t', E, '\n\t', V))
		print('\n')
		return 0
