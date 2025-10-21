import numpy as np
import matplotlib.pyplot as plt

from numpy import zeros,sqrt,pi,dot,exp,sin,cos,array,amax,arange,concatenate,argmin,log10,append, transpose, identity, matmul, kron

from scipy.special import wofz
from scipy.interpolate import interp1d

from scipy.constants import physical_constants, epsilon_0, hbar, c, e, h, k

from sympy import Symbol, cos, sin, simplify, eye, powsimp, powdenest, lambdify, solve, solveset

from sympy.matrices import det, Matrix

import scipy.linalg as la
from scipy.linalg import qr, eig, eigh
import scipy

S=0.5 #Electron spin
gs = -physical_constants['electron g factor'][0]
muB=physical_constants['Bohr magneton'][0] 
kB=physical_constants['Boltzmann constant'][0] 
amu=physical_constants['atomic mass constant'][0] #An atomic mass unit in kg
e0=epsilon_0 #Permittivity of free space
a0=physical_constants['Bohr radius'][0]
kB = physical_constants['Boltzmann constant'][0]

def CalcNumberDensity(T,atom):
	""" Helper function to tidy up code in spectra_SPD.py 
		Calls one of the other functions in this module, based on atom parameter
		
		Temperature in Kelvin
		Number density returned in inverse cubic metres
	"""
	
	if atom in ['Rb85','Rb87','Rb']:
		return numDenRb(T)
	elif atom=='Cs':
		return numDenCs(T)
	elif atom in ['K','K39','K40','K41']:
		return numDenK(T)
	elif atom=='Na':
		return numDenNa(T)
	elif atom in ['Ag107','Ag109','Ag']:
		return numDenAg(T)

def numDenRb(T):
    """Calculates the rubidium number density"""
    if T<312.46:
        p=10.0**(4.857-4215./T)
    else:
        p=10.0**(8.316-4275./T-1.3102*log10(T))
    NumberDensity=101325.0*p/(kB*T)
    return NumberDensity

def numDenK(T):
    '''Potassium number density'''
    if T<336.8:
        p=10.0**(4.961-4646.0/T)
    else:
        p=10.0**(8.233-4693.0/T-1.2403*log10(T))
    NumberDensity=101325.0*p/(kB*T)
    return NumberDensity

def numDenCs(T):
    '''Caesium number density'''
    if T<301.65:
        p=10.0**(4.711-3999./T)
    else:
        p=10.0**(8.232-4062./T-1.3359*log10(T))
    NumberDensity=101325.0*p/(kB*T)
    return NumberDensity

def numDenNa(T):
    '''Sodium number density'''
    if T<370.95:
        p=10.0**(5.298-5603./T)
    else:
        p=10.0**(8.400-5634./T-1.1748*log10(T))
    NumberDensity=101325.0*p/(kB*T)
    return NumberDensity

def numDenAg(T):
    NumberDensity=1e16
    return NumberDensity

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
			
			
			
			'''	
		## OLD and slow method::
			# Sub in values of susceptibility
			DielMat_sub1a = DielMat_sub1.subs(e_x, 0.5*(2.+cL+cR))
			DielMat_sub1a = DielMat_sub1a.subs(e_xy, 0.5j*(cR-cL))
			DielMat_sub1a = DielMat_sub1a.subs(e_z, (1.0+cZ))
			
			# Evaluate and convert to numpy array
			DM = np.array(DielMat_sub1a.evalf())
			DMa = np.zeros((3,3),dtype='complex')
			for ii in range(3):
				for jj in range(3):
					DMa[ii,jj] = np.complex128(DM[ii,jj])
		
			# use scipy to find eigenvector
			#ev1 = Matrix(DMa).nullspace()
			#print 'Sympy: ', ev1
			
			ev1old = nullOld(DMa).T[0]
			#ev1 = null(DMaNP).T
			
			# sub in for ref. index
			n1soln = solns[0].subs(e_x, 0.5*(2.+cL+cR))
			n1soln = n1soln.subs(e_xy, 0.5j*(cR-cL))
			n1soln = n1soln.subs(e_z, (1.0+cZ))
			
			# Populate the refractive index array
			n1old[i] = np.sqrt(np.complex128(n1soln.evalf()))
		## /OLD method
			'''
		
			# NEW method
			
			# Sub in values of susceptibility
			DMaNP = Dsub1(0.5*(2.+cL+cR), 0.5j*(cR-cL), (1.0+cZ))
			#print DMa
			ev1 = null(DMaNP).T
			# Populate the refractive index array
			#n1[i] = np.sqrt(nsub1(0.5*(2.+cL+cR), 0.5j*(cR-cL), (1.0+cZ)))
			
			
			'''
			## METHOD COMPARISON
			
			print 'SymPy:'
			print DMa
			print DMa.shape, type(DMa)
			print 'Numpy'
			print DMaNP
			print DMaNP.shape, type(DMaNP)
			
			print 'Eigenvectors ...'
			print 'Old: ', ev1old			
			print 'New: ',ev1
			'''
			
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
			
			'''
		## OLD
			# Evaluate and convert to numpy array
			DielMat_sub2a = DielMat_sub2.subs(e_x, 0.5*(2.+cL+cR))
			DielMat_sub2a = DielMat_sub2a.subs(e_xy, 0.5j*(cR-cL))
			DielMat_sub2a = DielMat_sub2a.subs(e_z, (1.0+cZ))
			
			DM = np.array(DielMat_sub2a.evalf())
			DMa = np.zeros((3,3),dtype='complex')
			for ii in range(3):
				for jj in range(3):
					DMa[ii,jj] = np.complex128(DM[ii,jj])
			
			# use scipy to find eigenvector
			ev2old = nullOld(DMa).T[0]
			
			# sub in for ref. index
			n2soln = solns[1].subs(e_x, 0.5*(2.+cL+cR))
			n2soln = n2soln.subs(e_xy, 0.5j*(cR-cL))
			n2soln = n2soln.subs(e_z, (1.0+cZ))
			
			# Populate the refractive index array
			n2old[i] = np.sqrt(np.complex128(n2soln.evalf()))
			'''
			
			
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
		
def test_null():
	A = np.matrix([[2,3,5],[-4,2,3],[0,0,0]])
	SymA = Matrix(A)
	
	nv = null(A)
	nvold = nullOld(A)
	
	print((nv.T))
	print((nvold.T[0]))
	print((SymA.nullspace()[0].evalf()))
	
	print((A * nv))
	
def test_solveset():
	x = Symbol('x')
	A = Matrix([[x,2,x*x],[4,5,x],[x,8,9]])
	
	solns = solve(det(A), x)
	solns_set = list(solveset(det(A), x))
	
	print(solns)
	print('\n')
	print(solns_set)
	
	print('\n\n\n')
	print((solns[0]))
	print('\n')
	print((solns_set[0]))
	
	soln_sub = solns[0].subs(x, 1)
	solnset_sub = solns_set[0].subs(x, 1)
	
	s1 = soln_sub.evalf()
	s1set = solnset_sub.evalf()
	
	s2set = solns_set[1].subs(x, 1).evalf()
	
	print(s1)
	print(s1set)
	print(s2set)

def nullOld(A, eps=1e-14):
	""" Find the null eigenvector x of matrix A, such that Ax=0"""
	# Taken with gratitude from http://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix
	u, s, vh = la.svd(A)
	null_mask = (s <= eps)
	null_space = scipy.compress(null_mask, vh, axis=0)
	return scipy.transpose(null_space)	

'''
def null(A, atol=1e-15, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = la.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    print nnz
    ns = vh[nnz:].conj().T
    return ns
'''

# Horizontal polarisers (P_x)
HorizPol_xy = np.matrix([[1,0],[0,0]])
HorizPol_lr = 1./2 * np.matrix([[1,1],[1,1]])

# Vertical polarisers (P_y)
VertPol_xy = np.matrix([[0,0],[0,1]])
VertPol_lr = 1./2 * np.matrix([[1,-1],[-1,1]])

# Linear polarisers at plus 45 degrees wrt x-axis
LPol_P45_xy = 1./2 * np.matrix([[1,1],[1,1]])
LPol_P45_lr = 1./2 * np.matrix([[1,-1.j],[1.j,1]])

# Linear polarisers at minus 45 degrees wrt x-axis
LPol_M45_xy = 1./2 * np.matrix([[1,-1],[-1,1]])
LPol_M45_lr = 1./2 * np.matrix([[1,1.j],[-1.j,1]])

# Left Circular polariser (in circular basis only)
CPol_L_lr = np.matrix([[1,0],[0,0]])
CPol_R_lr = np.matrix([[0,0],[0,1]])

# {{1,0,0},{0,cos(-b),-sin(-b)},{0,sin(-b),cos(-b)}}

# {{cos(a), 0, sin(a)},{0,1,0},{-sin(a),0,cos(a)}}

def rotate_forward(input_vec,phi,theta,test=False):
	"""
	With respect to the lab frame (x,y,z), the magnetic field vector is
	( 	cos(phi) sin(theta)
		sin(phi)
		cos(phi) cos(theta) )
	or in python:
	BVec = np.matrix([[cos(phi)*sin(theta)],[sin(phi)],[cos(phi)*cos(theta)]])
	
	We need to rotate this so the field points along the z' direction in the new coordinates (x',y',z')
	
	To do this we need two transforms - one around the y-axis and one around the x-axis
	
	input_vec must be either 
		[x,y,z] as list or array {input_vec.shape == (3,)} 
		or 
		[[x1,y1,z1],[x2,y2,z2],...] {input_vec.shape == (n,3)}
	"""
	BVec = np.matrix([[cos(phi)*sin(theta)],[sin(phi)],[cos(phi)*cos(theta)]])

	# input_vec is given as a row vector [Ex,Ey,Ez] - translate to column vector [[Ex],[Ey],[Ez]]
	input_vec_col = np.array([input_vec]).T # note - need the extra []'s around input_vec!
	
	# rotation around the x-axis into xz plane
	R1 = np.matrix([[1,0,0],[0,cos(phi),-sin(phi)],[0,sin(phi),cos(phi)]])
	# rotation around the y-axis oriented around z
	R2 = np.matrix([[cos(-theta), 0, sin(-theta)],[0,1,0],[-sin(-theta),0,cos(-theta)]])
	
	if test:
		return R1 * R2 * np.matrix(input_vec_col), R1 * R2 * BVec
	else:
		## apply rotation matrices, and output as row vector [Ex,Ey,Ez] - need another transpose
		return np.array(R1 * R2 * np.matrix(input_vec_col)).T
		
def rotate_back(input_vec,phi,theta):
	""" Reverse the rotation that rotate_foward() method does """
	# input_vec is given as a row vector [Ex,Ey,Ez] - translate to column vector [[Ex],[Ey],[Ez]]
	input_vec_col = np.array([input_vec]).T # note - need the extra []'s around input_vec!
	
	R1 = np.matrix([[1,0,0],[0,cos(phi),-sin(phi)],[0,sin(phi),cos(phi)]])
	R2 = np.matrix([[cos(-theta), 0, sin(-theta)],[0,1,0],[-sin(-theta),0,cos(-theta)]])
	
	# apply inverse matrices in the reverse order
	return np.array(R2.I * R1.I * np.matrix(input_vec_col))

def rotate_around_z(input_vec,phi):
	"""
	Rotate 3d vector around the 3rd dimension, counter-clockwise
	"""
	#print 'input shape: ',input_vec[0].shape
	
	# input_vec is given as a row vector [Ex,Ey,Ez] - translate to column vector [[Ex],[Ey],[Ez]]
	input_vec_col = np.array([input_vec]).T # note - need the extra []'s around input_vec!
	
	# rotation around the z-axis (rotation in the xy plane)
	R1 = np.matrix([[cos(phi),-sin(phi),0],[sin(phi),cos(phi),0],[0,0,1]])
	
	#print np.dot(R1,np.matrix(input_vec_col))
	return np.array(R1 * np.matrix(input_vec_col))

def rotate_around_z2(input_vec,phi):
	"""
	Rotate 3d vector around the 3rd dimension, counter-clockwise
	"""
	#This function adds to rotate_around_z the capability to act on a stacked array of vectors.
	#print 'input shape: ',input_vec[0].shape
	
	# input_vec is given as an array n of row vectors, eg. n x [Ex,Ey,Ez] - translate to array of n column vectors n x [[Ex],[Ey],[Ez]]
	input_vec_col = np.transpose(np.array([input_vec]),(1,2,0)) # note - need the extra []'s around input_vec!
	
	# rotation around the z-axis (rotation in the xy plane)
	R1 = np.array([[cos(phi),-sin(phi),0],[sin(phi),cos(phi),0],[0,0,1]])
	
	#print np.dot(R1,np.matrix(input_vec_col))
	return np.array(R1 @ input_vec_col) #Return rotated vectors as a stacked array
		
def test_forward_rotn():	
	""" Testing ... """

# field along Z
	theta = 0
	phi = 0
	
	print('B-Field along Z')
	
	# X-axis expressed in x',y',z' coords
	X_new, test = rotate_forward([[1],[0],[0]], phi, theta, True)
	print(('This should be (0,0,1) by definition... \n',test))
	print(('X_new: \n',X_new))
	# Y-axis expressed in x',y',z' coords
	Y_new, test = rotate_forward([[0],[1],[0]], phi, theta, True)
	print(('Y_new: \n',Y_new))
	# Z-axis expressed in x',y',z' coords
	Z_new, test = rotate_forward([[0],[0],[1]], phi, theta, True)
	print(('Z_new: \n',Z_new))

# field along Y
	theta = 0
	phi = np.pi/2
	
	print('\n\nB-Field along Y')

	# X-axis expressed in x',y',z' coords
	X_new, test = rotate_forward([[1],[0],[0]], phi, theta, True)
	print(('This should be (0,0,1) by definition... \n',test))
	print(('X_new: \n',X_new))
	# Y-axis expressed in x',y',z' coords
	Y_new, test = rotate_forward([[0],[1],[0]], phi, theta, True)
	print(('Y_new: \n',Y_new))
	# Z-axis expressed in x',y',z' coords
	Z_new, test = rotate_forward([[0],[0],[1]], phi, theta, True)
	print(('Z_new: \n',Z_new))

# field along X
	theta = np.pi/2
	phi = 0
	
	print('\n\nB-Field along X')

	# X-axis expressed in x',y',z' coords
	X_new, test = rotate_forward([[1],[0],[0]], phi, theta, True)
	print(('This should be (0,0,1) by definition... \n',test))
	print(('X_new: \n',X_new))
	# Y-axis expressed in x',y',z' coords
	Y_new, test = rotate_forward([[0],[1],[0]], phi, theta, True)
	print(('Y_new: \n',Y_new))
	# Z-axis expressed in x',y',z' coords
	Z_new, test = rotate_forward([[0],[0],[1]], phi, theta, True)
	print(('Z_new: \n',Z_new))

def test_reverse_rotn():
	""" Testing reverse rotation ... """
	
	lab_frame_E = np.matrix([[0],[1],[0]])
	phi = np.random.random()*np.pi
	theta = np.random.random()*2*np.pi
	
	print('Lab frame input: ')
	print(lab_frame_E)
	
	E_new = rotate_forward(lab_frame_E, phi, theta)
	print('Field: ')
	print(E_new)
	
	E_original = rotate_back(E_new,phi,theta)
	print('Rotated back:')
	print(E_original)

def sz(L,S,I):
    Sz=jz(S)
    gL=int(2*L+1)
    Li=identity(gL)
    gI=int(2*I+1)
    Ii=identity(gI)
    sz=kron(kron(Li,Sz),Ii)
    return sz

def lz(L,S,I):
    gS=int(2*S+1)
    Si=identity(gS)
    Lz=jz(L)
    gI=int(2*I+1)
    Ii=identity(gI)
    lz=kron(kron(Lz,Si),Ii)
    return lz

def Iz(L,S,I):
    gS=int(2*S+1)
    gL=int(2*L+1)
    Si=identity(gS)
    Li=identity(gL)
    Iz_num=jz(I)
    Iz=kron(kron(Li,Si),Iz_num)
    return Iz

def fz(L,S,I):
    gS=int(2*S+1)
    Sz=jz(S)
    Si=identity(gS)
    gL=int(2*L+1)
    Lz=jz(L)
    Li=identity(gL)
    gJ=gL*gS
    Jz=kron(Lz,Si)+kron(Li,Sz)
    Ji=identity(gJ)
    gI=int(2*I+1)
    Iz=jz(I)
    Ii=identity(gI)
    Fz=kron(Jz,Ii)+kron(Ji,Iz)
    return Fz

def Hfs(L,S,I):
    """Provides the L dot S matrix (fine structure)"""
    gS=int(2*S+1) #number of mS values
    Sx=jx(S)
    Sy=jy(S)
    Sz=jz(S)
    Si=identity(gS)

    gL=int(2*L+1)
    Lx=jx(L)
    Ly=jy(L)
    Lz=jz(L)
    Li=identity(gL)

    gJ=gL*gS
    Jx=kron(Lx,Si)+kron(Li,Sx)
    Jy=kron(Ly,Si)+kron(Li,Sy)
    Jz=kron(Lz,Si)+kron(Li,Sz)
    J2=dot(Jx,Jx)+dot(Jy,Jy)+dot(Jz,Jz)

    gI=int(2*I+1)
    Ii=identity(gI)
    gF=gJ*gI
    Fi=identity(gF)
    Hfs=0.5*(kron(J2,Ii)-L*(L+1)*Fi-S*(S+1)*Fi) # fine structure in m_L,m_S,m_I basis
    return Hfs
        
def Hhfs(L,S,I):
    """Provides the I dot J matrix (hyperfine structure interaction)"""
    gS=int(2*S+1)
    Sx=jx(S)
    Sy=jy(S)
    Sz=jz(S)
    Si=identity(gS)

    gL=int(2*L+1)
    Lx=jx(L)
    Ly=jy(L)
    Lz=jz(L)
    Li=identity(gL)

    gJ=gL*gS
    Jx=kron(Lx,Si)+kron(Li,Sx)
    Jy=kron(Ly,Si)+kron(Li,Sy)
    Jz=kron(Lz,Si)+kron(Li,Sz)
    Ji=identity(gJ)
    J2=dot(Jx,Jx)+dot(Jy,Jy)+dot(Jz,Jz)

    gI=int(2*I+1)
    gF=gJ*gI
    Ix=jx(I)
    Iy=jy(I)
    Iz=jz(I)
    Ii=identity(gI)
    Fx=kron(Jx,Ii)+kron(Ji,Ix)
    Fy=kron(Jy,Ii)+kron(Ji,Iy)
    Fz=kron(Jz,Ii)+kron(Ji,Iz)
    Fi=identity(gF)
    F2=dot(Fx,Fx)+dot(Fy,Fy)+dot(Fz,Fz)
    Hhfs=0.5*(F2-I*(I+1)*Fi-kron(J2,Ii))
    return Hhfs

def Bbhfs(L,S,I):
    """Calculates electric quadrupole matrix.

    Calculates the part in square brakets from
    equation (8) in manual
    """
    gS=int(2*S+1)
    Sx=jx(S)
    Sy=jy(S)
    Sz=jz(S)
    Si=identity(gS)

    gL=int(2*L+1)
    Lx=jx(L)
    Ly=jy(L)
    Lz=jz(L)
    Li=identity(gL)

    gJ=gL*gS
    Jx=kron(Lx,Si)+kron(Li,Sx)
    Jy=kron(Ly,Si)+kron(Li,Sy)
    Jz=kron(Lz,Si)+kron(Li,Sz)

    gI=int(2*I+1)
    gF=gJ*gI
    Ix=jx(I)
    Iy=jy(I)
    Iz=jz(I)
    
    Fi=identity(gF)

    IdotJ=kron(Jx,Ix)+kron(Jy,Iy)+kron(Jz,Iz)
    IdotJ2=dot(IdotJ,IdotJ)

    if I != 0:
        Bbhfs=1./(6*I*(2*I-1))*(3*IdotJ2+3./2*IdotJ-I*(I+1)*15./4*Fi)
    else:
        Bbhfs = 0
    return Bbhfs

class Hamiltonian(object):
    """Functions to create the atomic hamiltonian."""

    def __init__(self, Isotope, Trans, gL, Bfield):
        """Ground and excited state Hamiltonian for an isotope"""
        if Isotope=='Rb87':
            atom = Rb87
        elif Isotope=='Rb85':
            atom = Rb85


        elif Isotope=='Ag107':
            atom = Ag107
        elif Isotope=='Ag109':
            atom = Ag109

            
        elif Isotope=='Cs':
            atom = Cs
        elif Isotope=='K39':
            atom = K39
        elif Isotope=='K40':
            atom = K40
        elif Isotope=='K41':
            atom = K41
        elif Isotope=='Na':
            atom = Na
        elif Isotope=='IdealAtom':
            atom = IdealAtom
            transition = IdealD1Transition
            atom_transition = Ideal_D1

        self.atom = atom
		
        if (Trans=='D1') and (Isotope=='Rb85'):
            transition = RbD1Transition
            atom_transition = Rb85_D1
        elif (Trans=='D2') and (Isotope=='Rb85'):
            transition = RbD2Transition
            atom_transition = Rb85_D2
        elif (Trans=='D1') and (Isotope=='Rb87'):
            transition = RbD1Transition
            atom_transition = Rb87_D1
        elif (Trans=='D2') and (Isotope=='Rb87'):
            transition = RbD2Transition
            atom_transition = Rb87_D2

        elif (Trans=='D2') and (Isotope=='Ag107'):
            transition = AgD2Transition
            atom_transition = Ag107_D2
        elif (Trans=='D2') and (Isotope=='Ag109'):
            transition = AgD2Transition
            atom_transition = Ag109_D2

        elif (Trans=='D1') and (Isotope=='Cs'):
            transition = CsD1Transition
            atom_transition = Cs_D1
        elif (Trans=='D2') and (Isotope=='Cs'):
            transition = CsD2Transition
            atom_transition = Cs_D2
        elif (Trans=='D1') and (Isotope=='Na'):
            transition = NaD1Transition
            atom_transition = Na_D1
        elif (Trans=='D2') and (Isotope=='Na'):
            transition = NaD2Transition
            atom_transition = Na_D2
        elif (Trans=='D1') and (Isotope=='K39'):
            transition = KD1Transition
            atom_transition = K39_D1
        elif (Trans=='D2') and (Isotope=='K39'):
            transition = KD2Transition
            atom_transition = K39_D2
        elif (Trans=='D1') and (Isotope=='K40'):
            transition = KD1Transition
            atom_transition = K40_D1
        elif (Trans=='D2') and (Isotope=='K40'):
            transition = KD2Transition
            atom_transition = K40_D2
        elif (Trans=='D1') and (Isotope=='K41'):
            transition = KD1Transition
            atom_transition = K41_D1
        elif (Trans=='D2') and (Isotope=='K41'):
            transition = KD2Transition
            atom_transition = K41_D2
			
        if Bfield == 0.0:
            Bfield += 1e-5 # avoid degeneracy problem..?

        #Useful quantities to return
        self.ds=int((2*S+1)*(2*atom.I+1)) #Dimension of S-term matrix
        self.dp=int(3*(2*S+1)*(2*atom.I+1)) #Dimension of P-term matrix

        self.groundManifold, self.groundEnergies = self.groundStateManifold(atom.gI,atom.I,atom.As,
                                atom_transition.IsotopeShift,Bfield)
        self.excitedManifold, self.excitedEnergies = self.excitedStateManifold(gL,atom.gI,atom.I,
                                atom_transition.Ap,atom_transition.Bp,Bfield)
    
    def groundStateManifold(self,gI,I,A_hyp_coeff,IsotopeShift,Bfield):
        """Function to produce the ground state manifold"""
        ds = int((2*S+1)*(2*I+1))  # total dimension of matrix
        #print 'Matrix dim:', ds
        As = A_hyp_coeff
        # Add the S-term hyperfine interaction
        S_StateHamiltonian = As*Hhfs(0.0,S,I)+IsotopeShift*identity(ds)
        Ez = muB*Bfield*1.e-4/(hbar*2.0*pi*1.0e6)
        S_StateHamiltonian += Ez*(gs*sz(0.0,S,I)+gI*Iz(0.0,S,I)) # Add Zeeman
        EigenSystem = eigh(S_StateHamiltonian)
        EigenValues = EigenSystem[0].real
        EigenVectors = EigenSystem[1]
        stateManifold = append([EigenValues],EigenVectors,axis=0)
        sortedManifold = sorted(transpose(stateManifold),key=(lambda i:i[0]))
        return sortedManifold, EigenValues

    def excitedStateManifold(self,gL,gI,I,A_hyp_coeff,B_hyp_coeff,Bfield):
        """Function to produce the excited state manifold"""
        dp = int(3*(2*S+1)*(2*I+1))  # total dimension of matrix
        # The actual value of FS is unimportant.
        FS = self.atom.FS # Fine structure splitting
        Ap = A_hyp_coeff
        Bp = B_hyp_coeff
        # Add P-term fine and hyperfine interactions
        if Bp==0.0:
            P_StateHamiltonian=FS*Hfs(1.0,S,I)+FS*identity(dp)+Ap*Hhfs(1.0,S,I)
        if Bp!=0.0:
            P_StateHamiltonian=FS*Hfs(1.0,S,I)-(FS/2.0)*identity(dp)+Ap*Hhfs(1.0,S,I)
            P_StateHamiltonian+=Bp*Bbhfs(1.0,S,I) # add p state quadrupole
        E=muB*(Bfield*1.0e-4)/(hbar*2.0*pi*1.0e6)
        # Add magnetic interaction
        P_StateHamiltonian+=E*(gL*lz(1.0,S,I)+gs*sz(1.0,S,I)+gI*Iz(1.0,S,I))
        ep=eigh(P_StateHamiltonian)
        EigenValues=ep[0].real
        EigenVectors=ep[1]
        stateManifold=append([EigenValues],EigenVectors,axis=0)
        sortedManifold=sorted(transpose(stateManifold),key=(lambda i:i[0]))
        return sortedManifold, EigenValues

def jp(jj):
    b = 0
    dim = int(2*jj + 1)
    Jp = np.zeros((dim, dim))
    z = np.arange(dim)
    m = jj - z
    while b < dim - 1:
        mm = m[b + 1]
        Jp[b, b + 1] = np.sqrt(jj * (jj + 1) - mm * (mm + 1))
        b += 1
    return Jp

def jx(jj):
    Jp = jp(jj)
    Jm = Jp.T
    Jx = 0.5 * (Jp + Jm)
    return Jx

def jy(jj):
    Jp = jp(jj)
    Jm = Jp.T
    Jy = 0.5j * (Jm - Jp)
    return Jy

def jz(jj):
    Jp = jp(jj)
    Jm = Jp.T
    Jz = 0.5 * (np.dot(Jp, Jm) - np.dot(Jm, Jp))
    return Jz

class IdealAtom:
	""" Constants for an ideal atom with no hyperfine structure, and only electron spin """
	I = 0
	As = 0
	gI = 0
	mass = 85*amu
	
class Rb85:
    """Constants relating to the rubidium-85 atom"""
    I  = 2.5         #Nuclear spin
    As = 1011.910813 #Ground state hyperfine constant in units of MHz
    gI = -0.00029364 #nuclear spin g-factor
    mass = 84.911789732*amu
    FS = 7.123e6 # Fine-structure splitting

class Rb87:
    """Constants relating to the rubidium-87 atom"""
    I  = 1.5 
    As = 3417.341305452145 
    gI = -0.0009951414 
    mass = 86.909180520*amu
    FS = 7.123e6 # Fine-structure splitting (MHz)

class Ag107:
    """Constants for silver-107 atom"""
    I = 0.5
    As = 1977.0 * 0.8637  # MHz (ground-state hyperfine A constant, Uhlenberg et al 2000)
    gI = 1.234e-4   # convert nuclear g to Bohr magneton units
    mass = 106.90509 * amu
    FS = 0.0  # you can leave this 0 since Ag D1/D2 FS isn't used in practice

class Ag109:
    """Constants for silver-109 atom"""
    I = 0.5
    As = 1977.0  # scaled by ratio of nuclear magnetic moments
    gI = 1.426e-4
    mass = 108.904755 * amu
    FS = 0.0

class Cs:
    """Constants relating to the caesium-133 atom"""
    I  = 3.5         #Nuclear spin
    As = 2298.1579425 #Ground state hyperfine constant in units of MHz
    gI = -0.00039885395 #nuclear spin g-factor
    mass = 132.905451931*amu
    FS = 351725718.50 - 335116048.807 # Fine-structure splitting (MHz)

class K39:
    """Constants relating to the potassium-39 atom"""
    I  = 1.5
    As = 230.8598601
    gI = -0.00014193489
    mass = 38.96370668*amu
    FS = 391016185.94 - 389286074.580 # Fine-structure splitting (MHz)

class K40:
    """Constants relating to the potassium-40 atom"""
    I  = 4.0
    As = -285.7308
    gI = 0.000176490
    mass = 39.96399848*amu
    FS = 391016185.94 - 389286074.580 # Fine-structure splitting (MHz)
	
class K41:
    """Constants relating to the potassium-41 atom"""
    I  = 1.5
    As = 127.0069352
    gI = -0.00007790600
    mass = 40.96182576*amu
    FS = 391016185.94 - 389286074.580 # Fine-structure splitting (MHz)
	
class Na:
    """Constants relating to the sodium-23 atom"""
    I  = 1.5
    As = 885.81306440
    gI = -0.00080461080
    mass = 22.9897692807*amu
    FS = 508.8487162e6 - 508.3331958e6 # Fine-structure splitting (MHz)
	
# Element-Transition constants

class RbD1Transition:
    """Constants relating to the rubidium D1 transition"""
    wavelength=794.978969380e-9 #The weighted linecentre of the rubidium D1 line in m
    wavevectorMagnitude=2.0*pi/wavelength #Magnitude of the wavevector
    NatGamma=5.746 #Rubidium D1 natural linewidth in MHz
    dipoleStrength=3.0*sqrt(e0*hbar*(2.0*NatGamma*(10.0**6))*(wavelength**3)/(8.0*pi))
    v0=377107407.299e6 #The weighted linecentre of the rubidium D1 line in Hz

class RbD2Transition:
    """Constants relating to the rubidium D2 transition"""
    wavelength=780.2413272e-9
    wavevectorMagnitude=2.0*pi/wavelength
    NatGamma=6.065
    dipoleStrength=3.0*sqrt(e0*hbar*(2.0*NatGamma*(10.0**6))*(wavelength**3)/(8.0*pi))
    v0=384230426.6e6

class AgD2Transition:
    """Constants relating to the silver D2 transition"""
    wavelength=328.0680e-9
    wavevectorMagnitude=2.0*pi/wavelength
    NatGamma=22.28
    dipoleStrength=3.0*sqrt(e0*hbar*(2.0*NatGamma*(10.0**6))*(wavelength**3)/(8.0*pi))
    v0= 9.1342e14

class CsD1Transition:
    """Constants relating to the caesium D1 transition"""
    wavelength=894.59295986e-9 #The weighted linecentre of the caesium D1 line in m
    wavevectorMagnitude=2.0*pi/wavelength #Magnitude of the wavevector
    NatGamma=4.584 #Caesium D1 natural linewidth in MHz
    dipoleStrength=3.0*sqrt(e0*hbar*(2.0*NatGamma*(10.0**6))*(wavelength**3)/(8.0*pi))
    v0=335116048.807e6 #The weighted linecentre of the caesium D1 line in Hz

class CsD2Transition:
    """Constants relating to the caesium D2 transition"""
    wavelength=852.34727582e-9
    wavevectorMagnitude=2.0*pi/wavelength
    NatGamma=5.225
    dipoleStrength=3.0*sqrt(e0*hbar*(2.0*NatGamma*(10.0**6))*(wavelength**3)/(8.0*pi))
    v0=351725718.50e6

class KD1Transition:
    """Constants relating to the potassium D1 transition"""
    wavelength=770.108353667e-9 #The linecentre of Potassium in metres
    wavevectorMagnitude=2.0*pi/wavelength #Magnitude of the wavevector
    NatGamma=5.956 #Potassium D1 natural linewidth in MHz
    dipoleStrength=3.0*sqrt(e0*hbar*(2.0*NatGamma*(10.0**6))*(wavelength**3)/(8.0*pi))
    v0=389286074.580e6 #Potassium linecentre D1 transition in Hz

class KD2Transition:
    """Constants relating to the potassium D2 transition"""
    wavelength=766.700890602e-9 #The linecentre of Potassium in metres
    wavevectorMagnitude=2.0*pi/wavelength #Magnitude of the wavevector
    NatGamma=6.035 #Potassium D1 natural linewidth in MHz
    dipoleStrength=3.0*sqrt(e0*hbar*(2.0*NatGamma*(10.0**6))*(wavelength**3)/(8.0*pi))
    v0=391016185.94e6 #Potassium linecentre D2 transition in Hz

class NaD1Transition:
    """Constants relating to the sodium D1 transition"""
    wavelength=589.7558147e-9 #The weighted linecentre of the sodium D1 line
    wavevectorMagnitude=2.0*pi/wavelength
    NatGamma=9.765
    dipoleStrength=3.0*sqrt(e0*hbar*(2.0*NatGamma*(10.0**6))*(wavelength**3)/(8.0*pi))
    v0=508.3331958e12 #Sodium D1 linecentre in Hz

class NaD2Transition:
    """Constants relating to the sodium D2 transition"""
    wavelength=589.1583264e-9 #The weighted linecentre of the sodium D1 line
    wavevectorMagnitude=2.0*pi/wavelength
    NatGamma=9.7946
    dipoleStrength=3.0*sqrt(e0*hbar*(2.0*NatGamma*(10.0**6))*(wavelength**3)/(8.0*pi))
    v0=508.8487162e12 #Sodium D1 linecentre in Hz

class IdealD1Transition:
    """Constants relating to the rubidium D1 transition"""
    wavelength = 780e-9 #The weighted linecentre of the rubidium D1 line in m
    wavevectorMagnitude = 2.0*pi/wavelength #Magnitude of the wavevector
    NatGamma = 6 #Rubidium D1 natural linewidth in MHz
    dipoleStrength = 3.0*sqrt(e0*hbar*(2.0*NatGamma*(10.0**6))*(wavelength**3)/(8.0*pi))
    v0 = 377107407.299e6 #The weighted linecentre of the rubidium D1 line in Hz

# transitions dictionary
transitions = {'RbD1':RbD1Transition, 'RbD2':RbD2Transition,
						'CsD1':CsD1Transition, 'CsD2':CsD2Transition,
						'KD1':KD1Transition, 'KD2':KD2Transition,
						'NaD1':NaD1Transition, 'NaD2':NaD2Transition,
                        'AgD2': AgD2Transition,
						'IdealD1':IdealD1Transition
					}
					
# Isotope-Transition constants

class Ideal_D1:
	Ap = 0
	Bp = 0
	IsotopeShift = 0
	
class Rb85_D1:
    """Constants relating to rubidium-85 and the D1 transition"""
    #Hyperfine constants in units of MHz
    Ap = 120.640
    Bp = 0.0
    IsotopeShift = 21.624 #MHz. Shifts the ground (S) manifold up.

class Rb87_D1:
    """Constants relating to rubidium-87 and the D1 transition"""
    #Hyperfine constants in units of MHz
    Ap = 406.147
    Bp = 0.0
    IsotopeShift = -56.077 #MHz

class Rb85_D2:
    """Constants relating to rubidium-85 and the D2 transition"""
    #Hyperfine constants in units of MHz
    Ap = 25.038
    Bp = 26.011
    IsotopeShift = 21.734 #MHz

class Rb87_D2:
    """Constants relating to rubidium-87 and the D2 transition"""
    #Hyperfine constants in units of MHz
    Ap = 84.7185
    Bp = 12.4965
    IsotopeShift = -56.361 #MHz

class Ag107_D2:
    """Constants relating to rubidium-85 and the D2 transition"""
    #Hyperfine constants in units of MHz
    Ap = (75/2)*0.8637
    Bp = 0
    IsotopeShift = 476.6 #MHz

class Ag109_D2:
    """Constants relating to rubidium-87 and the D2 transition"""
    #Hyperfine constants in units of MHz
    Ap = 75e6/2
    Bp = 0
    IsotopeShift = 0 #MHz

class Cs_D1:
    """Constants relating to the caesium-133 atom and the D1 transition"""
    #Hyperfine constants in units of MHz
    Ap = 291.9201
    Bp = 0.0
    IsotopeShift = 0.0 #Only one isotope for Caesium

class Cs_D2:
    """Constants relating to the caesium-133 atom and the D2 transition"""
    #Hyperfine constants in units of MHz
    Ap = 50.28827
    Bp = -0.4934
    IsotopeShift = 0.0

class K39_D1:
    """Constants relating to the potassium-39 atom and the D1 transition"""
    #Hyperfine constants in units of MHz
    Ap = 27.775
    Bp = 0.0
    IsotopeShift = 15.864 #MHz. If positive, shifts the ground (S) manifold up.

class K39_D2:
    """Constants relating to the potassium-39 atom and the D2 transition"""
    #Hyperfine constants in units of MHz
    Ap = 6.093
    Bp = 2.786
    IsotopeShift = 15.91

class K40_D1:
    """Constants relating to the potassium-40 atom and the D1 transition"""
    #Hyperfine constants in units of MHz
    Ap = -34.523
    Bp = 0.0
    IsotopeShift = -109.773

class K40_D2:
    """Constants relating to the potassium-40 atom and the D2 transition"""
    #Hyperfine constants in units of MHz
    Ap = -7.585
    Bp = -3.445
    IsotopeShift = -110.11

class K41_D1:
    """Constants relating to the potassium-41 atom and the D1 transition"""
    #Hyperfine constants in units of MHz
    Ap = 127.0069352
    Bp = 0.0
    IsotopeShift = -219.625

class K41_D2:
    """Constants relating to the potassium-41 atom and the D2 transition"""
    #Hyperfine constants in units of MHz
    Ap = 3.363
    Bp = 3.351
    IsotopeShift = -220.27

class Na_D1:
    """Constants relating to the sodium-23 atom and the D1 transition"""
    #Hyperfine constants in units of MHz
    Ap = 94.44
    Bp = 0
    IsotopeShift = 0.0 #Only one isotope.

class Na_D2:
    """Constants relating to the sodium-23 atom and the D2 transition"""
    #Hyperfine constants in units of MHz
    Ap = 18.534
    Bp = 2.724
    IsotopeShift = 0.0 #Only one isotope.

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

# Default values for parameters
p_dict_defaults = {	'Elem':'Rb', 'Dline':'D2', 
							'lcell':75e-3,'Bfield':0., 'T':20., 
							'GammaBuf':0., 'shift':0.,
							# Polarisation of light
							'theta0':0., 'Pol':50., 
							# B-field angle w.r.t. light k-vector
							'Btheta':0, 'Bphi':0,
							'Constrain':True, 'DoppTemp':20.,
							'rb85frac':72.17, 'K40frac':0.01, 'K41frac':6.73,'Ag107frac':51.839, 'Ag109frac':48.161,
							'BoltzmannFactor':True}

def FreqStren(groundLevels,excitedLevels,groundDim,
			  excitedDim,Dline,hand,BoltzmannFactor=True,T=293.16):
	""" 
	Calculate transition frequencies and strengths by taking dot
	products of relevant parts of the ground / excited state eigenvectors 
	"""
	
	transitionFrequency = zeros(groundDim*2*groundDim) #Initialise lists
	transitionStrength = zeros(groundDim*2*groundDim)
	transNo = 0
	
	## Boltzmann factor: -- only really needed when energy splitting of ground state is large
	if BoltzmannFactor:
		groundEnergies = array(groundLevels)[:,0].real
		lowestEnergy = groundLevels[0][argmin(groundLevels,axis=0)[0]].real
		BoltzDist = exp(-(groundEnergies-lowestEnergy)*h*1.e6/(kB*T)) #Not normalised
		#print BoltzDist
		BoltzDist = BoltzDist/BoltzDist.sum() # Normalised

	# select correct columns of matrix corresponding to delta mL = -1, 0, +1
	if hand=='Right':
		bottom = 1
		top = groundDim + 1
	elif hand=='Z':
		bottom = groundDim + 1
		top = 2*groundDim + 1
	elif hand=='Left':
		bottom = 2*groundDim+1
		top = excitedDim+1

	# select correct rows of matrix corresponding to D1 / D2 lines
	if Dline=='D1':
		interatorList = list(range(groundDim))
	elif Dline=='D2':
		interatorList = list(range(groundDim,excitedDim))\
	
	# find difference in energies and do dot product between (all) ground states
	# 	and selected parts of excited state matrix
	for gg in range(groundDim):
		for ee in interatorList:
			cleb = dot(groundLevels[gg][1:],excitedLevels[ee][bottom:top]).real
			cleb2 = cleb*cleb
			if cleb2 > 0.0005: #If negligable don't calculate.
				transitionFrequency[transNo] = int((-groundLevels[gg][0].real
											  +excitedLevels[ee][0].real))
				# We choose to perform the ground manifold reduction (see
				# equation (4) in manual) here for convenience.
				
				## Boltzmann factor:
				if BoltzmannFactor: 
					transitionStrength[transNo] = 1./3 * cleb2 * BoltzDist[gg]
				else:
					transitionStrength[transNo] = 1./3 * cleb2 * 1./groundDim
				
				transNo += 1

	#print 'No transitions (ElecSus): ',transNo
	return transitionFrequency, transitionStrength, transNo

def add_voigt(d,DoppTemp,atomMass,wavenumber,gamma,voigtwidth,
		ltransno,lenergy,lstrength,
		rtransno,renergy,rstrength,
		ztransno,zenergy,zstrength):
	xpts = len(d)
	npts = 2*voigtwidth+1
	detune = 2.0*pi*1.0e6*(arange(npts)-voigtwidth) # Angular detuning (2pi Hz)
	wavenumber =  wavenumber + detune/c # Allow the wavenumber to change (large detuning)
	u = sqrt(2.0*kB*DoppTemp/atomMass)
	ku = wavenumber*u
	
	# Fadeeva function:
	a = gamma/ku
	b = detune/ku
	y = 1.0j*(0.5*sqrt(pi)/ku)*wofz(b+0.5j*a)
	
	ab = y.imag
	disp = y.real
	#interpolate lineshape functions
	f_ab = interp1d(detune,ab)
	f_disp = interp1d(detune,disp)
	
	#Add contributions from all transitions to user defined detuning axis
	lab = zeros(xpts)
	ldisp = zeros(xpts)
	for line in range(ltransno+1):
		xc = lenergy[line]
		lab += lstrength[line]*f_ab(2.0*pi*(d-xc)*1.0e6)
		ldisp += lstrength[line]*f_disp(2.0*pi*(d-xc)*1.0e6)
	rab = zeros(xpts)
	rdisp = zeros(xpts)
	for line in range(rtransno+1):
		xc = renergy[line]
		rab += rstrength[line]*f_ab(2.0*pi*(d-xc)*1.0e6)
		rdisp += rstrength[line]*f_disp(2.0*pi*(d-xc)*1.0e6)
	zab = zeros(xpts)
	zdisp = zeros(xpts)
	for line in range(ztransno+1):
		xc = zenergy[line]
		zab += zstrength[line]*f_ab(2.0*pi*(d-xc)*1.0e6)
		zdisp += zstrength[line]*f_disp(2.0*pi*(d-xc)*1.0e6)
	return lab, ldisp, rab, rdisp, zab, zdisp

def calc_chi(X, p_dict,verbose=False):			   
	"""
	Computes the complex susceptibility for sigma plus/minus and pi transitions as a 1D array

	Arguments:
	
		X: 		Detuning axis (float, list, or numpy array) in MHz
		p_dict: 	Dictionary containing all parameters (the order of parameters is therefore not important)
				
			Dictionary keys:

			Key				DataType	Unit		Description
			---				---------	----		-----------
			Elem	   			str			--			The chosen alkali element.
			Dline	  			str			--			Specifies which D-line transition to calculate for (D1 or D2)
			
			# Experimental parameters
			Bfield	 			float			Gauss	Magnitude of the applied magnetic field
			T		 			float			Celsius	Temperature used to calculate atomic number density
			GammaBuf   	float			MHz		Extra lorentzian broadening (usually from buffer gas 
															but can be any extra homogeneous broadening)
			shift	  			float			MHz		A global frequency shift of the atomic resonance frequencies

			DoppTemp   	float			Celsius	Temperature linked to the Doppler width (used for
															independent Doppler width and number density)
			Constrain  		bool			--			If True, overides the DoppTemp value and sets it to T

			# Elemental abundancies, where applicable
			rb85frac   		float			%			percentage of rubidium-85 atoms
			K40frac			float			%			percentage of potassium-40 atoms
			K41frac			float			%			percentage of potassium-41 atoms
			
			lcell	  			float			m			length of the vapour cell
			theta0	 		float			degrees	Linear polarisation angle w.r.t. to the x-axis
			Pol				float			%			Percentage of probe beam that drives sigma minus (50% = linear polarisation)
			
			NOTE: If keys are missing from p_dict, default values contained in p_dict_defaults will be loaded.
			
			Any additional keys in the dict are ignored.
	"""
	
	# get parameters from dictionary
	if 'Elem' in list(p_dict.keys()):
		Elem = p_dict['Elem']
	else:
		Elem = p_dict_defaults['Elem']
	if 'Dline' in list(p_dict.keys()):
		Dline = p_dict['Dline']
	else:
		Dline = p_dict_defaults['Dline']
	if 'T' in list(p_dict.keys()):
		T = p_dict['T']
	else:
		T = p_dict_defaults['T']
	if 'Bfield' in list(p_dict.keys()):
		Bfield = p_dict['Bfield']
	else:
		Bfield = p_dict_defaults['Bfield']
	if 'GammaBuf' in list(p_dict.keys()):
		GammaBuf = p_dict['GammaBuf']
	else:
		GammaBuf = p_dict_defaults['GammaBuf']
	if 'shift' in list(p_dict.keys()):
		shift = p_dict['shift']
	else:
		shift = p_dict_defaults['shift']
	if 'Constrain' in list(p_dict.keys()):
		Constrain = p_dict['Constrain']
	else:
		Constrain = p_dict_defaults['Constrain']
	if 'rb85frac' in list(p_dict.keys()):
		rb85frac = p_dict['rb85frac']
	else:
		rb85frac = p_dict_defaults['rb85frac']
	if 'DoppTemp' in list(p_dict.keys()):
		DoppTemp = p_dict['DoppTemp']
	else:
		DoppTemp = p_dict_defaults['DoppTemp']
	if 'K40frac' in list(p_dict.keys()):
		K40frac = p_dict['K40frac']
	else:
		K40frac = p_dict_defaults['K40frac']
	if 'K41frac' in list(p_dict.keys()):
		K41frac = p_dict['K41frac']
	else:
		K41frac = p_dict_defaults['K41frac']





	if 'Ag107frac' in list(p_dict.keys()):
		Ag107frac = p_dict['Ag107frac']
	else:
		Ag107frac = p_dict_defaults['Ag107frac']

	if 'Ag109frac' in list(p_dict.keys()):
		Ag109frac = p_dict['Ag109frac']
	else:
		Ag109frac = p_dict_defaults['Ag109frac']





	if 'BoltzmannFactor' in list(p_dict.keys()):
		BoltzmannFactor =  p_dict['BoltzmannFactor']
	else:
		BoltzmannFactor =  p_dict_defaults['BoltzmannFactor']
	
	
	if verbose: print(('Temperature: ', T, '\tBfield: ', Bfield))
	# convert X to array if needed (does nothing otherwise)
	X = array(X)
   
	# Change to fraction from %
	rb85frac = rb85frac/100.0
	K40frac  = K40frac/100.0
	K41frac  = K41frac/100.0

	Ag107frac = Ag107frac/100.0
	Ag109frac = Ag109frac/100.0


	if Bfield==0.0:
		Bfield = 0.0001 #To avoid degeneracy problem at B = 0.

	# Rubidium energy levels
	if Elem=='Rb':
		rb87frac=1.0-rb85frac  # Rubidium-87 fraction
		if rb85frac!=0.0: #Save time if no rubidium-85 required
			Rb85atom = Rb85
			#Hamiltonian(isotope,transition,gL,Bfield)
			Rb85_ES = Hamiltonian('Rb85',Dline,1.0,Bfield)

			# Rb-85 allowed transitions for light driving sigma minus
			lenergy85, lstrength85, ltransno85 = FreqStren(
													Rb85_ES.groundManifold,
													Rb85_ES.excitedManifold,
													Rb85_ES.ds,Rb85_ES.dp,
													Dline,'Left',BoltzmannFactor,T+273.16)		  

			# Rb-85 allowed transitions for light driving sigma plus
			renergy85, rstrength85, rtransno85 = FreqStren(
													Rb85_ES.groundManifold,
													Rb85_ES.excitedManifold,
													Rb85_ES.ds,Rb85_ES.dp,
													Dline,'Right',BoltzmannFactor,T+273.16)
			
			# Rb-85 allowed transitions for light driving pi
			zenergy85, zstrength85, ztransno85 = FreqStren(
													Rb85_ES.groundManifold,
													Rb85_ES.excitedManifold,
													Rb85_ES.ds,Rb85_ES.dp,
													Dline,'Z',BoltzmannFactor,T+273.16)
			

		if rb87frac!=0.0:
			Rb87atom = Rb87
			#Hamiltonian(isotope,transition,gL,Bfield)
			Rb87_ES = Hamiltonian('Rb87',Dline,1.0,Bfield)
			# Rb-87 allowed transitions for light driving sigma minus
			lenergy87, lstrength87, ltransno87 = FreqStren(
													Rb87_ES.groundManifold,
													Rb87_ES.excitedManifold,
													Rb87_ES.ds,Rb87_ES.dp,
													Dline,'Left',BoltzmannFactor,T+273.16)

			# Rb-87 allowed transitions for light driving sigma plus
			renergy87, rstrength87, rtransno87 = FreqStren(
													Rb87_ES.groundManifold,
													Rb87_ES.excitedManifold,
													Rb87_ES.ds,Rb87_ES.dp,
													Dline,'Right',BoltzmannFactor,T+273.16)

			# Rb-87 allowed transitions for light driving sigma plus
			zenergy87, zstrength87, ztransno87 = FreqStren(
													Rb87_ES.groundManifold,
													Rb87_ES.excitedManifold,
													Rb87_ES.ds,Rb87_ES.dp,
													Dline,'Z',BoltzmannFactor,T+273.16)
													
			
													
		if Dline=='D1':
			transitionConst = RbD1Transition
		elif Dline=='D2':
			transitionConst = RbD2Transition

		if (rb85frac!=0.0) and (rb87frac!=0.0):
			AllEnergyLevels = concatenate((lenergy87,lenergy85,
														renergy87,renergy85,
														zenergy87,zenergy85))
		elif (rb85frac!=0.0) and (rb87frac==0.0):
			AllEnergyLevels = concatenate((lenergy85,renergy85,zenergy85))
		elif (rb85frac==0.0) and (rb87frac!=0.0):
			AllEnergyLevels = concatenate((lenergy87,renergy87,zenergy87))

	# Caesium energy levels
	elif Elem=='Cs':
		CsAtom = Cs
		Cs_ES = Hamiltonian('Cs',Dline,1.0,Bfield)

		lenergy, lstrength, ltransno = FreqStren(Cs_ES.groundManifold,
												 Cs_ES.excitedManifold,
												 Cs_ES.ds,Cs_ES.dp,Dline,
												 'Left',BoltzmannFactor,T+273.16)
		renergy, rstrength, rtransno = FreqStren(Cs_ES.groundManifold,
												 Cs_ES.excitedManifold,
												 Cs_ES.ds,Cs_ES.dp,Dline,
												 'Right',BoltzmannFactor,T+273.16)		
		zenergy, zstrength, ztransno = FreqStren(Cs_ES.groundManifold,
												 Cs_ES.excitedManifold,
												 Cs_ES.ds,Cs_ES.dp,Dline,
												 'Z',BoltzmannFactor,T+273.16)

		if Dline=='D1':
			transitionConst = CsD1Transition
		elif Dline =='D2':
			transitionConst = CsD2Transition
		AllEnergyLevels = concatenate((lenergy,renergy,zenergy))

	# Sodium energy levels
	elif Elem=='Na':
		NaAtom = Na
		Na_ES = Hamiltonian('Na',Dline,1.0,Bfield)

		lenergy, lstrength, ltransno = FreqStren(Na_ES.groundManifold,
												 Na_ES.excitedManifold,
												 Na_ES.ds,Na_ES.dp,Dline,
												 'Left',BoltzmannFactor,T+273.16)
		renergy, rstrength, rtransno = FreqStren(Na_ES.groundManifold,
												 Na_ES.excitedManifold,
												 Na_ES.ds,Na_ES.dp,Dline,
												 'Right',BoltzmannFactor,T+273.16)
		zenergy, zstrength, ztransno = FreqStren(Na_ES.groundManifold,
												 Na_ES.excitedManifold,
												 Na_ES.ds,Na_ES.dp,Dline,
												 'Z',BoltzmannFactor,T+273.16)												 
		if Dline=='D1':
			transitionConst = NaD1Transition
		elif Dline=='D2':
			transitionConst = NaD2Transition
		AllEnergyLevels = concatenate((lenergy,renergy,zenergy))

	#Potassium energy levels <<<<< NEED TO ADD Z-COMPONENT >>>>>
	elif Elem=='K':
		K39frac=1.0-K40frac-K41frac #Potassium-39 fraction
		if K39frac!=0.0:
			K39atom = K39
			K39_ES = Hamiltonian('K39',Dline,1.0,Bfield)
			
			lenergy39, lstrength39, ltransno39 = FreqStren(
													K39_ES.groundManifold,
													K39_ES.excitedManifold,
													K39_ES.ds,K39_ES.dp,Dline,
													'Left',BoltzmannFactor,T+273.16)
			renergy39, rstrength39, rtransno39 = FreqStren(
													K39_ES.groundManifold,
													K39_ES.excitedManifold,
													K39_ES.ds,K39_ES.dp,Dline,
													'Right',BoltzmannFactor,T+273.16)
			zenergy39, zstrength39, ztransno39 = FreqStren(K39_ES.groundManifold,
													K39_ES.excitedManifold,
													K39_ES.ds,K39_ES.dp,Dline,
													'Z',BoltzmannFactor,T+273.16)
		if K40frac!=0.0:
			K40atom = K40
			K40_ES = Hamiltonian('K40',Dline,1.0,Bfield)
			
			lenergy40, lstrength40, ltransno40 = FreqStren(
													K40_ES.groundManifold,
													K40_ES.excitedManifold,
													K40_ES.ds,K40_ES.dp,Dline,
													'Left',BoltzmannFactor,T+273.16)
			renergy40, rstrength40, rtransno40 = FreqStren(
													K40_ES.groundManifold,
													K40_ES.excitedManifold,
													K40_ES.ds,K40_ES.dp,Dline,
													'Right',BoltzmannFactor,T+273.16)
			zenergy40, zstrength40, ztransno40 = FreqStren(K40_ES.groundManifold,
													K40_ES.excitedManifold,
													K40_ES.ds,K40_ES.dp,Dline,
													'Z',BoltzmannFactor,T+273.16)
		if K41frac!=0.0:
			K41atom = K41
			K41_ES = Hamiltonian('K41',Dline,1.0,Bfield)
			
			lenergy41, lstrength41, ltransno41 = FreqStren(
													K41_ES.groundManifold,
													K41_ES.excitedManifold,
													K41_ES.ds,K41_ES.dp,Dline,
													'Left',BoltzmannFactor,T+273.16)
			renergy41, rstrength41, rtransno41 = FreqStren(
													K41_ES.groundManifold,
													K41_ES.excitedManifold,
													K41_ES.ds,K41_ES.dp,Dline,
													'Right',BoltzmannFactor,T+273.16)
			zenergy41, zstrength41, ztransno41 = FreqStren(K41_ES.groundManifold,
													K41_ES.excitedManifold,
													K41_ES.ds,K41_ES.dp,Dline,
													'Z',BoltzmannFactor,T+273.16)
		if Dline=='D1':
			transitionConst = KD1Transition
		elif Dline=='D2':
			transitionConst = KD2Transition
		if K39frac!=0.0:
			AllEnergyLevels = concatenate((lenergy39,renergy39,zenergy39))
		if K40frac!=0.0 and K39frac!=0.0:
			AllEnergyLevels = concatenate((AllEnergyLevels,lenergy40,renergy40,zenergy40))
		elif K40frac!=0.0 and K39frac==0.0:
			AllEnergyLevels = concatenate((lenergy40,renergy40,zenergy40))
		if K41frac!=0.0 and (K39frac!=0.0 or K40frac!=0.0):
			AllEnergyLevels = concatenate((AllEnergyLevels,lenergy41,renergy41,zenergy41))
		elif K41frac!=0.0 and (K39frac==0.0 and K40frac==0.0):
			AllEnergyLevels = concatenate((lenergy41,renergy41,zenergy41))

	# Silver energy levels
	elif Elem == 'Ag':
		# Define isotope(s)
		Ag107frac = 1.0 - Ag109frac  # Example: natural silver isotopes

		if Ag107frac != 0.0:
			Ag107atom = Ag107
			Ag107_ES = Hamiltonian('Ag107', Dline, 1.0, Bfield)

			lenergy107, lstrength107, ltransno107 = FreqStren(
				Ag107_ES.groundManifold,
				Ag107_ES.excitedManifold,
				Ag107_ES.ds, Ag107_ES.dp, Dline,
				'Left', BoltzmannFactor, T + 273.16
			)
			renergy107, rstrength107, rtransno107 = FreqStren(
				Ag107_ES.groundManifold,
				Ag107_ES.excitedManifold,
				Ag107_ES.ds, Ag107_ES.dp, Dline,
				'Right', BoltzmannFactor, T + 273.16
			)
			zenergy107, zstrength107, ztransno107 = FreqStren(
				Ag107_ES.groundManifold,
				Ag107_ES.excitedManifold,
				Ag107_ES.ds, Ag107_ES.dp, Dline,
				'Z', BoltzmannFactor, T + 273.16
			)

		if Ag109frac != 0.0:
			Ag109atom = Ag109
			Ag109_ES = Hamiltonian('Ag109', Dline, 1.0, Bfield)

			lenergy109, lstrength109, ltransno109 = FreqStren(
				Ag109_ES.groundManifold,
				Ag109_ES.excitedManifold,
				Ag109_ES.ds, Ag109_ES.dp, Dline,
				'Left', BoltzmannFactor, T + 273.16
			)
			renergy109, rstrength109, rtransno109 = FreqStren(
				Ag109_ES.groundManifold,
				Ag109_ES.excitedManifold,
				Ag109_ES.ds, Ag109_ES.dp, Dline,
				'Right', BoltzmannFactor, T + 273.16
			)
			zenergy109, zstrength109, ztransno109 = FreqStren(
				Ag109_ES.groundManifold,
				Ag109_ES.excitedManifold,
				Ag109_ES.ds, Ag109_ES.dp, Dline,
				'Z', BoltzmannFactor, T + 273.16
			)

		# Choose transition constants for the selected D-line
		if Dline == 'D1':
			transitionConst = AgD2Transition
		elif Dline == 'D2':
			transitionConst = AgD2Transition

		# Combine isotope contributions
		if Ag107frac != 0.0 and Ag109frac != 0.0:
			AllEnergyLevels = concatenate((
				lenergy107, renergy107, zenergy107,
				lenergy109, renergy109, zenergy109
			))
		elif Ag107frac != 0.0:
			AllEnergyLevels = concatenate((lenergy107, renergy107, zenergy107))
		elif Ag109frac != 0.0:
			AllEnergyLevels = concatenate((lenergy109, renergy109, zenergy109))

#Calculate Voigt

	DoppTemp = T #Set doppler temperature to the number density temperature
	T += 273.15
	DoppTemp += 273.15

	# For thin cells: Don't add Doppler effect, by setting DopplerTemperature to near-zero
	# can then convolve with different velocity distribution later on
		
	d = (array(X)-shift) #Linear detuning
	xpts = len(d)
	maxdev = amax(abs(d))

	if Elem=='Rb':
		NDensity=numDenRb(T) #Calculate number density
	elif Elem=='Cs':
		NDensity=numDenCs(T)
	elif Elem=='K':
		NDensity=numDenK(T)
	elif Elem=='Na':
		NDensity=numDenNa(T)
	elif Elem== 'Ag':
		NDensity=numDenAg(T)

	#Calculate lorentzian broadening and shifts
	gamma0 = 2.0*pi*transitionConst.NatGamma*1.e6
	if Dline=='D1': # D1 self-broadening parameter
		gammaself = 2.0*pi*gamma0*NDensity*\
					(transitionConst.wavelength/(2.0*pi))**(3)
	elif Dline=='D2': # D2 self-broadening parameter
		gammaself = 2.0*pi*gamma0*NDensity*1.414213562373095*\
					(transitionConst.wavelength/(2.0*pi))**(3)
	gamma = gamma0 + gammaself
	gamma = gamma + (2.0*pi*GammaBuf*1.e6) #Add extra lorentzian broadening
		
	maxShiftedEnergyLevel = amax(abs(AllEnergyLevels)) #integer value in MHz
	voigtwidth = int(1.1*(maxdev+maxShiftedEnergyLevel))
	wavenumber = transitionConst.wavevectorMagnitude
	dipole = transitionConst.dipoleStrength
	prefactor=2.0*NDensity*dipole**2/(hbar*e0)

	if Elem=='Rb':
		lab85, ldisp85, rab85, rdisp85, zab85, zdisp85 = 0,0,0,0,0,0
		lab87, ldisp87, rab87, rdisp87, zab87, zdisp87 = 0,0,0,0,0,0
		if rb85frac!=0.0:
			lab85, ldisp85, rab85, rdisp85, zab85, zdisp85 = add_voigt(d,DoppTemp,
													   Rb85atom.mass,
													   wavenumber,gamma,
													   voigtwidth,
													   ltransno85,lenergy85,lstrength85,
													   rtransno85,renergy85,rstrength85,
													   ztransno85,zenergy85,zstrength85)
		if rb87frac!=0.0:
			lab87, ldisp87, rab87, rdisp87, zab87, zdisp87 = add_voigt(d,DoppTemp,
													   Rb87atom.mass,
													   wavenumber,gamma,
													   voigtwidth,
													   ltransno87,lenergy87,lstrength87,
													   rtransno87,renergy87,rstrength87,
													   ztransno87,zenergy87,zstrength87)
		# Make the parts of the susceptibility
		ChiRealLeft= prefactor*(rb85frac*ldisp85+rb87frac*ldisp87)
		ChiRealRight= prefactor*(rb85frac*rdisp85+rb87frac*rdisp87)
		ChiRealZ = prefactor*(rb85frac*zdisp85 + rb87frac*zdisp87)
		ChiImLeft = prefactor*(rb85frac*lab85+rb87frac*lab87)
		ChiImRight = prefactor*(rb85frac*rab85+rb87frac*rab87)
		ChiImZ = prefactor*(rb85frac*zab85 + rb87frac*zab87)
		

	elif Elem=='Cs':
		lab, ldisp, rab, rdisp, zab, zdisp = 0,0,0,0,0,0
		lab, ldisp, rab, rdisp, zab, zdisp = add_voigt(d,DoppTemp,CsAtom.mass,wavenumber,
										   gamma,voigtwidth,
										   ltransno,lenergy,lstrength,
										   rtransno,renergy,rstrength,
										   ztransno,zenergy,zstrength)
		ChiRealLeft = prefactor*ldisp
		ChiRealRight = prefactor*rdisp
		ChiRealZ = prefactor*zdisp
		ChiImLeft = prefactor*lab
		ChiImRight = prefactor*rab
		ChiImZ = prefactor*zab
	elif Elem=='Na':
		lab, ldisp, rab, rdisp, zab, zdisp = 0,0,0,0,0,0
		lab, ldisp, rab, rdisp, zab, zdisp = add_voigt(d,DoppTemp,NaAtom.mass,wavenumber,
										   gamma,voigtwidth,
										   ltransno,lenergy,lstrength,
										   rtransno,renergy,rstrength,
										   ztransno,zenergy,zstrength)
		ChiRealLeft = prefactor*ldisp
		ChiRealRight = prefactor*rdisp
		ChiRealZ = prefactor*zdisp
		ChiImLeft = prefactor*lab
		ChiImRight = prefactor*rab
		ChiImZ = prefactor*zab
	elif Elem=='K':
		lab39, ldisp39, rab39, rdisp39, zab39, zdisp39 = 0,0,0,0,0,0
		lab40, ldisp40, rab40, rdisp40, zab40, zdisp40 = 0,0,0,0,0,0
		lab41, ldisp41, rab41, rdisp41, zab41, zdisp41 = 0,0,0,0,0,0
		if K39frac!=0.0:
			lab39, ldisp39, rab39, rdisp39, zab39, zdisp39 = add_voigt(d,DoppTemp,K39atom.mass,
													   wavenumber,gamma,
													   voigtwidth,
													   ltransno39,lenergy39,lstrength39,
													   rtransno39,renergy39,rstrength39,
													   ztransno39,zenergy39,zstrength39)
		if K40frac!=0.0:
			lab40, ldisp40, rab40, rdisp40, zab40, zdisp40 = add_voigt(d,DoppTemp,K40atom.mass,
													   wavenumber,gamma,
													   voigtwidth,
													   ltransno40,lenergy40,lstrength40,
													   rtransno40,renergy40,rstrength40,
													   ztransno40,zenergy40,zstrength40)
		if K41frac!=0.0:
			lab41, ldisp41, rab41, rdisp41, zab41, zdisp41 = add_voigt(d,DoppTemp,K41atom.mass,
													   wavenumber,gamma,
													   voigtwidth,
													   ltransno41,lenergy41,lstrength41,
													   rtransno41,renergy41,rstrength41,
													   ztransno41,zenergy41,zstrength41)

		ChiRealLeft = prefactor*(K39frac*ldisp39+K40frac\
									*ldisp40+K41frac*ldisp41)
		ChiRealRight = prefactor*(K39frac*rdisp39+K40frac\
									*rdisp40+K41frac*rdisp41)
		ChiRealZ = prefactor*(K39frac*zdisp39+K40frac\
									*zdisp40+K41frac*zdisp41)
		ChiImLeft = prefactor*(K39frac*lab39+K40frac*lab40+K41frac*lab41)
		ChiImRight = prefactor*(K39frac*rab39+K40frac*rab40+K41frac*rab41)
		ChiImZ = prefactor*(K39frac*zab39+K40frac*zab40+K41frac*zab41)
	elif Elem == 'Ag':

		# Initialise all Lorentzian/dispersion components
		lab107, ldisp107, rab107, rdisp107, zab107, zdisp107 = 0,0,0,0,0,0
		lab109, ldisp109, rab109, rdisp109, zab109, zdisp109 = 0,0,0,0,0,0

		# Add isotope-resolved Voigt profiles
		if Ag107frac != 0.0:
			lab107, ldisp107, rab107, rdisp107, zab107, zdisp107 = add_voigt(
				d, DoppTemp, Ag107atom.mass, wavenumber,
				gamma, voigtwidth,
				ltransno107, lenergy107, lstrength107,
				rtransno107, renergy107, rstrength107,
				ztransno107, zenergy107, zstrength107
			)

		if Ag109frac != 0.0:
			lab109, ldisp109, rab109, rdisp109, zab109, zdisp109 = add_voigt(
				d, DoppTemp, Ag109atom.mass, wavenumber,
				gamma, voigtwidth,
				ltransno109, lenergy109, lstrength109,
				rtransno109, renergy109, rstrength109,
				ztransno109, zenergy109, zstrength109
			)

		# Build real and imaginary parts of susceptibility
		ChiRealLeft  = prefactor * (Ag107frac * ldisp107 + Ag109frac * ldisp109)
		ChiRealRight = prefactor * (Ag107frac * rdisp107 + Ag109frac * rdisp109)
		ChiRealZ     = prefactor * (Ag107frac * zdisp107 + Ag109frac * zdisp109)

		ChiImLeft  = prefactor * (Ag107frac * lab107 + Ag109frac * lab109)
		ChiImRight = prefactor * (Ag107frac * rab107 + Ag109frac * rab109)
		ChiImZ     = prefactor * (Ag107frac * zab107 + Ag109frac * zab109)


	# Reconstruct total susceptibility and index of refraction
	totalChiPlus = ChiRealLeft + 1.j*ChiImLeft
	totalChiMinus = ChiRealRight + 1.j*ChiImRight
	totalChiZ = ChiRealZ + 1.j*ChiImZ
	
	return totalChiPlus, totalChiMinus, totalChiZ

def get_Efield(X, E_in, Chi, p_dict, verbose=False):
	""" 
	Most general form of calculation - return the electric field vector E_out. 
	Can use Jones matrices to calculate all other experimental spectra from 
	this, as in the get_spectra2() method
	
	Electric field is in the lab frame, in the X/Y/Z basis:
		- light propagation is along the Z axis, X is horizontal and Y is vertical dimension
		
	To change between x/y and L/R bases, one may use:
			E_left = 1/sqrt(2) * ( E_x - i.E_y )
			E_right = 1/sqrt(2) * ( E_x + i.E_y )
	(See BasisChanger.py module)
	
	Allows calculation with non-uniform B fields by slicing the cell with total length L into 
	n smaller parts with length L/n - assuming that the field is uniform over L/n,
	which can be checked by convergence testing. E_out can then be used as the new E_in 
	for each subsequent cell slice.
	
	Different to get_spectra() in that the input electric field, E_in, must be defined, 
	and the only quantity returned is the output electric field, E_out.
	
	Arguments:
	
		X:			Detuning in MHz
		E_in:		Array of [E_x, E_y, E_z], complex
						If E_x/y/z are each 1D arrays (with the same dimensions as X) 
						then the polarisation depends on detuning - used e.g. when simulating
						a non-uniform magnetic field
						If E_x/y/z are single (complex) values, then the input polarisation is
						assumed to be independent of the detuning.
		Chi:		(3,len(X)) array of susceptibility values, for sigma+, sigma-, and pi transitions
		p_dict:	Parameter dictionary - see get_spectra() docstring for details.
	
	Returns:
		
		E_out:	Detuning-dependent output Electric-field vector.
					2D-Array of [E_x, E_y, E_z] where E_x/y/z are 1D arrays with same dimensions
					as the detuning axis.
					The output polarisation always depends on the detuning, in the presence of an
					applied magnetic field.
	"""

	# if not already numpy arrays, convert
	X = array(X)
	E_in = array(E_in)
	if verbose:
		print('Electric field input:')
		print(E_in)
	
	#print 'Input Efield shape: ',E_in.shape
	# check detuning axis X has the correct dimensions. If not, make it so.
	if E_in.shape != (3,len(X)):
		#print 'E field not same size as detuning'
		if E_in.shape == (3,):
			E_in = np.array([np.ones(len(X))*E_in[0],np.ones(len(X))*E_in[1],np.ones(len(X))*E_in[2]])
		else:
			raise ValueError( 'ERROR in method get_Efield(): INPUT ELECTRIC FIELD E_in BADLY DEFINED' )
	
	#print 'New Efield shape: ', E_in.shape

	# fill in required dictionary keys from defaults if not given
	if 'lcell' in list(p_dict.keys()):
		lcell = p_dict['lcell']
	else:
		lcell = p_dict_defaults['lcell']
		
	if 'Elem' in list(p_dict.keys()):
		Elem = p_dict['Elem']
	else:
		Elem = p_dict_defaults['Elem']
	if 'Dline' in list(p_dict.keys()):
		Dline = p_dict['Dline']
	else:
		Dline = p_dict_defaults['Dline']
	
	## get magnetic field spherical coordinates
	# defaults to 0,0 i.e. B aligned with kvector of light (Faraday)
	if 'Bfield' in list(p_dict.keys()):
		Bfield = p_dict['Bfield']
	else:
		Bfield = p_dict_defaults['Bfield']
	if 'Btheta' in list(p_dict.keys()):
		Btheta = p_dict['Btheta']
	else:
		Btheta = p_dict_defaults['Btheta']
	if 'Bphi' in list(p_dict.keys()):
		Bphi = p_dict['Bphi']
	else:
		Bphi = p_dict_defaults['Bphi']
		
	# direction of wavevector (for double-pass geometry simulations)
	if 'wavevector_dirn' in list(p_dict.keys()):
		wavevector_dirn = p_dict['wavevector_dirn']
	else:
		wavevector_dirn = 1
	
	if wavevector_dirn == -1:
		# light propagating backwards, but always calculate assuming light travelling in the +z direction,
		# therefore flip B-field 180 degrees around (can use either theta or phi for this)
		Btheta += np.pi
	
	# get wavenumber
	transition = transitions[Elem+Dline]
	wavenumber = transition.wavevectorMagnitude * wavevector_dirn

	# get susceptibility (already calculated, input to this method)
	ChiPlus, ChiMinus, ChiZ = Chi

	# Rotate initial Electric field so that B field lies in x-z plane
	# (Effective polarisation rotation)
	E_xz = rotate_around_z(E_in.T,Bphi)
		
	# Find eigen-vectors for propagation and create rotation matrix
	RM_ary, n1, n2 = solve_diel(ChiPlus,ChiMinus,ChiZ,Btheta,Bfield)

	# take conmplex conjugate of rotation matrix
	RM_ary = RM_ary.conjugate()
	
	# propagation matrix
	PropMat = np.array(
				[	[exp(1.j*n1*wavenumber*lcell),np.zeros(len(n1)),np.zeros(len(n1))],
					[np.zeros(len(n1)),exp(1.j*n2*wavenumber*lcell),np.zeros(len(n1))],
					[np.zeros(len(n1)),np.zeros(len(n1)),np.ones(len(n1))]	])
	#print 'prop matrix shape:',PropMat.T.shape
	#print 'prop mat [0]: ', PropMat.T[0]
	
	# calcualte output field - a little messy to make it work nicely with array operations
	# - need to play around with matrix dimensions a bit
	# Effectively this does this, element-wise: E_out_xz = RotMat.I * PropMat * RotMat * E_xz
	
	E_xz = np.reshape(E_xz.T, (len(X),3,1))
	
	# E_out_xz = np.zeros((len(X),3,1),dtype='complex')
	# E_out = np.zeros_like(E_out_xz)
	# for i in range(len(X)):
	# 	#print 'Propagation Matrix:\n',PropMat.T[i]
	# 	#print 'Rotation matrix:\n',RM_ary.T[i]
	# 	#inverse rotation matrix
	# 	RMI_ary = np.matrix(RM_ary.T[i].T).I
	# 	#print 'Inverse rotation matrix:\n',RMI_ary
		
	# 	E_out_xz[i] = RMI_ary * np.matrix(PropMat.T[i]) * np.matrix(RM_ary.T[i].T)*np.matrix(E_xz[i]) 
	# 	#print 'E out xz i: ',E_out_xz[i].T
	# 	E_out[i] = RM.rotate_around_z(E_out_xz[i].T[0],-Bphi)

	ary_of_RMI_ary = np.linalg.inv(np.transpose(RM_ary.T,(0,2,1))) #Gives full array of RMI matrices
	fast_E_out_xz = matmul(matmul(matmul(ary_of_RMI_ary,PropMat.T),np.transpose(RM_ary.T,(0,2,1))),E_xz)
	fast_E_out = rotate_around_z2(np.transpose(fast_E_out_xz,(0,2,1))[::,0],-Bphi) 

	#print 'E out [0]: ',E_out[0]
	#print 'E out shape: ',E_out.shape
	
	## return electric field vector - can then use Jones matrices to do everything else
	#return E_out.T[0], np.matrix(RM_ary.T[i])
	return fast_E_out.T[0], np.matrix(RM_ary.T[-1])

def get_spectra(X, E_in, p_dict, outputs=None):
	""" 
	Calls get_Efield() to get Electric field, then use Jones matrices 
	to calculate experimentally useful quantities.
	
	Alias for the get_spectra2 method in libs.spectra.
	
	Inputs:
		detuning_range [ numpy 1D array ] 
			The independent variable and defines the detuning points over which to calculate. Values in MHz
		
		E_in [ numpy 1/2D array ] 
			Defines the input electric field vector in the xyz basis. The z-axis is always the direction of propagation (independent of the magnetic field axis), and therefore the electric field should be a plane wave in the x,y plane. The array passed to this method should be in one of two formats:
				(1) A 1D array of (Ex,Ey,Ez) which is the input electric field for all detuning values;
				or
				(2) A 2D array with dimensions (3,len(detuning_range)) - i.e. each detuning has a different electric field associated with it - which will happen on propagation through a birefringent/dichroic medium
		
		p_dict [ dictionary ]
			Dictionary containing all parameters (the order of parameters is therefore not important)
				Dictionary keys:
	
				Key				DataType	Unit		Description
				---				---------	----		-----------
				Elem	   			str			--			The chosen alkali element.
				Dline	  			str			--			Specifies which D-line transition to calculate for (D1 or D2)
				
				# Experimental parameters
				Bfield	 			float			Gauss	Magnitude of the applied magnetic field
				T		 			float			Celsius	Temperature used to calculate atomic number density
				GammaBuf   	float			MHz		Extra lorentzian broadening (usually from buffer gas 
																but can be any extra homogeneous broadening)
				shift	  			float			MHz		A global frequency shift of the atomic resonance frequencies
				DoppTemp   	float			Celsius	Temperature linked to the Doppler width (used for
																independent Doppler width and number density)
				Constrain  		bool			--			If True, overides the DoppTemp value and sets it to T

				# Elemental abundancies, where applicable
				rb85frac   		float			%			percentage of rubidium-85 atoms
				K40frac			float			%			percentage of potassium-40 atoms
				K41frac			float			%			percentage of potassium-41 atoms
				
				lcell	  			float			m			length of the vapour cell
				theta0	 		float			degrees	Linear polarisation angle w.r.t. to the x-axis
				Pol				float			%			Percentage of probe beam that drives sigma minus (50% = linear polarisation)
				
				NOTE: If keys are missing from p_dict, default values contained in p_dict_defaults will be loaded.
		
		outputs: an iterable (list,tuple...) of strings that defines which spectra are returned, and in which order.
			If not specified, defaults to None, in which case a default set of outputs is returned, which are:
				S0, S1, S2, S3, Ix, Iy, I_P45, I_M45, alphaPlus, alphaMinus, alphaZ
	
	Returns:
		A list of output arrays as defined by the 'outputs' keyword argument.
		
		
	Example usage:
		To calculate the room temperature absorption of a 75 mm long Cs reference cell in an applied magnetic field of 100 G aligned along the direction of propagation (Faraday geometry), between -10 and +10 GHz, with an input electric field aligned along the x-axis:
		
		detuning_range = np.linspace(-10,10,1000)*1e3 # GHz to MHz conversion
		E_in = np.array([1,0,0])
		p_dict = {'Elem':'Cs', 'Dline':'D2', 'Bfield':100, 'T':21, 'lcell':75e-3}
		
		[Transmission] = calculate(detuning_range,E_in,p_dict,outputs=['S0'])
		
		
		More examples are available in the /tests/ folder
	"""
	
	# get some parameters from p dictionary
	
	# need in try/except or equiv.
	if 'Elem' in list(p_dict.keys()):
		Elem = p_dict['Elem']
	else:
		Elem = p_dict_defaults['Elem']
	if 'Dline' in list(p_dict.keys()):
		Dline = p_dict['Dline']
	else:
		Dline = p_dict_defaults['Dline']
	if 'shift' in list(p_dict.keys()):
		shift = p_dict['shift']
	else:
		shift = p_dict_defaults['shift']
	if 'lcell' in list(p_dict.keys()):
		lcell = p_dict['lcell']
	else:
		lcell = p_dict_defaults['lcell']
	if 'theta0' in list(p_dict.keys()):
		theta0 = p_dict['theta0']
	else:
		theta0 = p_dict_defaults['theta0']
	if 'Pol' in list(p_dict.keys()):
		Pol = p_dict['Pol']
	else:
		Pol = p_dict_defaults['Pol']

	# get wavenumber
	transition = transitions[Elem+Dline]

	wavenumber = transition.wavevectorMagnitude
	
	# Calculate Susceptibility
	ChiPlus, ChiMinus, ChiZ = calc_chi(X, p_dict)
	Chi = [ChiPlus, ChiMinus, ChiZ]
	
	# Complex refractive index
	nPlus = sqrt(1.0+ChiPlus) #Complex refractive index driving sigma plus transitions
	nMinus = sqrt(1.0+ChiMinus) #Complex refractive index driving sigma minus transitions
	nZ = sqrt(1.0+ChiZ) # Complex index driving pi transitions

	# convert (if necessary) detuning axis X to np array
	if type(X) in (int, float, int):
		X = np.array([X])
	else:
		X = np.array(X)

	# Calculate E_field
	E_out, R = get_Efield(X, E_in, Chi, p_dict)
	#print 'Output E field (Z): \n', E_out[2]
	

	## Apply Jones matrices
	
	# Transmission - total intensity - just E_out**2 / E_in**2
	E_in = np.array(E_in)
	if E_in.shape == (3,):
			E_in = np.array([np.ones(len(X))*E_in[0],np.ones(len(X))*E_in[1],np.ones(len(X))*E_in[2]])
	
	# normalised by input intensity
	I_in = (E_in * E_in.conjugate()).sum(axis=0)
	
	S0 = (E_out * E_out.conjugate()).sum(axis=0) / I_in
	
	Iz = (E_out[2] * E_out[2].conjugate()).real / I_in
	
	Transmission = S0
	
	
	## Some quantities from Faraday geometry don't make sense when B and k not aligned, but leave them here for historical reasons
	TransLeft = exp(-2.0*nPlus.imag*wavenumber*lcell)
	TransRight = exp(-2.0*nMinus.imag*wavenumber*lcell)
	
	# Faraday rotation angle (including incident linear polarisation angle)
	phiPlus = wavenumber*nPlus.real*lcell
	phiMinus = wavenumber*nMinus.real*lcell
	phi = (phiMinus-phiPlus)/2.0 
	##
	
	#Stokes parameters

	#S1#
	Ex = np.array(HorizPol_xy * E_out[:2])
	Ix =  (Ex * Ex.conjugate()).sum(axis=0) / I_in
	Ey =  np.array(VertPol_xy * E_out[:2])
	Iy =  (Ey * Ey.conjugate()).sum(axis=0) / I_in
	
	S1 = Ix - Iy
	
	#S2#
	E_P45 =  np.array(LPol_P45_xy * E_out[:2])
	E_M45 =  np.array(LPol_M45_xy * E_out[:2])
	I_P45 = (E_P45 * E_P45.conjugate()).sum(axis=0) / I_in
	I_M45 = (E_M45 * E_M45.conjugate()).sum(axis=0) / I_in
	
	S2 = I_P45 - I_M45
	
	#S3#
	# change to circular basis
	E_out_lrz = xyz_to_lrz(E_out)
	El =  np.array(CPol_L_lr * E_out_lrz[:2])
	Er =  np.array(CPol_R_lr * E_out_lrz[:2])
	Il = (El * El.conjugate()).sum(axis=0) / I_in
	Ir = (Er * Er.conjugate()).sum(axis=0) / I_in
	
	S3 = Ir - Il
	
	Ir = Ir.real
	Il = Il.real
	Ix = Ix.real
	Iy = Iy.real

	## (Real part) refractive indices
	#nMinus = nPlus.real
	#nPlus = nMinus.real

	## Absorption coefficients - again not a physically relevant quantity anymore since propagation is not as simple as k * Im(Chi) * L in a non-Faraday geometry
	alphaPlus = 2.0*nMinus.imag*wavenumber
	alphaMinus = 2.0*nPlus.imag*wavenumber
	alphaZ = 2.0*nZ.imag*wavenumber

	# Refractive/Group indices for left/right/z also no longer make any sense
	#d = (array(X)-shift) #Linear detuning
	#dnWRTv = derivative(d,nMinus.real)
	#GIPlus = nMinus.real + (X + transition.v0*1.0e-6)*dnWRTv
	#dnWRTv = derivative(d,nPlus.real)
	#GIMinus = nPlus.real + (X + transition.v0*1.0e-6)*dnWRTv
		
	# Valid outputs
	op = {'S0':S0, 'S1':S1, 'S2':S2, 'S3':S3, 'Ix':Ix, 'Iy':Iy, 'Il':Il, 'Ir':Ir, 
				'I_P45':I_P45, 'I_M45':I_M45, 
				'alphaPlus':alphaPlus, 'alphaMinus':alphaMinus, 'alphaZ':alphaZ, 
				'E_out':E_out, 
				'Chi':Chi, 'ChiPlus':ChiPlus, 'ChiMinus':ChiMinus, 'ChiZ':ChiZ
			}
	
	if (outputs == None) or ('All' in outputs):
		# Default - return 'all' outputs (as used by GUI)
		return S0.real,S1.real,S2.real,S3.real,Ix.real,Iy.real,I_P45.real,I_M45.real,alphaPlus,alphaMinus,alphaZ
	else:
	# Return the variable names mentioned in the outputs list of strings
		# the strings in outputs must exactly match the local variable names here!
		return [op[output_str] for output_str in outputs]
	
def output_list():
	""" Helper method that returns a list of all possible variables that get_spectra can return """
	tstr = " \
	All possible outputs from the get_spectra method: \n\n\
	Variable Name		Description \n \
	S0						Total transmission through the cell (Ix + Iy) \n\
	S1						Stokes parameter - Ix - Iy \n\
	S2						Stokes parameter - I_45 - I_-45 \n\
	S3						Stokes parameter - I- - I+ \n\
	TransLeft			Transmission of only left-circularly polarised light \n\
	TransRight			Transmission of only right-circularly polarised light \n\
	ChiPlus				Complex susceptibility of left-circularly polarised light \n\
	ChiMinus				Complex susceptibility of right-circularly polarised light \n\
	nPlus					Complex Refractive index of left-circularly polarised light \n\
	nMinus				Complex Refractive index of right-circularly polarised light \n\
	phiPlus				Rotation of linear polarisation caused by sigma-plus transitions \n\
	phiMinus				Rotation of linear polarisation caused by sigma-minus transitions \n\
	phi					Total rotation of linear polarisation \n\
	Ix						Intensity of light transmitted through a linear polariser aligned with the x-axis \n\
	Iy						Intensity of light transmitted through a linear polariser aligned with the y-axis \n\
	Ir						Intensity of right-circularly polarised light\n\
	Il						Intensity of left-circularly polarised light\n\
	alphaPlus			Absorption coefficient due to sigma-plus transitions \n\
	alphaMinus			Absorption coefficient due to sigma-minus transitions \n\
	GIMinus				Group index of left-circularly polarised light \n\
	GIPlus				Group index of right-circularly polarised light \n\
	"	
	print(tstr)

def main():
	""" General test method """
	
	p_dict = {'Bfield':700,'rb85frac':1,'Btheta':88*np.pi/180,'Bphi':0*np.pi/180,'lcell':75e-3,'T':84,'Dline':'D2','Elem':'Cs'}
	chiL,chiR,chiZ = calc_chi(np.linspace(-3500,3500,10),p_dict)
	
	#print 'ez: ',chiZ + 1 # ez / e0
	#print 'ex: ',0.5*(2+chiL+chiR) # ex / e0
	#print 'exy: ',0.5j*(chiR-chiL) # exy / e0
	
	RotMat, n1, n2 = solve_diel(chiL,chiR,chiZ,88*np.pi/180)
	print((RotMat.shape))

def calculation_time_analysis():
	""" Test method for looking at timing performance """

	p_dict = {'Bfield':700,'rb85frac':1,'Btheta':88*np.pi/180,'Bphi':0*np.pi/180,'lcell':75e-3,'T':84,'Dline':'D2','Elem':'Cs'}
	chiL,chiR,chiZ = calc_chi([-3500],p_dict)
	
	for angle in [0, np.pi/32, np.pi/16, np.pi/8, np.pi/4, np.pi/2]:
		print(('Angle (degrees): ',angle*180/np.pi))
		RotMat, n1, n2 = solve_diel(chiL,chiR,chiZ,angle)
		
def test_equivalence():
	""" Test numeric vs analytic solutions """
	
	
	#analytic
	p_dict = {'Bfield':15000,'rb85frac':1,'Btheta':0*np.pi/180,'Bphi':0*np.pi/180,'lcell':1e-3,'T':84,'Dline':'D2','Elem':'Rb'}
	chiL1,chiR1,chiZ1 = calc_chi([-18400],p_dict)
	RotMat1, n11, n21 = solve_diel(chiL1,chiR1,chiZ1,0,150,force_numeric=False)
	
	#numeric
	chiL2, chiR2, chiZ2 = chiL1, chiR1, chiZ1
	#chiL2,chiR2,chiZ2 = sp.calc_chi([-18400],p_dict)
	RotMat2, n12, n22 = solve_diel(chiL2,chiR2,chiZ2,0,150,force_numeric=True)
	
	print('RM 1')
	print(RotMat1)

	print('RM 2')
	print(RotMat2)	
	
	print('n1_1 (analytic)')
	print(n11)
	print('n1_2')
	print(n12)
	print('n2_1 (analytic)')
	print(n21)
	print('n2_2')
	print(n22)
	
	print('chi1')
	print((chiL1, chiR1, chiZ1))

	print('chi2')
	print((chiL2, chiR2, chiZ2))
	
#if __name__ == '__main__':
#	test_equivalence()

Detuning=np.linspace(-10,10,1000)*1e3 #Detuning range between -10 and 10 GHz. Needs to be input in MHz
E_in=np.array([1,0,0]) #Horizontal Linear Light input. We define E_in = [Ex,Ey,Ez]
p_dict={'Elem':'Ag','Dline':'D2','T':25,'lcell':75e-3,'Bfield':0,'Btheta':0} #A 75 mm cell of natural abundance Rb at 20C. No bfield and hence no angle Btheta between the k-vector and the mag field. 
[S0,S1,S2,S3,E_out,Ix,Iy]=get_spectra(Detuning,E_in,p_dict,outputs=['S0','S1','S2','S3','E_out','Ix','Iy'])

plt.plot(Detuning/1e3,S0)

plt.show()