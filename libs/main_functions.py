import numpy as np
from numpy import zeros,sqrt,pi,dot,exp,array,amax,arange,concatenate,argmin, matmul
import matplotlib.pyplot as plt

from scipy.special import wofz
from scipy.interpolate import interp1d

from scipy.constants import physical_constants, epsilon_0, hbar, c, h

S=0.5 #Electron spin
gs = -physical_constants['electron g factor'][0]
muB=physical_constants['Bohr magneton'][0] 
kB=physical_constants['Boltzmann constant'][0] 
amu=physical_constants['atomic mass constant'][0] #An atomic mass unit in kg
e0=epsilon_0 #Permittivity of free space
a0=physical_constants['Bohr radius'][0]
kB = physical_constants['Boltzmann constant'][0]

from libs import atomic_constants as ac
from libs import numDensity as nd
from libs import solve_diel as solvd
from libs import rotations as rot
from libs import Hamiltonian as ht
from libs import convert_basis as cb

# Default values for parameters
p_dict_defaults = {	'Elem':'Rb', 'Dline':'D2', 
							'lcell':75e-3,'Bfield':0., 'T':20., 
							'GammaBuf':0., 'shift':0.,
							# Polarisation of light
							'theta0':0., 'Pol':50., 
							# B-field angle w.r.t. light k-vector
							'Btheta':0, 'Bphi':0,
							'Constrain':True, 'DoppTemp':20.,
							'Rb85frac':72.17, 'K40frac':0.01, 'K41frac':6.73,'Ag107frac':51.839,
							'BoltzmannFactor':True, 'AgNumden': 3e15, 'Isotope_Combination': 0}

def FreqStren(groundLevels, excitedLevels, groundDim,
			  excitedDim, Dline, hand,
			  BoltzmannFactor=True, T=293.16,
			  group_F=False, ground_F_lookup=None, excited_F_lookup=None):

	""" 
	Calculate transition frequencies and strengths by taking dot
	products of relevant parts of the ground / excited state eigenvectors 
	"""
	
	transitionFrequency = zeros(groundDim*2*groundDim) #Initialise lists
	transitionStrength = zeros(groundDim*2*groundDim)
	transNo = 0
	
	groundLevelsT = []
	excitedLevelsT = []
	transitionFrequencyT = []

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
	
	Fg_all, Fe_all = [], []


	# find difference in energies and do dot product between (all) ground states
	# 	and selected parts of excited state matrix
	for gg in range(groundDim):
		for ee in interatorList:
			cleb = dot(groundLevels[gg][1:],excitedLevels[ee][bottom:top]).real
			cleb2 = cleb*cleb
			if cleb2 > 0.0005: #If negligible don't calculate.
				if group_F and (ground_F_lookup is not None) and (excited_F_lookup is not None):
					Fg_all.append(ground_F_lookup[gg])
					Fe_all.append(excited_F_lookup[ee])
				transitionFrequency[transNo] = int((-groundLevels[gg][0].real
											  +excitedLevels[ee][0].real))
				# We choose to perform the ground manifold reduction (see
				# equation (4) in manual) here for convenience.
				transitionFrequencyT.append(transitionFrequency[transNo])
				groundLevelsT.append(groundLevels[gg][0].real)
				excitedLevelsT.append(excitedLevels[ee][0].real)
				
				## Boltzmann factor:
				if BoltzmannFactor: 
					transitionStrength[transNo] = 1./3 * cleb2 * BoltzDist[gg]
				else:
					transitionStrength[transNo] = 1./3 * cleb2 * 1./groundDim
				
				print(f"  Transition {transNo}: ground={gg}, excited={ee}, "
		  		f"ΔE={transitionFrequency[transNo]:.2f} MHz, strength={transitionStrength[transNo]:.3e}")

				transNo += 1

	#print 'No transitions (ElecSus): ',transNo
	print(f"[FreqStren] {Dline} {hand} → {transNo} hyperfine transitions found")

	if not group_F:
		return transitionFrequency, transitionStrength, transNo

	# --- Group by (Fg, Fe) ---
	tf = np.array(transitionFrequency[:transNo], dtype=float)
	ts = np.array(transitionStrength[:transNo], dtype=float)
	Fg_all = np.array(Fg_all)
	Fe_all = np.array(Fe_all)

	pairs = list(zip(Fg_all, Fe_all))
	unique_pairs = sorted(set(pairs))

	grouped_freqs = []
	grouped_strengths = []
	for (Fg, Fe) in unique_pairs:
		m = (Fg_all == Fg) & (Fe_all == Fe)
		s_sum = ts[m].sum()
		f_avg = (tf[m] * ts[m]).sum() / s_sum
		grouped_freqs.append(f_avg)
		grouped_strengths.append(s_sum)

	# Print summary
	for (Fg, Fe), f, s in zip(unique_pairs, grouped_freqs, grouped_strengths):
		print(f"  Fg={Fg:g} → Fe={Fe:g}, ΔE={f:.2f} MHz, strength={s:.3e}")

	return np.array(grouped_freqs), np.array(grouped_strengths), len(unique_pairs)


	#return transitionFrequency, transitionStrength, transNo

def add_voigt_old(d,DoppTemp,atomMass,wavenumber,gamma,voigtwidth,
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
	
	'''
	# Example: only compute the nth transition
	n = 0  # or any valid index

	lab = zeros(xpts)
	ldisp = zeros(xpts)
	if n < ltransno:
		xc = lenergy[n]
		lab += lstrength[n]*f_ab(2.0*pi*(d-xc)*1.0e6)
		ldisp += lstrength[n]*f_disp(2.0*pi*(d-xc)*1.0e6)

	rab = zeros(xpts)
	rdisp = zeros(xpts)
	if n < rtransno:
		xc = renergy[n]
		rab += rstrength[n]*f_ab(2.0*pi*(d-xc)*1.0e6)
		rdisp += rstrength[n]*f_disp(2.0*pi*(d-xc)*1.0e6)

	zab = zeros(xpts)
	zdisp = zeros(xpts)
	if n < ztransno:
		xc = zenergy[n]
		zab += zstrength[n]*f_ab(2.0*pi*(d-xc)*1.0e6)
		zdisp += zstrength[n]*f_disp(2.0*pi*(d-xc)*1.0e6)
	'''

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

def add_voigt(d, DoppTemp, atomMass, wavenumber, gamma, voigtwidth,
			  ltransno, lenergy, lstrength,
			  rtransno, renergy, rstrength,
			  ztransno, zenergy, zstrength):
	"""
	Computes individual Voigt profiles for each transition and stores them
	in 2D arrays of shape (n_transitions + 1, xpts), where the last row
	contains the summed total profile.
	"""

	from scipy.special import wofz
	from scipy.interpolate import interp1d
	import numpy as np

	xpts = len(d)
	npts = 2 * voigtwidth + 1
	detune = 2.0 * np.pi * 1.0e6 * (np.arange(npts) - voigtwidth)  # Angular detuning (2π Hz)
	wavenumber = wavenumber + detune / c  # Allow wavenumber to vary (large detuning)
	u = np.sqrt(2.0 * kB * DoppTemp / atomMass)
	ku = wavenumber * u

	# Faddeeva (complex error) function
	a = gamma / ku
	b = detune / ku
	y = 1.0j * (0.5 * np.sqrt(np.pi) / ku) * wofz(b + 0.5j * a)
	ab = y.imag
	disp = y.real

	# Interpolators
	f_ab = interp1d(detune, ab, fill_value="extrapolate")
	f_disp = interp1d(detune, disp, fill_value="extrapolate")

	# --- Allocate 2D arrays ---
	lab = np.zeros((ltransno + 1, xpts))
	ldisp = np.zeros((ltransno + 1, xpts))
	rab = np.zeros((rtransno + 1, xpts))
	rdisp = np.zeros((rtransno + 1, xpts))
	zab = np.zeros((ztransno + 1, xpts))
	zdisp = np.zeros((ztransno + 1, xpts))

	# --- Compute Voigt profiles per transition ---
	for i in range(ltransno):
		xc = lenergy[i]
		lab[i, :] = lstrength[i] * f_ab(2.0 * np.pi * (d - xc) * 1.0e6)
		ldisp[i, :] = lstrength[i] * f_disp(2.0 * np.pi * (d - xc) * 1.0e6)

	for i in range(rtransno):
		xc = renergy[i]
		rab[i, :] = rstrength[i] * f_ab(2.0 * np.pi * (d - xc) * 1.0e6)
		rdisp[i, :] = rstrength[i] * f_disp(2.0 * np.pi * (d - xc) * 1.0e6)

	for i in range(ztransno):
		xc = zenergy[i]
		zab[i, :] = zstrength[i] * f_ab(2.0 * np.pi * (d - xc) * 1.0e6)
		zdisp[i, :] = zstrength[i] * f_disp(2.0 * np.pi * (d - xc) * 1.0e6)

	# --- Compute summed total and store as last row ---
	lab[-1, :] = lab[:-1, :].sum(axis=0)
	ldisp[-1, :] = ldisp[:-1, :].sum(axis=0)
	rab[-1, :] = rab[:-1, :].sum(axis=0)
	rdisp[-1, :] = rdisp[:-1, :].sum(axis=0)
	zab[-1, :] = zab[:-1, :].sum(axis=0)
	zdisp[-1, :] = zdisp[:-1, :].sum(axis=0)

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
	if 'Rb85frac' in list(p_dict.keys()):
		Rb85frac = p_dict['Rb85frac']
	else:
		Rb85frac = p_dict_defaults['Rb85frac']
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
	if 'BoltzmannFactor' in list(p_dict.keys()):
		BoltzmannFactor =  p_dict['BoltzmannFactor']
	else:
		BoltzmannFactor =  p_dict_defaults['BoltzmannFactor']
	if 'Isotope_Combination' in list(p_dict.keys()):
		Isotope_Combination =  p_dict['Isotope_Combination']
	else:
		Isotope_Combination =  p_dict_defaults['Isotope_Combination']
	
	
	if verbose: print(('Temperature: ', T, '\tBfield: ', Bfield))
	# convert X to array if needed (does nothing otherwise)
	X = array(X)
   
	# Change to fraction from %
	Rb85frac = Rb85frac/100.0
	print(Rb85frac)
	K40frac  = K40frac/100.0
	K41frac  = K41frac/100.0

	Ag107frac = Ag107frac/100.0

	if Bfield==0.0:
		Bfield = 0.0001 #To avoid degeneracy problem at B = 0.

	# Rubidium energy levels
	if Elem=='Rb':
		Rb87frac=1.0-Rb85frac  # Rubidium-87 fraction
		print('Rb87frac is')
		print(Rb87frac)
		if Rb85frac!=0.0: #Save time if no rubidium-85 required
			Rb85atom = ac.Rb85
			#Hamiltonian(isotope,transition,gL,Bfield)
			Rb85_ES = ht.Hamiltonian('Rb85',Dline,1.0,Bfield)

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
		if Rb87frac!=0.0:
			Rb87atom = ac.Rb87
			#Hamiltonian(isotope,transition,gL,Bfield)
			Rb87_ES = ht.Hamiltonian('Rb87',Dline,1.0,Bfield)
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
			transitionConst = ac.RbD1Transition
		elif Dline=='D2':
			transitionConst = ac.RbD2Transition
		if (Rb85frac!=0.0) and (Rb87frac!=0.0):
			AllEnergyLevels = concatenate((lenergy87,lenergy85,
														renergy87,renergy85,
														zenergy87,zenergy85))
		elif (Rb85frac!=0.0) and (Rb87frac==0.0):
			AllEnergyLevels = concatenate((lenergy85,renergy85,zenergy85))
		elif (Rb85frac==0.0) and (Rb87frac!=0.0):
			AllEnergyLevels = concatenate((lenergy87,renergy87,zenergy87))

	# Caesium energy levels
	elif Elem=='Cs':
		CsAtom = ac.Cs
		Cs_ES = ht.Hamiltonian('Cs',Dline,1.0,Bfield)

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
			transitionConst = ac.CsD1Transition
		elif Dline =='D2':
			transitionConst = ac.CsD2Transition
		AllEnergyLevels = concatenate((lenergy,renergy,zenergy))

	# Sodium energy levels
	elif Elem=='Na':
		NaAtom = ac.Na
		Na_ES = ht.Hamiltonian('Na',Dline,1.0,Bfield)

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
			transitionConst = ac.NaD1Transition
		elif Dline=='D2':
			transitionConst = ac.NaD2Transition
		AllEnergyLevels = concatenate((lenergy,renergy,zenergy))

	#Potassium energy levels <<<<< NEED TO ADD Z-COMPONENT >>>>>
	elif Elem=='K':
		K39frac=1.0-K40frac-K41frac #Potassium-39 fraction
		if K39frac!=0.0:
			K39atom = ac.K39
			K39_ES = ht.Hamiltonian('K39',Dline,1.0,Bfield)
			
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
			K40atom = ac.K40
			K40_ES = ht.Hamiltonian('K40',Dline,1.0,Bfield)
			
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
			K41atom = ac.K41
			K41_ES = ht.Hamiltonian('K41',Dline,1.0,Bfield)
			
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
			transitionConst = ac.KD1Transition
		elif Dline=='D2':
			transitionConst = ac.KD2Transition
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
		Ag109frac = 1.0 - Ag107frac
		if Ag107frac != 0.0:
			Ag107atom = ac.Ag107
			Ag107_ES = ht.Hamiltonian('Ag107', Dline, 1.0, Bfield)

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
			Ag109atom = ac.Ag109
			Ag109_ES = ht.Hamiltonian('Ag109', Dline, 1.0, Bfield)

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
		if Dline == 'D2':
			transitionConst = ac.AgD2Transition
		else:
			raise ValueError("Only D2 data available for Ag")
		# Combine isotope contributions
		if Ag107frac != 0.0 and Ag109frac != 0.0:
			AllEnergyLevels = concatenate((lenergy107, lenergy109,
								  			renergy107,renergy109,
											zenergy107, zenergy109))
		elif (Ag107frac!=0.0) and (Ag109frac==0.0):
			AllEnergyLevels = concatenate((lenergy107, renergy107, zenergy107))
		elif (Ag107frac==0.0) and (Ag109frac!=0.0):
			AllEnergyLevels = concatenate((lenergy109, renergy109, zenergy109))

#Calculate Voigt

	if Constrain:
		DoppTemp = T #Set doppler temperature to the number density temperature
	T += 273.15
	DoppTemp += 273.15

	print(T,DoppTemp)

	# For thin cells: Don't add Doppler effect, by setting DopplerTemperature to near-zero
	# can then convolve with different velocity distribution later on
		
	d = (array(X)-shift) #Linear detuning
	xpts = len(d)
	maxdev = amax(abs(d))

	if Elem=='Rb':
		NDensity=nd.numDenRb(T) #Calculate number density
	elif Elem=='Cs':
		NDensity=nd.numDenCs(T)
	elif Elem=='K':
		NDensity=nd.numDenK(T)
	elif Elem=='Na':
		NDensity=nd.numDenNa(T)
	elif Elem== 'Ag':
		if 'AgNumden' in list(p_dict.keys()):
			NDensity = p_dict['AgNumden']
		else:
			NDensity= p_dict_defaults['AgNumden']
		print(NDensity)

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
		if Rb85frac!=0.0:
			lab85, ldisp85, rab85, rdisp85, zab85, zdisp85 = add_voigt(d,DoppTemp,
													   Rb85atom.mass,
													   wavenumber,gamma,
													   voigtwidth,
													   ltransno85,lenergy85,lstrength85,
													   rtransno85,renergy85,rstrength85,
													   ztransno85,zenergy85,zstrength85)
		if Rb87frac!=0.0:
			lab87, ldisp87, rab87, rdisp87, zab87, zdisp87 = add_voigt(d,DoppTemp,
													   Rb87atom.mass,
													   wavenumber,gamma,
													   voigtwidth,
													   ltransno87,lenergy87,lstrength87,
													   rtransno87,renergy87,rstrength87,
													   ztransno87,zenergy87,zstrength87)
		# Make the parts of the susceptibility

		if Isotope_Combination == 0:
			ChiRealLeft= prefactor*(Rb85frac*ldisp85[-1]+Rb87frac*ldisp87[-1])
			ChiRealRight= prefactor*(Rb85frac*rdisp85[-1]+Rb87frac*rdisp87[-1])
			ChiRealZ = prefactor*(Rb85frac*zdisp85[-1] + Rb87frac*zdisp87[-1])
			ChiImLeft = prefactor*(Rb85frac*lab85[-1]+Rb87frac*lab87[-1])
			ChiImRight = prefactor*(Rb85frac*rab85[-1]+Rb87frac*rab87[-1])
			ChiImZ = prefactor*(Rb85frac*zab85[-1] + Rb87frac*zab87[-1])
		elif Isotope_Combination == 1:
			ChiRealLeft= prefactor*(Rb85frac*ldisp85)
			ChiRealRight= prefactor*(Rb85frac*rdisp85)
			ChiRealZ = prefactor*(Rb85frac*zdisp85)
			ChiImLeft = prefactor*(Rb85frac*lab85)
			ChiImRight = prefactor*(Rb85frac*rab85)
			ChiImZ = prefactor*(Rb85frac*zab85)
		else:
			ChiRealLeft= prefactor*(Rb87frac*ldisp87)
			ChiRealRight= prefactor*(Rb87frac*rdisp87)
			ChiRealZ = prefactor*(Rb87frac*zdisp87)
			ChiImLeft = prefactor*(Rb87frac*lab87)
			ChiImRight = prefactor*(Rb87frac*rab87)
			ChiImZ = prefactor*(Rb87frac*zab87)

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

		# --- Add isotope-resolved Voigt profiles ---
		if Ag107frac != 0.0:
			(lab107, ldisp107,
			rab107, rdisp107,
			zab107, zdisp107) = add_voigt(
				d, DoppTemp, Ag107atom.mass, wavenumber,
				gamma, voigtwidth,
				ltransno107, lenergy107, lstrength107,
				rtransno107, renergy107, rstrength107,
				ztransno107, zenergy107, zstrength107
			)

		if Ag109frac != 0.0:
			(lab109, ldisp109,
			rab109, rdisp109,
			zab109, zdisp109) = add_voigt(
				d, DoppTemp, Ag109atom.mass, wavenumber,
				gamma, voigtwidth,
				ltransno109, lenergy109, lstrength109,
				rtransno109, renergy109, rstrength109,
				ztransno109, zenergy109, zstrength109
			)



		# --- Build total susceptibility ---
		if Isotope_Combination == 0:
			ChiRealLeft  = prefactor * (Ag107frac * ldisp107[-1] + Ag109frac * ldisp109[-1])
			ChiRealRight = prefactor * (Ag107frac * rdisp107[-1] + Ag109frac * rdisp109[-1])
			ChiRealZ     = prefactor * (Ag107frac * zdisp107[-1] + Ag109frac * zdisp109[-1])
			ChiImLeft  = prefactor * (Ag107frac * lab107[-1] + Ag109frac * lab109[-1])
			ChiImRight = prefactor * (Ag107frac * rab107[-1] + Ag109frac * rab109[-1])
			ChiImZ     = prefactor * (Ag107frac * zab107[-1] + Ag109frac * zab109[-1])
		elif Isotope_Combination == 1:
			ChiRealLeft  = prefactor * (Ag107frac * ldisp107)
			ChiRealRight = prefactor * (Ag107frac * rdisp107)
			ChiRealZ     = prefactor * (Ag107frac * zdisp107)
			ChiImLeft  = prefactor * (Ag107frac * lab107)
			ChiImRight = prefactor * (Ag107frac * rab107)
			ChiImZ     = prefactor * (Ag107frac * zab107)
		else:
			ChiRealLeft  = prefactor * (Ag109frac * ldisp109)
			ChiRealRight = prefactor * (Ag109frac * rdisp109)
			ChiRealZ     = prefactor * (Ag109frac * zdisp109)
			ChiImLeft  = prefactor * (Ag109frac * lab109)
			ChiImRight = prefactor * (Ag109frac * rab109)
			ChiImZ     = prefactor * (Ag109frac * zab109)
	
	totalChiPlus  = ChiRealLeft  + 1j * ChiImLeft
	totalChiMinus = ChiRealRight + 1j * ChiImRight
	totalChiZ     = ChiRealZ     + 1j * ChiImZ

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
	transition = ac.transitions[Elem+Dline]
	wavenumber = transition.wavevectorMagnitude * wavevector_dirn

	# get susceptibility (already calculated, input to this method)
	ChiPlus, ChiMinus, ChiZ = Chi

	# Rotate initial Electric field so that B field lies in x-z plane
	# (Effective polarisation rotation)
	E_xz = rot.rotate_around_z(E_in.T,Bphi)
		
	# Find eigen-vectors for propagation and create rotation matrix
	RM_ary, n1, n2 = solvd.solve_diel(ChiPlus,ChiMinus,ChiZ,Btheta,Bfield)

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
	fast_E_out = rot.rotate_around_z2(np.transpose(fast_E_out_xz,(0,2,1))[::,0],-Bphi) 

	#print 'E out [0]: ',E_out[0]
	#print 'E out shape: ',E_out.shape
	
	## return electric field vector - can then use Jones matrices to do everything else
	#return E_out.T[0], np.matrix(RM_ary.T[i])
	return fast_E_out.T[0], np.matrix(RM_ary.T[-1])

def get_spectra_old(X, E_in, p_dict, outputs=None):
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
	transition = ac.transitions[Elem+Dline]

	wavenumber = transition.wavevectorMagnitude
	
	# Calculate Susceptibility
	#ChiPlus, ChiMinus, ChiZ, voigt_components = calc_chi(X, p_dict)
	ChiPlus, ChiMinus, ChiZ = calc_chi(X, p_dict)

	index = 0

	Chi = [ChiPlus, ChiMinus, ChiZ]
	
	# Complex refractive index
	nPlus = sqrt(1.0+ChiPlus)[index] #Complex refractive index driving sigma plus transitions
	nMinus = sqrt(1.0+ChiMinus)[index] #Complex refractive index driving sigma minus transitions
	nZ = sqrt(1.0+ChiZ)[index] # Complex index driving pi transitions

	# convert (if necessary) detuning axis X to np array
	if type(X) in (int, float, int):
		X = np.array([X])
	else:
		X = np.array(X)

	# Calculate E_field
	E_out, R = get_Efield(X, E_in, [ChiPlus[index], ChiMinus[index], ChiZ[index]], p_dict)
	#print 'Output E field (Z): \n', E_out[2]
	

	## Apply Jones matrices
	
	# Transmission - total intensity - just E_out**2 / E_in**2
	E_in = np.array(E_in)
	if E_in.shape == (3,):
			E_in = np.array([np.ones(len(X))*E_in[0],np.ones(len(X))*E_in[1],np.ones(len(X))*E_in[2]])
	
	# normalised by input intensity
	I_in = (E_in * E_in.conjugate()).sum(axis=0)
	
	S0 = ((E_out * E_out.conjugate()).sum(axis=0) / I_in)
	
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
	Ex = np.array(rot.HorizPol_xy * E_out[:2])
	Ix =  (Ex * Ex.conjugate()).sum(axis=0) / I_in
	Ey =  np.array(rot.VertPol_xy * E_out[:2])
	Iy =  (Ey * Ey.conjugate()).sum(axis=0) / I_in
	
	S1 = Ix - Iy
	
	#S2#
	E_P45 =  np.array(rot.LPol_P45_xy * E_out[:2])
	E_M45 =  np.array(rot.LPol_M45_xy * E_out[:2])
	I_P45 = (E_P45 * E_P45.conjugate()).sum(axis=0) / I_in
	I_M45 = (E_M45 * E_M45.conjugate()).sum(axis=0) / I_in
	
	S2 = I_P45 - I_M45
	
	#S3#
	# change to circular basis
	E_out_lrz = cb.xyz_to_lrz(E_out)
	El =  np.array(rot.CPol_L_lr * E_out_lrz[:2])
	Er =  np.array(rot.CPol_R_lr * E_out_lrz[:2])
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
	'''
	# --- add per-Voigt data ---
	op['voigt_components'] = voigt_components  # from calc_chi()
	if 'voigt_transmissions' in locals():
		op['voigt_transmissions'] = voigt_transmissions
	'''
	if (outputs == None) or ('All' in outputs):
		# Default - return 'all' outputs (as used by GUI)
		return S0.real,S1.real,S2.real,S3.real,Ix.real,Iy.real,I_P45.real,I_M45.real,alphaPlus,alphaMinus,alphaZ
	else:
	# Return the variable names mentioned in the outputs list of strings
		# the strings in outputs must exactly match the local variable names here!
		return [op[output_str] for output_str in outputs]

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
	transition = ac.transitions[Elem+Dline]

	wavenumber = transition.wavevectorMagnitude
	
	# --- Calculate susceptibilities ---
	ChiPlus, ChiMinus, ChiZ = calc_chi(X, p_dict)
	n_trans = ChiPlus.shape[0] if ChiPlus.ndim > 1 else 1
	n_det = len(X)

	# --- Prepare output containers ---
	S0_all = np.zeros((n_trans, n_det), dtype=complex)
	S1_all = np.zeros_like(S0_all)
	S2_all = np.zeros_like(S0_all)
	S3_all = np.zeros_like(S0_all)
	Ix_all = np.zeros_like(S0_all)
	Iy_all = np.zeros_like(S0_all)
	E_out_all = np.zeros((n_trans, 3, n_det), dtype=complex)

	# --- Loop over transitions ---
	for index in range(n_trans):
		# Extract single-transition χ arrays
		chip = ChiPlus[index] if n_trans > 1 else ChiPlus
		chim = ChiMinus[index] if n_trans > 1 else ChiMinus
		chiz = ChiZ[index] if n_trans > 1 else ChiZ

		nPlus = np.sqrt(1.0 + chip)
		nMinus = np.sqrt(1.0 + chim)
		nZ = np.sqrt(1.0 + chiz)

		# --- Electric field ---
		E_out, R = get_Efield(X, E_in, [chip, chim, chiz], p_dict)
		E_out_all[index] = E_out

		# --- Input field normalization ---
		E_in_arr = np.array(E_in)
		if E_in_arr.shape == (3,):
			E_in_arr = np.array([
				np.ones(len(X)) * E_in_arr[0],
				np.ones(len(X)) * E_in_arr[1],
				np.ones(len(X)) * E_in_arr[2]
			])
		I_in = (E_in_arr * E_in_arr.conjugate()).sum(axis=0)

		# --- Transmission and Stokes ---
		S0 = ((E_out * E_out.conjugate()).sum(axis=0) / I_in)
		Ex = np.array(rot.HorizPol_xy * E_out[:2])
		Ey = np.array(rot.VertPol_xy * E_out[:2])
		Ix = (Ex * Ex.conjugate()).sum(axis=0) / I_in
		Iy = (Ey * Ey.conjugate()).sum(axis=0) / I_in
		S1 = Ix - Iy

		E_P45 = np.array(rot.LPol_P45_xy * E_out[:2])
		E_M45 = np.array(rot.LPol_M45_xy * E_out[:2])
		I_P45 = (E_P45 * E_P45.conjugate()).sum(axis=0) / I_in
		I_M45 = (E_M45 * E_M45.conjugate()).sum(axis=0) / I_in
		S2 = I_P45 - I_M45

		E_out_lrz = cb.xyz_to_lrz(E_out)
		El = np.array(rot.CPol_L_lr * E_out_lrz[:2])
		Er = np.array(rot.CPol_R_lr * E_out_lrz[:2])
		Il = (El * El.conjugate()).sum(axis=0) / I_in
		Ir = (Er * Er.conjugate()).sum(axis=0) / I_in
		S3 = Ir - Il

		# --- Store results ---
		S0_all[index] = S0.real
		S1_all[index] = S1.real
		S2_all[index] = S2.real
		S3_all[index] = S3.real
		Ix_all[index] = Ix.real
		Iy_all[index] = Iy.real

	# --- Return arrays (one per transition) ---
	op = {
		'S0': S0_all,
		'S1': S1_all,
		'S2': S2_all,
		'S3': S3_all,
		'Ix': Ix_all,
		'Iy': Iy_all,
		'E_out': E_out_all
	}

	if outputs is None or 'All' in outputs:
		return S0_all, S1_all, S2_all, S3_all, Ix_all, Iy_all
	else:
		return [op[o] for o in outputs]



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
	
	RotMat, n1, n2 = solvd.solve_diel(chiL,chiR,chiZ,88*np.pi/180)
	print((RotMat.shape))

def calculation_time_analysis():
	""" Test method for looking at timing performance """

	p_dict = {'Bfield':700,'rb85frac':1,'Btheta':88*np.pi/180,'Bphi':0*np.pi/180,'lcell':75e-3,'T':84,'Dline':'D2','Elem':'Cs'}
	chiL,chiR,chiZ = calc_chi([-3500],p_dict)
	
	for angle in [0, np.pi/32, np.pi/16, np.pi/8, np.pi/4, np.pi/2]:
		print(('Angle (degrees): ',angle*180/np.pi))
		RotMat, n1, n2 = solvd.solve_diel(chiL,chiR,chiZ,angle)
		
def test_equivalence():
	""" Test numeric vs analytic solutions """
	
	
	#analytic
	p_dict = {'Bfield':15000,'rb85frac':1,'Btheta':0*np.pi/180,'Bphi':0*np.pi/180,'lcell':1e-3,'T':84,'Dline':'D2','Elem':'Rb'}
	chiL1,chiR1,chiZ1 = calc_chi([-18400],p_dict)
	RotMat1, n11, n21 = solvd.solve_diel(chiL1,chiR1,chiZ1,0,150,force_numeric=False)
	
	#numeric
	chiL2, chiR2, chiZ2 = chiL1, chiR1, chiZ1
	#chiL2,chiR2,chiZ2 = sp.calc_chi([-18400],p_dict)
	RotMat2, n12, n22 = solvd.solve_diel(chiL2,chiR2,chiZ2,0,150,force_numeric=True)
	
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