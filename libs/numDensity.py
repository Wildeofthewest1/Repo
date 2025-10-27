from numpy import log10
from scipy.constants import physical_constants, epsilon_0

S=0.5 #Electron spin
gs = -physical_constants['electron g factor'][0]
muB=physical_constants['Bohr magneton'][0] 
kB=physical_constants['Boltzmann constant'][0] 
amu=physical_constants['atomic mass constant'][0] #An atomic mass unit in kg
e0=epsilon_0 #Permittivity of free space
a0=physical_constants['Bohr radius'][0]
kB = physical_constants['Boltzmann constant'][0]


def CalcNumberDensity(T,atom):
	
	#Temperature in Kelvin
	#Number density returned in inverse cubic metres
	
	
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
    #Calculates the rubidium number density
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