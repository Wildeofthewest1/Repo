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
    As = 1712.512111  # MHz (ground-state hyperfine A constant, Uhlenberg et al 2000)
    gI = 0
    mass = 106.90509 * amu
    FS = 0.0

class Ag109:
    """Constants for silver-109 atom"""
    I = 0.5
    As = 1976.932075
    gI = 0
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
    wavelength=328.1625e-9#m
    wavevectorMagnitude=2.0*pi/wavelength
    NatGamma=1.4e2
    dipoleStrength=3.0*sqrt(e0*hbar*(2.0*NatGamma*(10.0**6))*(wavelength**3)/(8.0*pi))
    v0 = c/wavelength

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
    Ap = 31.7
    Bp = 0
    IsotopeShift = 229.24 #MHz

class Ag109_D2:
    """Constants relating to rubidium-87 and the D2 transition"""
    #Hyperfine constants in units of MHz
    Ap = 36.7
    Bp = 0
    IsotopeShift = -246.76 #MHz

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