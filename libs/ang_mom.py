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