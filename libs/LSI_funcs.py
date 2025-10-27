from numpy import dot, identity, kron
from scipy.constants import physical_constants, epsilon_0
from libs import ang_mom as am

S=0.5 #Electron spin
gs = -physical_constants['electron g factor'][0]
muB=physical_constants['Bohr magneton'][0] 
kB=physical_constants['Boltzmann constant'][0] 
amu=physical_constants['atomic mass constant'][0] #An atomic mass unit in kg
e0=epsilon_0 #Permittivity of free space
a0=physical_constants['Bohr radius'][0]
kB = physical_constants['Boltzmann constant'][0]

def sz(L,S,I):
    Sz=am.jz(S)
    gL=int(2*L+1)
    Li=identity(gL)
    gI=int(2*I+1)
    Ii=identity(gI)
    sz=kron(kron(Li,Sz),Ii)
    return sz

def lz(L,S,I):
    gS=int(2*S+1)
    Si=identity(gS)
    Lz=am.jz(L)
    gI=int(2*I+1)
    Ii=identity(gI)
    lz=kron(kron(Lz,Si),Ii)
    return lz

def Iz(L,S,I):
    gS=int(2*S+1)
    gL=int(2*L+1)
    Si=identity(gS)
    Li=identity(gL)
    Iz_num=am.jz(I)
    Iz=kron(kron(Li,Si),Iz_num)
    return Iz

def fz(L,S,I):
    gS=int(2*S+1)
    Sz=am.jz(S)
    Si=identity(gS)
    gL=int(2*L+1)
    Lz=am.jz(L)
    Li=identity(gL)
    gJ=gL*gS
    Jz=kron(Lz,Si)+kron(Li,Sz)
    Ji=identity(gJ)
    gI=int(2*I+1)
    Iz=am.jz(I)
    Ii=identity(gI)
    Fz=kron(Jz,Ii)+kron(Ji,Iz)
    return Fz

def Hfs(L,S,I):
    """Provides the L dot S matrix (fine structure)"""
    gS=int(2*S+1) #number of mS values
    Sx=am.jx(S)
    Sy=am.jy(S)
    Sz=am.jz(S)
    Si=identity(gS)

    gL=int(2*L+1)
    Lx=am.jx(L)
    Ly=am.jy(L)
    Lz=am.jz(L)
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
    Sx=am.jx(S)
    Sy=am.jy(S)
    Sz=am.jz(S)
    Si=identity(gS)

    gL=int(2*L+1)
    Lx=am.jx(L)
    Ly=am.jy(L)
    Lz=am.jz(L)
    Li=identity(gL)

    gJ=gL*gS
    Jx=kron(Lx,Si)+kron(Li,Sx)
    Jy=kron(Ly,Si)+kron(Li,Sy)
    Jz=kron(Lz,Si)+kron(Li,Sz)
    Ji=identity(gJ)
    J2=dot(Jx,Jx)+dot(Jy,Jy)+dot(Jz,Jz)

    gI=int(2*I+1)
    gF=gJ*gI
    Ix=am.jx(I)
    Iy=am.jy(I)
    Iz=am.jz(I)
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
    Sx=am.jx(S)
    Sy=am.jy(S)
    Sz=am.jz(S)
    Si=identity(gS)

    gL=int(2*L+1)
    Lx=am.jx(L)
    Ly=am.jy(L)
    Lz=am.jz(L)
    Li=identity(gL)

    gJ=gL*gS
    Jx=kron(Lx,Si)+kron(Li,Sx)
    Jy=kron(Ly,Si)+kron(Li,Sy)
    Jz=kron(Lz,Si)+kron(Li,Sz)

    gI=int(2*I+1)
    gF=gJ*gI
    Ix=am.jx(I)
    Iy=am.jy(I)
    Iz=am.jz(I)
    
    Fi=identity(gF)

    IdotJ=kron(Jx,Ix)+kron(Jy,Iy)+kron(Jz,Iz)
    IdotJ2=dot(IdotJ,IdotJ)

    if I != 0:
        Bbhfs=1./(6*I*(2*I-1))*(3*IdotJ2+3./2*IdotJ-I*(I+1)*15./4*Fi)
    else:
        Bbhfs = 0
    return Bbhfs
