from numpy import pi,append, transpose, identity

from scipy.constants import physical_constants, epsilon_0, hbar

from scipy.linalg import eigh

from libs import atomic_constants as ac
from libs import LSI_funcs as lsi

S=0.5 #Electron spin
gs = -physical_constants['electron g factor'][0]
muB=physical_constants['Bohr magneton'][0] 
kB=physical_constants['Boltzmann constant'][0] 
amu=physical_constants['atomic mass constant'][0] #An atomic mass unit in kg
e0=epsilon_0 #Permittivity of free space
a0=physical_constants['Bohr radius'][0]
kB = physical_constants['Boltzmann constant'][0]


class Hamiltonian(object):
    """Functions to create the atomic hamiltonian."""

    def __init__(self, Isotope, Trans, gL, Bfield):
        """Ground and excited state Hamiltonian for an isotope"""
        if Isotope=='Rb87':
            atom = ac.Rb87
        elif Isotope=='Rb85':
            atom = ac.Rb85


        elif Isotope=='Ag107':
            atom = ac.Ag107
        elif Isotope=='Ag109':
            atom = ac.Ag109

            
        elif Isotope=='Cs':
            atom = ac.Cs
        elif Isotope=='K39':
            atom = ac.K39
        elif Isotope=='K40':
            atom = ac.K40
        elif Isotope=='K41':
            atom = ac.K41
        elif Isotope=='Na':
            atom = ac.Na
        elif Isotope=='IdealAtom':
            atom = ac.IdealAtom
            transition = ac.IdealD1Transition
            atom_transition = ac.Ideal_D1

        self.atom = atom
		
        if (Trans=='D1') and (Isotope=='Rb85'):
            transition = ac.RbD1Transition
            atom_transition = ac.Rb85_D1
        elif (Trans=='D2') and (Isotope=='Rb85'):
            transition = ac.RbD2Transition
            atom_transition = ac.Rb85_D2
        elif (Trans=='D1') and (Isotope=='Rb87'):
            transition = ac.RbD1Transition
            atom_transition = ac.Rb87_D1
        elif (Trans=='D2') and (Isotope=='Rb87'):
            transition = ac.RbD2Transition
            atom_transition = ac.Rb87_D2

        elif (Trans=='D2') and (Isotope=='Ag107'):
            transition = ac.AgD2Transition
            atom_transition = ac.Ag107_D2
        elif (Trans=='D2') and (Isotope=='Ag109'):
            transition = ac.AgD2Transition
            atom_transition = ac.Ag109_D2

        elif (Trans=='D1') and (Isotope=='Cs'):
            transition = ac.CsD1Transition
            atom_transition = ac.Cs_D1
        elif (Trans=='D2') and (Isotope=='Cs'):
            transition = ac.CsD2Transition
            atom_transition = ac.Cs_D2
        elif (Trans=='D1') and (Isotope=='Na'):
            transition = ac.NaD1Transition
            atom_transition = ac.Na_D1
        elif (Trans=='D2') and (Isotope=='Na'):
            transition = ac.NaD2Transition
            atom_transition = ac.Na_D2
        elif (Trans=='D1') and (Isotope=='K39'):
            transition = ac.KD1Transition
            atom_transition = ac.K39_D1
        elif (Trans=='D2') and (Isotope=='K39'):
            transition = ac.KD2Transition
            atom_transition = ac.K39_D2
        elif (Trans=='D1') and (Isotope=='K40'):
            transition = ac.KD1Transition
            atom_transition = ac.K40_D1
        elif (Trans=='D2') and (Isotope=='K40'):
            transition = ac.KD2Transition
            atom_transition = ac.K40_D2
        elif (Trans=='D1') and (Isotope=='K41'):
            transition = ac.KD1Transition
            atom_transition = ac.K41_D1
        elif (Trans=='D2') and (Isotope=='K41'):
            transition = ac.KD2Transition
            atom_transition = ac.K41_D2
			
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
        S_StateHamiltonian = As*lsi.Hhfs(0.0,S,I)+IsotopeShift*identity(ds)
        Ez = muB*Bfield*1.e-4/(hbar*2.0*pi*1.0e6)
        S_StateHamiltonian += Ez*(gs*lsi.sz(0.0,S,I)+gI*lsi.Iz(0.0,S,I)) # Add Zeeman
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
            P_StateHamiltonian=FS*lsi.Hfs(1.0,S,I)+FS*identity(dp)+Ap*lsi.Hhfs(1.0,S,I)
        if Bp!=0.0:
            P_StateHamiltonian=FS*lsi.Hfs(1.0,S,I)-(FS/2.0)*identity(dp)+Ap*lsi.Hhfs(1.0,S,I)
            P_StateHamiltonian+=Bp*lsi.Bbhfs(1.0,S,I) # add p state quadrupole
        E=muB*(Bfield*1.0e-4)/(hbar*2.0*pi*1.0e6)
        # Add magnetic interaction
        P_StateHamiltonian+=E*(gL*lsi.lz(1.0,S,I)+gs*lsi.sz(1.0,S,I)+gI*lsi.Iz(1.0,S,I))
        ep=eigh(P_StateHamiltonian)
        EigenValues=ep[0].real
        EigenVectors=ep[1]
        stateManifold=append([EigenValues],EigenVectors,axis=0)
        sortedManifold=sorted(transpose(stateManifold),key=(lambda i:i[0]))
        return sortedManifold, EigenValues