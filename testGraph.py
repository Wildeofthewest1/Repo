import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

# Font setup
fontsz = 16

# Change the default font family and size 
rcParams['font.family'] = 'serif' # e.g. 'sans-serif', 'monospace', etc.
rcParams['font.serif'] = ['Times New Roman'] # specify a particular font
rcParams['font.size'] = fontsz
 # Optional: change math text font
rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm', 'stix', 'custom'

#yData = np.load(r"C:\Users\Matt\Desktop\Lvl_4\Project\data.npy")#GHz
#xData = np.linspace(-10,10,1000)
Temp = 20

############### 

#328 nm D2 transition line
#5s2S1/2 > 5p2P3/2

################################################################


# Rubidium D2 Transmission Spectrum (showing individual Voigt dips)


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from sympy.physics.wigner import wigner_6j


# Physical constants from elecsus
c = 2.99792458e8
kB = 1.380649e-23
amu = 1.66053906660e-27

# Ag D2 line parameters
# Transition 5s2S1/2 > 5p2P3/2


#Kramida, A., Ralchenko, Yu., Reader, J., and NIST ASD Team (2024). NIST Atomic Spectra Database (ver. 5.12), [Online]. Available: https://physics.nist.gov/asd [2025, October 22]. National Institute of Standards and Technology, Gaithersburg, MD. DOI: https://doi.org/10.18434/T4W30F
lambda0 = 328.1625e-9 #NIST vacuum wavelength
Rblambda0 = 780.241e-9
#^ Central transition frequency in m
nu0 = c / lambda0

natural_linewidth = 1.4e8 #[rad/s] as given on NIST

Rb_nat = 2 * np.pi * 6.066e6

Gamma_nat = natural_linewidth   # natural linewidth [rad/s]

mass_Ag = 107.8682 * amu #Standard atomic weight for silver from NIST

abundance107 = 0.51839 #NIST Ag107 abundace

ratio1 = natural_linewidth / Rb_nat

ratio2 = Rblambda0 / lambda0

print( ratio1)

print( ratio2 * 1.1)

isotopes = {
    "Ag107": {
        "frac": abundance107,
        "I": 1/2,
        "A_g": 1712.512111e6,#Zeitschrift fi.ir Physik 200, 456--466 (1967)
        "A_e": 30.196e6,#From dodgy paper
        "B_e": 0, #B = 0 for I = 1/2 #from dodgy paper
        "shift": 476.6e6, #Uhlenberg et al 476.6 MHz
        "mass": 106.9050916 * amu #NIST
    },
    "Ag109": {
        "frac": 1-abundance107,
        "I": 1/2,
        "A_g": 1976.932075e6, #Zeitschrift fi.ir Physik 200, 456--466 (1967)
        "A_e": 75e6/2, #Uhlenberg et al splitting = 75Mhz, Divide by 2 as splitting = 2A for excited state
        "B_e": 0, #B = 0 for I = 1/2 #from dodgy paper
        "shift": 0.0,
        "mass": 108.9047553 * amu #NIST
    }
}


# Utility functions

def hyperfine_energies(I, J, A, B):
    """Return hyperfine energies (in Hz) for all F levels."""
    F_values = np.arange(abs(I - J), I + J + 1)
    E = []
    for F in F_values:
        K = F*(F+1) - I*(I+1) - J*(J+1)
        E_hf = 0.5*A*K
        if B != 0:
            E_hf += B * (3*K*(K+1) - 4*I*(I+1)*J*(J+1)) / (8*I*(2*I-1)*J*(2*J-1))
        E.append((F, E_hf))
    return E

def wigner6j(Fg, Fe, I, Jg, Je):
    """Return relative transition strength (squared 6j coefficient)."""
    return float((2*Fe+1)*(2*Fg+1) *
                 abs(wigner_6j(Je, Fe, I, Fg, Jg, 1))**2)

def voigt(x, sigma, gamma):
    """Voigt profile (normalised)."""
    z = (x + 1j*gamma) / (sigma*np.sqrt(2))
    return np.real(wofz(z)) / (sigma*np.sqrt(2*np.pi))

def gamma_nat():
    """Convert natural linewidth to MHz."""
    return Gamma_nat / (2*np.pi*1e6)

def sigma_doppler(T):
    """Doppler width (1-sigma) in MHz."""
    return nu0 * np.sqrt(2*kB*T*np.log(2)/mass_Ag/c**2) / 1e6

# Spectrum calculation

def transmission(T_C=Temp, detuning_GHz=10, show_components=True):
    T = T_C + 273.15
    det_MHz = np.linspace(-detuning_GHz, detuning_GHz, 2000) * 1e3
    alpha_total = np.zeros_like(det_MHz)
    components = []

    for name, iso in isotopes.items():
        I = iso["I"]
        A_g, A_e, B_e = iso["A_g"], iso["A_e"], iso["B_e"]
        frac = iso["frac"]
        shift = iso["shift"]
        Jg, Je = 0.5, 1.5 # J from 5s1/2 to 5p3/2

        ground = hyperfine_energies(I, Jg, A_g, 0)
        excited = hyperfine_energies(I, Je, A_e, B_e)

        for Fg, Eg in ground:
            for Fe, Ee in excited:
                # Selection rule |ΔF| ≤ 1
                if abs(Fe - Fg) > 1:
                    continue
                S = wigner6j(Fg, Fe, I, Jg, Je)
                if S == 0:
                    continue
                delta = (Ee - Eg + shift) / 1e6  # MHz
                profile = frac * S * voigt(det_MHz - delta,
                                           sigma_doppler(T),
                                           gamma_nat())
                alpha_total += profile

                if show_components:
                    label = f"{name}: Fg={Fg}→Fe={Fe}"
                    components.append((det_MHz, profile, label))

    scale = 0.69

    # Convert absorption profiles into transmission dips
    T_total = np.exp(-alpha_total / np.max(alpha_total) * scale)

    component_transmissions = []
    for det, prof, label in components:
        T_comp = np.exp(-prof / np.max(alpha_total) * scale)
        component_transmissions.append((det, T_comp, label))

    return det_MHz / 1e3, T_total, component_transmissions  # detuning in GHz


#################################################################


plt.figure(figsize=(5, 3.5))

x, T_total, comps = transmission(T_C=Temp)

plt.plot(x, T_total, color = "grey", lw=1)
#plt.plot(xData, yData, color = "grey", lw=1)

# --- Colour map for groups of 3 Voigts ---
colours = ['deepskyblue', 'firebrick', 'purple', 'darkkhaki']

# --- Colour coding by isotope and transition type ---
for det, T_comp, label in comps:
    if "Ag107" in label:
        # Use red/blue for Ag107
        if "Fg=0" in label and "Fe=1" in label:
            colour = colours[0]       # F=0→1
        elif "Fg=1" in label and "Fe=2" in label:
            colour = colours[1]      # F=1→2
        else:
            colour = colours[1]      # fallback
    elif "Ag109" in label:
        # Use purple/khaki for Ag109
        if "Fg=0" in label and "Fe=1" in label:
            colour = colours[2]    # F=0→1
        elif "Fg=1" in label and "Fe=2" in label:
            colour = colours[3] # F=1→2
        else:
            colour = colours[3] # fallback
    else:
        colour = "grey"  # default, in case of missing label

    plt.plot(det/1e3, T_comp, '--', alpha=0.8, lw=1.5, color=colour, label=label)

plt.fill_between(x, T_total, 1, color='lightgrey', alpha=0.5)
plt.axhline(1, color='grey', lw=1)

plt.ylabel("Transmission")
plt.xlabel("Linear Detuning (GHz)")

## Labels (Adding labels to go with the transition level diagram)

adjust = 0.17

plt.text(x=-8, y=1.09, s="Ag-D$_2$", fontsize=fontsz+2, ha = "left", va = "center") ##Metal
plt.text(x=8, y=1.09, s="{}$\degree$C".format(Temp), fontsize=fontsz+2, ha = "right", va = "center") ##Temperature

plt.text(x=-8, y=0.12, s="$5^2$S$_{1/2}$", fontsize=fontsz, ha = "left", va = "center")#5s2S1/2
plt.text(x=-8, y=0.44, s="$5^2$P$_{3/2}$", fontsize=fontsz, ha = "left", va = "center")#5p2P3/2

plt.text(x=-3, y=0.27, s="D$_2$", fontsize=fontsz, ha = "left", va = "center")#5p2P3/2

plt.text(x=5.5+adjust, y=0.05, s="0", fontsize=fontsz, ha = "left", va = "center")#F=0
plt.text(x=5.5+adjust, y=0.18, s="1", fontsize=fontsz, ha = "left", va = "center")#F=1

plt.text(x=5.5+adjust, y=0.37, s="1", fontsize=fontsz, ha = "left", va = "center")#F'=1
plt.text(x=5.5+adjust, y=0.49, s="2", fontsize=fontsz, ha = "left", va = "center")#F'=2

plt.text(x=6.5+adjust, y=0.12, s="$F$", fontsize=fontsz, ha = "left", va = "center")#F
plt.text(x=6.5+adjust, y=0.44, s="$F^'$", fontsize=fontsz, ha = "left", va = "center")#F'

##

plt.ylim([0, 1.2])
plt.xlim([-8.5,8.5])

plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])
plt.xticks([-8, -4, 0, 4, 8])

# --- Overlay the image ---
img = mpimg.imread(r"C:\Users\Matt\Desktop\Lvl_4\Project\SilverD2Diagram.png")

# Add image to the plot using figimage or imshow
# Place image in axis coordinates (0-1)
plt.imshow(img, extent=[-5, 5+adjust, 0.05, 0.5], aspect='auto', alpha=0.7)

#plt.savefig(r"C:\Users\Matt\Desktop\Lvl_4\Project\voigt_combined.pdf", bbox_inches='tight')

plt.show()