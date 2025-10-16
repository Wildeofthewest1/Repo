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

yData = np.load(r"C:\Users\Matt\Desktop\Lvl_4\Project\data.npy")#GHz
xData = np.linspace(-10,10,1000)
Temp = 19

############### 

#328 nm D2 transition line
#5s2S1/2 > 5p2P3/2

#Detuning for transitions: 
#
#−2735.05, −2578.11, −2311.26 MHz
#−1371.29, −1307.87, −1186.91 MHz
#+1635.454, +1664.714, +1728.134 MHz
#+4027.403, +4099.625, +4256.57 MHz
#

abundance85 = 0.7217

################################################################

# ============================================================
# Rubidium D2 Transmission Spectrum (showing individual Voigt dips)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from sympy.physics.wigner import wigner_6j

# --------------------------
# Physical constants
# --------------------------
c = 2.99792458e8
kB = 1.380649e-23
amu = 1.66053906660e-27

# --------------------------
# Rb D2 line parameters
# --------------------------
lambda0 = 780.241e-9
nu0 = c / lambda0
Gamma_nat = 2 * np.pi * 6.066e6   # natural linewidth [rad/s]
mass_Rb = 85 * amu

isotopes = {
    "Rb85": {
        "frac": abundance85,
        "I": 5/2,
        "A_g": 1011.910e6,
        "A_e": 25.002e6,
        "B_e": 25.790e6,
        "shift": -63.4e6
    },
    "Rb87": {
        "frac": 1-abundance85,
        "I": 3/2,
        "A_g": 3417.341e6,
        "A_e": 84.7185e6,
        "B_e": 12.4965e6,
        "shift": 0.0
    }
}

# --------------------------
# Utility functions
# --------------------------
def hyperfine_energies(I, J, A, B):
    F_values = np.arange(abs(I - J), I + J + 1)
    E = []
    for F in F_values:
        K = F*(F+1) - I*(I+1) - J*(J+1)
        E_hf = 0.5*A*K
        if B != 0:
            E_hf += B * (3*K*(K+1) - 4*I*(I+1)*J*(J+1)) / (8*I*(2*I-1)*J*(2*J-1))
        E.append((F, E_hf))
    return E

def wigner6j_strength(Fg, Fe, I, Jg, Je):
    return float((2*Fe+1)*(2*Fg+1) * abs(wigner_6j(Je, Fe, I, Fg, Jg, 1))**2)

def voigt(x, sigma, gamma):
    z = (x + 1j*gamma) / (sigma*np.sqrt(2))
    return np.real(wofz(z)) / (sigma*np.sqrt(2*np.pi))

def gamma_nat():
    return Gamma_nat / (2*np.pi*1e6)  # MHz

def sigma_doppler(T):
    return nu0 * np.sqrt(2*kB*T*np.log(2)/mass_Rb/c**2) / 1e6  # MHz

# --------------------------
# Spectrum calculation
# --------------------------
def rb_d2_transmission(T_C=Temp, detuning_GHz=10, show_components=True):
    T = T_C + 273.15
    det_MHz = np.linspace(-detuning_GHz, detuning_GHz, 2000) * 1e3
    alpha_total = np.zeros_like(det_MHz)
    components = []

    for name, iso in isotopes.items():
        I = iso["I"]
        A_g, A_e, B_e = iso["A_g"], iso["A_e"], iso["B_e"]
        frac = iso["frac"]
        shift = iso["shift"]
        Jg, Je = 0.5, 1.5

        ground = hyperfine_energies(I, Jg, A_g, 0)
        excited = hyperfine_energies(I, Je, A_e, B_e)

        for Fg, Eg in ground:
            for Fe, Ee in excited:
                # Selection rule |ΔF| ≤ 1
                if abs(Fe - Fg) > 1:
                    continue
                S = wigner6j_strength(Fg, Fe, I, Jg, Je)
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

    # Convert absorption profiles into transmission dips
    scale = 0.69 # optical depth scaling (adjust for deeper or shallower dips)
    T_total = np.exp(-scale * alpha_total / np.max(alpha_total))

    component_transmissions = []
    for det, prof, label in components:
        T_comp = np.exp(-scale * prof / np.max(alpha_total))
        component_transmissions.append((det, T_comp, label))

    return det_MHz / 1e3, T_total, component_transmissions  # detuning in GHz

# --------------------------
# Plot: transmission dips
# --------------------------

#################################################################


plt.figure(figsize=(5, 3.5))

x, T_total, comps = rb_d2_transmission(T_C=Temp)

plt.plot(x, T_total, color = "grey", lw=1)
#plt.plot(xData, yData, color = "grey", lw=1)

# --- Colour map for groups of 3 Voigts ---
colours = ['deepskyblue', 'firebrick', 'purple', 'darkkhaki']

# Plot each component with grouped colours
for i, (det, T_comp, label) in enumerate(comps):
    colour = colours[i // 3 % len(colours)]  # groups of 3
    plt.plot(det/1e3, T_comp, '--', alpha=0.8, lw=1.5, color=colour, label=label)


plt.fill_between(x, T_total, 1, color='lightgrey', alpha=0.5)
plt.axhline(1, color='grey', lw=1)

plt.ylabel("Transmission")
plt.xlabel("Linear Detuning (GHz)")

## Labels

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
plt.text(x=6.5+adjust, y=0.44, s="$F^{\prime}$", fontsize=fontsz, ha = "left", va = "center")#F'

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
