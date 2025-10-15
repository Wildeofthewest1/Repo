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
################################################################


################################################################
plt.figure(figsize=(5, 3.5))
plt.plot(xData, yData, color = "grey", lw=1.5)
plt.fill_between(xData, yData, 1, color='lightgrey', alpha=0.5)
plt.axhline(1, color='grey', lw=1.5)

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
