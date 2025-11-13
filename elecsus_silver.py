import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from libs import main_functions as mf
from matplotlib import rcParams

import os
os.chdir(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
print("Now running in:", os.getcwd())


fontsz = 16
rcParams['font.family'] = 'serif' # e.g. 'sans-serif', 'monospace', etc.
rcParams['font.serif'] = ['Times New Roman'] # specify a particular font
rcParams['font.size'] = fontsz
rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm', 'stix', 'custom'

Detuning=np.linspace(-10,10,2000)*1e3 #Detuning range between -10 and 10 GHz. Needs to be input in MHz
E_in=np.array([1,0,0]) #Horizontal Linear Light input. We define E_in = [Ex,Ey,Ez]

choice = 1 #0 = Rb, 1 = Ag, 2 = K, 3 = Na, 4 = Cs
Temp = 25
AgNumberDensity = 1e16
Dline = 'D2'
lcell = 75e-3
Bfield = 0
Btheta = 0
ShowTransPlot = False

if choice == 0:
	element = 'Rb'
	if Dline == 'D2':
		Temp = 19
	else:
		Temp = 28
elif choice == 1:
	element = 'Ag'
elif choice == 2:
	element = 'K'
	if Dline == 'D2':
		Temp = 45
	else:
		Temp = 53
elif choice == 3:
	element = 'Na'
	if Dline == 'D2':
		Temp = 115
	else:
		Temp = 125
else:
	element = 'Cs'
	if Dline == 'D2':
		Temp = 3
	else:
		Temp = 11

p_dict={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta, 'AgNumden': AgNumberDensity, 'Isotope_Combination': 1}#, 'Ag107frac':100}
p_dict2={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta, 'AgNumden': AgNumberDensity, 'Isotope_Combination': 2}
p_dict3={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta, 'AgNumden': AgNumberDensity, 'Isotope_Combination': 0}

#A 75 mm cell of natural abundance Rb at 20C. No bfield and hence no angle Btheta between the k-vector and the mag field. 
[S0,S1,S2,S3,E_out,Ix,Iy]=mf.get_spectra(Detuning,E_in,p_dict,outputs=['S0','S1','S2','S3','E_out','Ix','Iy'])

[S0_1] = mf.get_spectra(Detuning,E_in,p_dict2,outputs=['S0'])

[S0_2] = mf.get_spectra(Detuning,E_in,p_dict3,outputs=['S0'])

plt.figure(figsize=(5, 3.5))

colours = ['deepskyblue', 'firebrick', 'purple', 'darkkhaki', 'orange', 'pink']

for i in range(len(S0)-1):

	if len(S0) >= 7:

		if i <= 2:
			color = colours[1]
		else:
			color = colours[0]
	else:
		
		if i <= 1:
			color = colours[1]
		else:
			color = colours[0]

	label = f'{i}'
	lw = 1.5
	alpha = 0.8

	plt.plot(Detuning / 1e3, S0[i].real, alpha=alpha, color=color, linewidth=lw, label=label, linestyle = "--")

if choice <= 2:
	for i in range(len(S0_1)-1):

		if len(S0_1) >= 7:
			if i <= 2:
				color = colours[3]
			else:
				color = colours[2]
		else:
			if i <= 1:
				color = colours[3]
			else:
				color = colours[2]

		label = f'{i}'
		lw = 1.5
		alpha = 0.8

		plt.plot(Detuning / 1e3, S0_1[i].real, alpha=alpha, color=color, linewidth=lw, label=label, linestyle = "--")

if choice == 2:#Extra for potassium as it has 3 isotopes
	p_dict4={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta, 'AgNumden': AgNumberDensity, 'Isotope_Combination': 3}
	[S0_3] = mf.get_spectra(Detuning,E_in,p_dict4,outputs=['S0'])	
	for i in range(len(S0_3)-1):

		if len(S0_3) >= 7:
			if i <= 2:
				color = colours[4]
			else:
				color = colours[5]
		else:
			if i <= 1:
				color = colours[4]
			else:
				color = colours[5]

		label = f'{i}'
		lw = 1.5
		alpha = 0.8

		plt.plot(Detuning / 1e3, S0_3[i].real, alpha=alpha, color=color, linewidth=lw, label=label, linestyle = "--")

plt.plot(Detuning / 1e3, S0_2[0].real, alpha = 0.8, color='grey', linewidth = 1, label='Total Transmission')
plt.fill_between(Detuning / 1e3, S0_2[0].real, 1, color='lightgrey', alpha=0.5)

plt.axhline(1, color='grey', lw=1)

plt.ylabel("Transmission")
plt.xlabel("Linear Detuning (GHz)")

## Labels (Adding labels to go with the transition level diagram)

adjust = 0.17

line = int(Dline[-1])

plt.text(x=-8, y=1.09, s=element+"-D$_{}$".format(line), fontsize=fontsz+2, ha = "left", va = "center") ##Ag-D2
plt.text(x=8, y=1.09, s="{}$\degree$C".format(Temp), fontsize=fontsz+2, ha = "right", va = "center") ##Temperature

def format_sci_tex(num):#format long numbers in standard form
	"""Return LaTeX-style scientific notation, e.g. 3×10¹⁵."""
	exp = int(np.floor(np.log10(num)))
	coeff = num / 10**exp
	return rf"${coeff:.1f} \times 10^{{{exp}}}$"

if choice == 1:
	plt.text(x=-8, y=0.91, s="$N_D$"+format_sci_tex(AgNumberDensity), fontsize=fontsz-2, ha = "left", va = "center") ##Temperature

if ShowTransPlot:
	plt.text(x=-8, y=0.12, s="$5^2$S$_{1/2}$", fontsize=fontsz, ha = "left", va = "center")#5s2S1/2
	plt.text(x=-8, y=0.44, s="$5^2$P$_{3/2}$", fontsize=fontsz, ha = "left", va = "center")#5p2P3/2
	plt.text(x=-3, y=0.28, s="D$_2$", fontsize=fontsz, ha = "left", va = "center")#D2
	plt.text(x=5.5+adjust, y=0.05, s="0", fontsize=fontsz, ha = "left", va = "center")#F=0
	plt.text(x=5.5+adjust, y=0.18, s="1", fontsize=fontsz, ha = "left", va = "center")#F=1
	plt.text(x=5.5+adjust, y=0.37, s="1", fontsize=fontsz, ha = "left", va = "center")#F'=1
	plt.text(x=5.5+adjust, y=0.49, s="2", fontsize=fontsz, ha = "left", va = "center")#F'=2
	plt.text(x=6.5+adjust, y=0.12, s="$F$", fontsize=fontsz, ha = "left", va = "center")#F
	plt.text(x=6.5+adjust, y=0.44, s="$F^'$", fontsize=fontsz, ha = "left", va = "center")#F'
	# --- Overlay the image ---
	img = mpimg.imread(r"C:\Users\Matt\Desktop\Lvl_4\Project\SilverD2Diagram109.png")
	plt.imshow(img, extent=[-5, 5.2+adjust, 0.05, 0.5], aspect='auto', alpha=0.7)

plt.ylim([0, 1.2])
plt.xlim([-8.5,8.5])

plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])
plt.xticks([-8, -4, 0, 4, 8])

plt.savefig(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Project\voight_full.pdf", dpi=600, bbox_inches='tight')

#plt.legend()

plt.show()

#name = "Ag_Spec_Matt/375_0.bmp"

#img1 = mpimg.imread(name)
#plt.imshow(img1)

#newname = name[13:-3]+"png"

#plt.savefig(newname, dpi=300, bbox_inches='tight')

#plt.show()