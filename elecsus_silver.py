import numpy as np
import matplotlib.pyplot as plt

from libs import main_functions as mf

Detuning=np.linspace(-10,10,2000)*1e3 #Detuning range between -10 and 10 GHz. Needs to be input in MHz
E_in=np.array([1,0,0]) #Horizontal Linear Light input. We define E_in = [Ex,Ey,Ez]
p_dict={'Elem':'Ag','Dline':'D2','T':20,'lcell':75e-3,'Bfield':0,'Btheta':0, 'Ag107frac': 54, 'AgNumden': 3e15} #A 75 mm cell of natural abundance Rb at 20C. No bfield and hence no angle Btheta between the k-vector and the mag field. 
[S0,S1,S2,S3,E_out,Ix,Iy]=mf.get_spectra(Detuning,E_in,p_dict,outputs=['S0','S1','S2','S3','E_out','Ix','Iy'])

plt.figure(figsize=(5, 3.5))

plt.plot(Detuning/1e3, S0.real, 'k-', linewidth=2, label='Total Transmission')

plt.axhline(1, color='grey', lw=1)

plt.ylabel("Transmission")
plt.xlabel("Linear Detuning (GHz)")

plt.ylim([0, 1.2])
plt.xlim([-8.5,8.5])

plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])
plt.xticks([-8, -4, 0, 4, 8])

plt.legend()
plt.show()