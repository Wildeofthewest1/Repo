import numpy as np
import matplotlib.pyplot as plt

from libs import main_functions as mf

Detuning=np.linspace(-10,10,1000)*1e3 #Detuning range between -10 and 10 GHz. Needs to be input in MHz
E_in=np.array([1,0,0]) #Horizontal Linear Light input. We define E_in = [Ex,Ey,Ez]
p_dict={'Elem':'Ag','Dline':'D2','T':25,'lcell':75e-3,'Bfield':0,'Btheta':0} #A 75 mm cell of natural abundance Rb at 20C. No bfield and hence no angle Btheta between the k-vector and the mag field. 
[S0,S1,S2,S3,E_out,Ix,Iy]=mf.get_spectra(Detuning,E_in,p_dict,outputs=['S0','S1','S2','S3','E_out','Ix','Iy'])

plt.plot(Detuning/1e3,S0)

plt.show()