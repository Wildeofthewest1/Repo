import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Change the default font family and size
rcParams['font.family'] = 'serif'        # e.g. 'sans-serif', 'monospace', etc.
rcParams['font.serif'] = ['Times New Roman']  # specify a particular font
rcParams['font.size'] = 14

# Optional: change math text font
rcParams['mathtext.fontset'] = 'dejavuserif'  # or 'cm', 'stix', 'custom'

points = 17

yData = np.ones (points)
xData = np.linspace (-8,8,points)

print (xData)

Temp = 42

plt.plot(xData,yData)

plt.ylabel("Transmission")
plt.xlabel("Linear Detuning (GHz)")

plt.ylim([0,1.25])
#plt.xlim([-8.5,8.5])

plt.yticks([0.00,0.25,0.50,0.75,1.00])
plt.xticks([-8,-4,0,4,8])

print("done")

plt.show()