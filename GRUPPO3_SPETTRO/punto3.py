
##Non ho ancora fatto nulla 


import numpy as np
import scipy
import matplotlib.pyplot as plt

x=np.linspace(0,2048,2048)
y=np.loadtxt('inverserootlawCs676mmtext.txt')
fondo=np.loadtxt('fondotext.txt')


plt.figure('Cesio a una certa distanza')
plt.plot(x, y, color='blue',marker = 'o')
plt.xlabel('chn')

z=(y/max(y))-(fondo/max(fondo))

z[z<0]=0
data=z*max(y)
plt.figure('Cesio senza fondo')
plt.plot(x, data, color='green',marker = 'o')
plt.xlabel('chn')

plt.ylabel('count')
plt.grid(True)
plt.show()
data[0:840]=0
data[1000:2048]=0
photopeakcount=np.sum(data[data>0])
print(data[data>0])
print(photopeakcount)


import pylab
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy import stats

sumphotopeak=np.array([89642.86,37152.39,19532.97,12149.04,13953.44,7728.73,5676.84,4271.79,3382.63,2859.33])
distance=np.array([266,316,366,400,416,476,526,576,626,676])