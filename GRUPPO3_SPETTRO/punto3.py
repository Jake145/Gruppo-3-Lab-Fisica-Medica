
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

##codice vero
y1=np.linspace(0,3000,1000)    #genero una ascissa a caso per il fit
energy,counts=pylab.loadtxt('punto1data.txt',unpack=True) 
Ds=10 #errore a caso, usa la fwhm dal fit gaussiano
def f1(x,mu,ch_0):

    y=ch_0*np.exp(-mu*x)
    
    return y 
     

     
popt, pcov= curve_fit(f1, channels, width, (0.,0.),Ds,absolute_sigma=False)
DOF=len(width)-2
chi2_1 = sum(((f1(width,*popt)-channels)/Ds)**2)
dmu,dch_0= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

mu=popt[0]
ch_0=popt[1]
print(*popt)

print(pcov)

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('il coefficiente di assorbimento è %.3f, la costante moltiplicativa è %.3f' % (mu,ch_0))
print('il chi2 è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto è=%.3f '% (chi2_1redux))
print('il pvalue è=%.3f'% (pvalue))
