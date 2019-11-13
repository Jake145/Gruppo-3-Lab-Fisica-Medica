import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats


y1=np.linspace(0,510,1000)    #genero una ascissa a caso per il fit
dose,amps=pylab.loadtxt('dosevsamps.txt',unpack=True) 
Ds=0.03*dose 
def f1(x,m,q):

    y=m*x+q
    
    return y 
     

     
popt, pcov= curve_fit(f1, amps, dose, (0.,0.),Ds,absolute_sigma=False)
DOF=len(amps)-3
chi2_1 = sum(((f1(amps,*popt)-dose)/Ds)**2)
dm,dq= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

m=popt[0]
q=popt[1]
print(*popt)

print(pcov)

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('il coefficiente angolare è %.3f, la intercetta è %.3f' % (m,q))
print('il chi2 è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto è=%.3f '% (chi2_1redux))
print('il pvalue è=%.3f'% (pvalue))

##plot 
pylab.figure('calibrazione')


pylab.errorbar( amps, dose, Ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('amps [ma]')
pylab.ylabel('dose [muGy]')


pylab.title('Dose vs Amps')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()