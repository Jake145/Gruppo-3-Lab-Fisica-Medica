import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats

signal=np.array([152.4,160.7,166.5,170.9,176.0,177.1,178.4,174.6,156.0,142.8,138.9,125.1,182.0,191.8])
counts=np.array([4846,3271,2159,1546,535,455,377,1023,4832,5120,5121,5122,207,5])
dc=np.sqrt((np.sqrt(counts)**2+(0.1*counts)**2))
y1=np.linspace(0,250,1000)    



def f1(x,mu,sigma,A1,A2):

    y=A2+(A1-A2)/(1+np.exp((x-mu)/sigma))
    
    return y 
     

     
popt, pcov= curve_fit(f1, signal, counts,(170,100,5000,0.),dc,absolute_sigma=False)
DOF=len(signal)-3
chi2_1 = sum(((f1(signal,*popt)-counts)/dc)**2)
dmu,dsigma,dA1,dA2= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

mu=popt[0]
sigma=popt[1]
A1=popt[2]
A2=popt[3]




pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('la media è %.3f pm %.3f, la sigma è %.3f pm %.3f , Ampiezza superiore è %.3f pm %.3f , Ampiezza2 è %.3f pm %.3f'  % (mu,dmu,sigma,dsigma,A1,dA1,A2,dA2))
print('il chi2 è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto è=%.3f '% (chi2_1redux))
print('il pvalue è=%.3f'% (pvalue))

pylab.figure('Discriminatore') 


pylab.errorbar( signal  , counts, dc , fmt= '.', ecolor= 'magenta')

pylab.xlabel('Amplitude [mV]')
pylab.ylabel('Counts')


pylab.title('Discriminatore')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()
