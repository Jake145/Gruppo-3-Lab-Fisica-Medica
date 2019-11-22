import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats

signal=np.array([152.0,163.7,173.9,180.9,167.4,141.0,123.1,195.4])
counts=np.array([4916,3606,1470,629,3020,5098,5122,13])
dc=np.sqrt((np.sqrt(counts)**2+(0.1*counts)**2))
y1=np.linspace(0,250,1000)    

def f1(x,mu,sigma,A1,A2):

    y=A2+(A1-A2)/(1+np.exp((x-mu)/sigma))
    
    return y 
     

     
popt, pcov= curve_fit(f1, signal, counts, (170,100,5000,0.),dc,absolute_sigma=False)
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


pylab.title('Discriminatore con C=18.6 pf')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()
