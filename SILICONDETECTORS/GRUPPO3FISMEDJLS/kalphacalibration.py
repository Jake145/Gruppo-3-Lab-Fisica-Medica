import numpy as np
import pylab
from scipy.optimize import curve_fit
from scipy import stats
channel=np.array([82.95,201.20,56.53,200.62,201.88,144.18,199.90,55.93,198.91])
energy=np.array([17.17,41.66,11.70,41.52,41.78,29.84,41.38,11.58,41.17])
ds=0.05*energy
chn=channel-11
y1=np.linspace(37,200,1000)
def f1(x,m,q):
    return m*x+q


popt, pcov= curve_fit(f1, chn, energy, (0.,0.),ds,absolute_sigma=False)
DOF=len(chn)-3
chi2_1 = sum(((f1(chn,*popt)-energy)/ds)**2)
dm,dq= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

m=popt[0]
q=popt[1]
print(*popt)

print(pcov)

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('il coefficiente angolare per la calibrazione  è %.3f pm %.3f, la intercetta è %.3f pm %.3f' % (m,dm,q,dq))
print('il chi2 per la calibrazione  è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto per la calibrazione  è=%.3f '% (chi2_1redux))
print('il pvalue per la calibrazione  è=%.3f'% (pvalue))
print('la funzione per la calibrazione   è ENERGY = %.2f * CHANNEL + %.2f '%(m,q))
##plot
pylab.figure('calibrazioneMVvsCHN')


pylab.errorbar( chn, energy,ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('Chn')
pylab.ylabel('energy[KeV]')


pylab.title('calibration k peaks')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()
#pylab.xlim(0,420)
#pylab.ylim(0,6)
pylab.savefig('calibration_test_signals_new.png')
pylab.show()