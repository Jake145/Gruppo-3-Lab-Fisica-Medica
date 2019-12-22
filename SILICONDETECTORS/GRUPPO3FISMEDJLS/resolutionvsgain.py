import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats

##RISOLUZIONE ENERGETICA VS GAIN
tension=np.array([1.04,2.16,3.68,4.16]) #in mv
gain=np.array([43,38,33.5,32.5]) #in db
chn=np.array([85.58,176.57,305.20,342.69])
fwhm=np.array([1.69,1.69,1.66,1.73])

capacity=1e-12 #in farad
Ee=3.6*1e-3 #in kev
e=1.6*1e-19 #in coulomb

fwhm=fwhm/chn
def energymvmvfrommv(x):
    return capacity*x*0.001*Ee/e
energymvmv=energymvmvfrommv(tension)
energres=fwhm/energymvmv
y1=np.linspace(0,50,1000)
p1=np.polyfit(gain,energres,1)
yfit1=p1[0]*gain+p1[1]
yres1=energres-yfit1
SSresid1=sum(pow(yres1,2))
SStotal1=len(energres)*np.var(energres)
rsq1=1-SSresid1/SStotal1
print('Per il fit lineare per la risoluzione vs gain il coefficiente del primo grado è %.3f, del termine costante è %.3f, R value è %.3f' % (p1[0],p1[1],rsq1))

p2=np.polyfit(gain,energres,2)
yfit2=p2[0]*gain**2+p2[1]*gain +p2[2]
yres2=energres-yfit2
SSresid2=sum(pow(yres2,2))
SStotal2=len(energres)*np.var(energres)
rsq2=1-SSresid2/SStotal2
print('Per il fit quadratico per la risoluzione vs gain il coefficiente del secondo grado è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p2[0],p2[1],p2[2],rsq2))

p3=np.polyfit(gain,energres,3)
yfit3=p3[0]*gain**3+p3[1]*gain**2 + p3[2]*gain +p3[3]
yres3=energres-yfit3
SSresid3=sum(pow(yres3,2))
SStotal3=len(energres)*np.var(energres)
rsq3=1-SSresid3/SStotal3
print('Per il fit cubico per la risoluzione vs gain il coefficiente del terzo grado è %.3f, del second è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p3[0],p3[1],p3[2],p3[3],rsq3))

p4=np.polyfit(gain,energres,4)
yfit4=p4[0]*gain**4+p4[1]*gain**3 + p4[2]*gain**2 +p4[3]*gain+p4[4]
yres4=energres-yfit4
SSresid4=sum(pow(yres4,2))
SStotal4=len(energres)*np.var(energres)
rsq4=1-SSresid4/SStotal4
print('Per il fit quartico per la risoluzione vs gain il coefficiente del quarto grado è %.6f, del terzo è %.3f,del secondo è %.3f, del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p4[0],p4[1],p4[2],p4[3],p4[4],rsq4))



pylab.figure('resvsgain')


pylab.errorbar( gain, energres, 0 , fmt= '.', ecolor= 'magenta',markersize=10)

pylab.xlabel('gain [db]')
pylab.ylabel('energy resolution ')


pylab.title('Energy Resolution vs Attenuation')

pylab.plot(y1, np.polyval(p1,y1),'g--',label="linear fit")
pylab.plot(y1, np.polyval(p2,y1),'b--',label="square fit")
pylab.plot(y1, np.polyval(p3,y1),'m-',label="cubic fit")
pylab.plot(y1, np.polyval(p4,y1),'r--', label="quartic fit")
pylab.legend()
pylab.grid()
pylab.xlim(32,44)
pylab.ylim(0,0.1)
#pylab.savefig('resvsgain.png')

pylab.show()
#pylab.close()
print('le risoluzioni energetiche dei segnali test sono:')
print(energres)