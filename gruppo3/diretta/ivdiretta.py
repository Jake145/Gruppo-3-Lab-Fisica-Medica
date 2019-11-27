import numpy as np
import pylab
from scipy.optimize import curve_fit

voltage,current = np.loadtxt('ivdiretta2ma.txt',unpack=True)
y1=np.linspace(-60,60,1000)
p1=np.polyfit(voltage,current,1)
yfit1=p1[0]*voltage+p1[1]
yres1=current-yfit1
SSresid1=sum(pow(yres1,2))
SStotal1=len(current)*np.var(current)
rsq1=1-SSresid1/SStotal1
print('Per il fit lineare per la risoluzione vs voltage il coefficiente del primo grado è %.3f, del termine costante è %.3f, R value è %.3f' % (p1[0],p1[1],rsq1))

p2=np.polyfit(voltage,current,2)
yfit2=p2[0]*voltage**2+p2[1]*voltage +p2[2]
yres2=current-yfit2
SSresid2=sum(pow(yres2,2))
SStotal2=len(current)*np.var(current)
rsq2=1-SSresid2/SStotal2
print('Per il fit quadratico per la risoluzione vs voltage il coefficiente del secondo grado è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p2[0],p2[1],p2[2],rsq2))

p3=np.polyfit(voltage,current,3)
yfit3=p3[0]*voltage**3+p3[1]*voltage**2 + p3[2]*voltage +p3[3]
yres3=current-yfit3
SSresid3=sum(pow(yres3,2))
SStotal3=len(current)*np.var(current)
rsq3=1-SSresid3/SStotal3
print('Per il fit cubico per la risoluzione vs voltage il coefficiente del terzo grado è %.3f, del second è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p3[0],p3[1],p3[2],p3[3],rsq3))

p4=np.polyfit(voltage,current,4)
yfit4=p4[0]*voltage**4+p4[1]*voltage**3 + p4[2]*voltage**2 +p4[3]*voltage+p4[4]
yres4=current-yfit4
SSresid4=sum(pow(yres4,2))
SStotal4=len(current)*np.var(current)
rsq4=1-SSresid4/SStotal4
print('Per il fit quartico per la risoluzione vs voltage il coefficiente del quarto grado è %.6f, del terzo è %.3f,del secondo è %.3f, del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p4[0],p4[1],p4[2],p4[3],p4[4],rsq4))


p5=np.polyfit(voltage,current,7)
yfit5=p5[0]*voltage**7+p5[1]*voltage**6 + p5[2]*voltage**5 +p5[3]*voltage**4+p5[4]*voltage**3+p5[5]*voltage**2+p5[6]*voltage+p5[7]
yres5=current-yfit5
SSresid5=sum(pow(yres5,2))
SStotal5=len(current)*np.var(current)
rsq5=1-SSresid5/SStotal5
print('Per il fit quintico  per la risoluzione vs voltage il coefficiente del quarto grado è %.6f, del terzo è %.3f,del secondo è %.3f, del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p4[0],p4[1],p4[2],p4[3],p4[4],rsq4))



pylab.figure('tappeto')


pylab.errorbar( voltage, current, 0 , fmt= '.', ecolor= 'magenta')

pylab.xlabel('voltage [V]')
pylab.ylabel('Current [muA] ')


pylab.title('Energy Resolution vs Attenuation')

pylab.plot(y1, np.polyval(p1,y1),'g--',label="linear fit")
pylab.plot(y1, np.polyval(p2,y1),'b--',label="square fit")
pylab.plot(y1, np.polyval(p3,y1),'m-',label="cubic fit")
pylab.plot(y1, np.polyval(p4,y1),'r--', label="quartic fit")
pylab.plot(y1, np.polyval(p5,y1),'g-', label="fifth order fit")
pylab.legend()
pylab.grid()
#pylab.xlim(32,44)
#pylab.ylim(0,0.1)
pylab.savefig('ivtappeto.png')

pylab.show()