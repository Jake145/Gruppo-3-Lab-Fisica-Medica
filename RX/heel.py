import numpy as np
from scipy.stats import*
import pylab

y1=np.linspace(0,40,1000) 

#hvl 100 kvp

dose,distance=np.loadtxt('heel100kvp.txt',unpack=True)
rel_I=dose/max(dose)
ds=0.03*rel_I

p1=np.polyfit(distance,rel_I,1)
yfit1=p1[0]*distance+p1[1]
yres1=rel_I-yfit1
SSresid1=sum(pow(yres1,2))
SStotal1=len(rel_I)*np.var(rel_I)
rsq1=1-SSresid1/SStotal1
print('Per il fit lineare il coefficiente del primo grado è %.3f, del termine costante è %.3f, R value è %.3f' % (p1[0],p1[1],rsq1))


p2=np.polyfit(distance,rel_I,2)
yfit2=p2[0]*distance**2+p2[1]*distance +p2[2]
yres2=rel_I-yfit2
SSresid2=sum(pow(yres2,2))
SStotal2=len(rel_I)*np.var(rel_I)
rsq2=1-SSresid2/SStotal2
print('Per il fit quadratico il coefficiente del secondo grado è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p2[0],p2[1],p2[2],rsq2))

p3=np.polyfit(distance,rel_I,3)
yfit3=p3[0]*distance**3+p3[1]*distance**2 + p3[2]*distance +p3[3]
yres3=rel_I-yfit3
SSresid3=sum(pow(yres3,2))
SStotal3=len(rel_I)*np.var(rel_I)
rsq3=1-SSresid3/SStotal3
print('Per il fit cubico il coefficiente del terzo grado è %.3f, del second è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p3[0],p3[1],p3[2],p3[3],rsq3))

pylab.figure('hvl100pvk')


pylab.errorbar( distance, rel_I, ds , fmt= '.', ecolor= 'magenta',)

pylab.xlabel('distance [mm]')
pylab.ylabel('relative intensity [%]')


pylab.title('Heel effect at 100 kvp')

pylab.plot(y1, np.polyval(p1,y1),'g--',label="linear fit")
pylab.plot(y1, np.polyval(p2,y1),'b--',label="square fit")
pylab.plot(y1, np.polyval(p3,y1),'r-',label="cubic fit")
pylab.legend()
pylab.grid()


pylab.show()