import numpy as np
from scipy.stats import*
import pylab

y1=np.linspace(0,1100,1000)    #genero una ascissa a caso per il fit
dose,time=pylab.loadtxt('dosevstime.txt',unpack=True) 
Ds=0.03*dose 
def mean(y):
    z=sum(y)/len(y)
    return z
    

slope,intercept,r_value,pvalue,std_err=linregress(time,dose)
print('Per il fit lineare coefficiente angolare è %.3f, la intercetta è %.3f, R value è %.3f, p value è %.3f, la deviazione standard è %.3f, la media è %.3f' % (slope,intercept,r_value,pvalue,std_err,mean(dose)))
p1=np.polyfit(time,dose,1)
p2=np.polyfit(time,dose,2)






##calcolo r^2
yfit=p2[0]*time**2+p2[1]*time + p2[2]
yres=dose-yfit
SSresid=sum(pow(yres,2))
SStotal=len(dose)*np.var(dose)
rsq=1-SSresid/SStotal
print('Per il fit quadratico il coefficiente del secondo grado è %.3f, del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p2[0],p2[1],p2[2],rsq))
##plot 

pylab.figure('calibrazione')


pylab.errorbar( time, dose, Ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('time [ms]')
pylab.ylabel('dose [muGy]')


pylab.title('Dose vs Time')
pylab.plot(y1,np.polyval(p1,y1), 'g-', label="linear fit")
pylab.plot(y1, np.polyval(p2,y1),'r--',label="quadratic fit")
pylab.legend()
pylab.grid()


pylab.show()