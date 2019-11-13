import numpy as np
from scipy.stats import*
import pylab
x=np.linspace(0,10,10)
t1=np.array([54.0, 54.2,54.8,54.5,54.6])
t2=np.array([30.8,31.0,30.8,30.8,31.0])
t3=np.array([9.76,9.84,9.79,9.79,9.81])
t4=np.array([67.6,68.1,68.0,67.8,67.9])
x1=np.linspace(1,5,5)

def mean(y):
    z=sum(y)/len(y)
    return z
    
p1=np.polyfit(x1,t1,1)
slope,intercept,r_value,pvalue,std_err=linregress(x1,t1)
print('Per 80 kVp il coefficiente angolare è %.3f, la intercetta è %.3f, R value è %.3f, p value è %.3f, la deviazione standard è %.3f, la media è %.3f' % (slope,intercept,r_value,pvalue,std_err,mean(t1)))
pylab.figure(1) 
pylab.errorbar( x1, t1, std_err , fmt= '.', ecolor= 'magenta')
pylab.xlabel('tries')
pylab.ylabel('dose [muGy]')
pylab.title('Riproducibilità a 80 kVp')
pylab.plot(x1,t1,'o', color='green', label="points")
pylab.plot(x,np.polyval(p1,x),'g')
pylab.xlim(0,6)
pylab.ylim(53.9,55)
pylab.grid()



p2=np.polyfit(x1,t2,1)
slope,intercept,r_value,pvalue,std_err=linregress(x1,t2)
print('Per 60 kVp il coefficiente angolare è %.3f, la intercetta è %.3f, R value è %.3f, p value è %.3f, la deviazione standard è %.3f, la media è %.3f' % (slope,intercept,r_value,pvalue,std_err,mean(t2)))
pylab.figure(2) 
pylab.errorbar( x1, t2, std_err , fmt= '.', ecolor= 'magenta')
pylab.xlabel('tries')
pylab.ylabel('dose [muGy]')
pylab.title('Riproducibilità a 60 kVp')
pylab.plot(x1,t2,'o', color='green', label="points")
pylab.plot(x,np.polyval(p2,x),'g')
pylab.xlim(0,6)
pylab.ylim(30.7,31.1)
pylab.grid()



p3=np.polyfit(x1,t3,1)
slope,intercept,r_value,pvalue,std_err=linregress(x1,t3)
print('Per 40 kVp il coefficiente angolare è %.3f, la intercetta è %.3f, R value è %.3f, p value è %.3f, la deviazione standard è %.3f, la media è %.3f' % (slope,intercept,r_value,pvalue,std_err,mean(t3)))
pylab.figure(3) 
pylab.errorbar( x1, t3, std_err , fmt= '.', ecolor= 'magenta')
pylab.xlabel('tries')
pylab.ylabel('dose [muGy]')
pylab.title('Riproducibilità a 40 kVp')
pylab.plot(x1,t3,'o', color='green', label="points")
pylab.plot(x,np.polyval(p3,x),'g')
pylab.xlim(0,6)
pylab.ylim(9.73,9.9)
pylab.grid()


p3=np.polyfit(x1,t4,1)
slope,intercept,r_value,pvalue,std_err=linregress(x1,t4)
print('Per 90 kVp il coefficiente angolare è %.3f, la intercetta è %.3f, R value è %.3f, p value è %.3f, la deviazione standard è %.3f, la media è %.3f' % (slope,intercept,r_value,pvalue,std_err,mean(t4)))
pylab.figure(4) 
pylab.errorbar( x1, t4, std_err , fmt= '.', ecolor= 'magenta')
pylab.xlabel('tries')
pylab.ylabel('dose [muGy]')
pylab.title('Riproducibilità a 90 kVp')
pylab.plot(x1,t3,'o', color='green', label="points")
pylab.plot(x,np.polyval(p3,x),'g')
pylab.xlim(0,6)
pylab.ylim(67.,69.)
pylab.grid()
pylab.show()

