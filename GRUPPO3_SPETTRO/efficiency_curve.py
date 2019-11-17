
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats
##questa prima parte è identica al punto 1
x=np.linspace(0,2048,2048)
y=np.loadtxt('Am563stext.txt')
fondo=np.loadtxt('fondotext.txt')


z=(y/max(y))-(fondo/max(fondo))

z[z<0]=0
data=z*max(y)

plt.figure('Elemento senza fondo')
plt.plot(x, data, color='green',marker = 'o')
plt.xlabel('chn')
plt.title('Cesio senza fondo')
plt.ylabel('count')
plt.grid(True)
plt.show()
##fit gaussiano 

a=60
b=110
mean=90
data[0:a]=0 
data[b:2048]=0
x[0:a]=0
x[b:2048]=0
x=x[x>0]
data=data[data>0]
photopeakcount=np.sum(data[data>0])
print('area sotto il fotopicco è %.3f '%(photopeakcount))
x1=np.linspace(0,2048,2048)
ds=np.sqrt(data) 
n = len(x)                        
  

def gaus(x,C,x0,sig):
    return C*np.exp(-(x-x0)**2/(2*sig**2))

popt,pcov = curve_fit(gaus,x,data,p0=[10,mean,30]) 
DOF=n-4 
chi2_1 = sum(((gaus(x,*popt)-data)/ds)**2) 
dC,dx0,dsig= np.sqrt(pcov.diagonal()) 
chi2_1redux=chi2_1/DOF 

C=popt[0]
x0=popt[1]
sig=popt[2]


pvalue=1 - stats.chi2.cdf(chi2_1, DOF)

##plot 
pylab.figure('fit gaussiano') 


pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('channel')
pylab.ylabel('counts')


pylab.title('gauss fit Cs')
pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()


##fotopicchi

print(photopeakcount)
measured_decays=np.array([27685,2066,21104,2120,2473])
## efficienza
A_0=74 #measured in kBq
d=4.76
a=5.94
solid_angle=2*np.pi*(1-d/(d**2+a**2))
geo_acceptance=4*np.pi/solid_angle
half_life=np.array([432.2,30.7,2.6,5.27,5.27]) #measured in years
energy=np.array([60,511,662,1332,1774])

def realdecays(tau,t1,t2):
    y=A_0*tau*(np.exp(t2/tau)-np.exp(t1/tau))
    return y
    
def efficiency(x,y,geo):
    z=(x/y)*geo
    return z
    
def seconds(x):
    j=365*24*60*60*x
    return j
times=seconds(half_life)
print(times)
realdecays=realdecays(seconds(half_life),seconds(15),seconds(15)+1000)
efficiencies=efficiency(measured_decays,realdecays,geo_acceptance)
print(efficiencies)
print(len(measured_decays),len(efficiencies))
print(realdecays)

pylab.figure(3)
pylab.title('Grafico Efficienza')
pylab.xlabel('Energia [kev]')
pylab.xscale('log')
pylab.yscale('log')
pylab.ylabel('efficienza')
pylab.grid()
pylab.plot(energy,efficiencies,'b--',marker='o')


pylab.show()
