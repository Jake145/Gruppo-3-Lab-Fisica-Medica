import numpy as np
import matplotlib.pyplot as plt
import pylab
x=np.array([133.6,140.0,150.3,161.4,171.5,181.3,195.6,156.1,166.2])
y=np.array([50818,50658,45013,25507,10344,1453,10,40628,21081])
#x=np.sort(x)
#y=np.sort(y)

plt.figure('tappeto')
plt.plot(x,y,marker='o',label='punti sperimentali')
plt.xlabel('Vth [mv]')
plt.ylabel('counts')
plt.grid()
plt.legend()

'''

pylab.figure('tappeto')

pylab.errorbar(x,y,0,marker='o',label='punti sperimentali')

pylab.xlabel('Vth [mv]')
pylab.ylabel('counts')
pylab.grid()
pylab.legend()
pylab.show()
'''
k=np.array([168.3,88,120,159,172,180,169.7,156.0,150,145.1,140.8,135.5,125.1,118.8,113.5,108.9,101.1])
z=np.array([75,379,342,150,24,9,58,137,226,245,290,312,315,324,329,351,341])
#k=np.sort(k)
#z=np.sort(z)
w=([101.1,116.3,132.0,141])
g=([57537,58162,60702,61450])
plt.figure('tappeto grande')
plt.plot(w,g,marker='o',label='punti sperimentali')
plt.xlabel('Vth [mv]')
plt.ylabel('counts')
plt.grid()
plt.legend()
plt.show()

plt.figure('tappeto più grande')
plt.plot(w,g,marker='o',label='punti')
plt.xlabel('Vth [mv]')
plt.ylabel('counts')
plt.grid()
plt.legend()
plt.show()


##Discriminatore
signal=k
counts=z
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
print('la media per %s è %.3f pm %.3f, la sigma è %.3f pm %.3f , Ampiezza superiore è %.3f pm %.3f , Ampiezza2 è %.3f pm %.3f'  % ((f.replace('Discriminatore.txt',''),mu,dmu,sigma,dsigma,A1,dA1,A2,dA2)))
print('il chi2 per %s è=%.3f, i DOF sono=%.3f' % (f.replace('Discriminatore.txt',''),chi2_1, DOF))
print('il chi2 ridotto per %s è=%.3f '% (f.replace('Discriminatore.txt',''),chi2_1redux))
print('il pvalue per %s è=%.3f'% (f.replace('Discriminatore.txt',''),pvalue))
    
pylab.figure('Discriminatore con %s'%(f.replace('Discriminatore.txt','')))
    
    
pylab.errorbar( signal  , counts, dc , fmt= '.', ecolor= 'magenta')
    
pylab.xlabel('Amplitude [mV]')
pylab.ylabel('Counts')
pylab.xlim(100,200)
    
pylab.title('Discriminatore con C=%s *10^-1 pf'%f.replace('CDiscriminatore.txt',''))
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()
    
pylab.savefig('discriminatore_%s.png'%f.replace('Discriminatore.txt',''))
pylab.show()
pylab.close()


