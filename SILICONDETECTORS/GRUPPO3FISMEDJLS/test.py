
import glob
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats


capacity=1e-12 #in farad
Ee=3.6*1e-3 #in kev
e=1.6*1e-19 #in coulomb

def energymvmvfrommv(x):
    return capacity*x*0.001*Ee/e
    
y = np.loadtxt('incognitacapaicta.txt',unpack=True)
x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali
plt.figure('Visuale a Cadmio') #plot per vedere i dati
plt.plot(x, y, color='blue',marker = 'o')
plt.xlabel('chn')
plt.title('Cadmio')
plt.ylabel('count')
#plt.xlim(0,200)
plt.grid(True)
plt.show()
plt.savefig('cadmio.png')

##fit gaussiano 
#per qualche motivo mi da un indexerror, ma non fa alcuna differenza al fine del fit
data=y
a=300
b=400
mean=350
data[0:a]=-1
data[b:2048]=-1
x[0:a]=-1
x[b:2048]=-1
x=x[x>=0]
data=data[data>=0]
for i in range(len(data)):
    if data[i] == 0:
         
        x[i]=0

data=data[data>0]
x=x[x>0]
photopeakcount=np.sum(data[data>=0])
    

x1=np.linspace(0,2048,2048)
ds=np.sqrt(data) 
    


n = len(x)  #serve per i gradi di libertà                        
  

def gaus(x,a,x0,sig):#funzione gaussiana per il fit
    return a*np.exp(-(x-x0)**2/(2*sig**2))
    
popt,pcov = curve_fit(gaus,x,data,p0=[100,mean,1]) #trova i parametri ottimali (popt) e la matrice di covarianza(pcov).I parametri iniziali li ho stimati ad occhio

DOF=n-4 #gradi di libertà
chi2_1 = sum(((gaus(x,*popt)-data)/ds)**2) #calcolo chi quadro
da,dx0,dsig= np.sqrt(pcov.diagonal()) #dalla diagonale della matrice di covarianza trovo le     incertezze al quadrato dei parametri ottimali
chi2_1redux=chi2_1/DOF #chi quadro normalizzato con i gradi di libertà. deve essere più piccolo di 1 ma non eccessivamente!

a=popt[0]#popt è un vettore con i parametri ottimali
x0=popt[1]
sig=popt[2]
    
pvalue=1 - stats.chi2.cdf(chi2_1, DOF)#pvalue, deve essere maggiore di 0.005
print('il fattore moltiplicativo per Cadmio è %.3f, la media è %.2f, la sigma è %.2f'%(a,x0,sig))
print('il chi2 per Cadmio è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto per Cadmio è=%.3f '% (chi2_1redux))
print('il pvalue per Cadmio è=%.3f'% (pvalue))
print('Area sotto il fotopicco per Cadmio è : %.3f'%(photopeakcount))
  
##plot 
pylab.figure('fit gaussiano con Cadmio') 


pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('channel')
pylab.ylabel('counts')
#pylab.xlim(40,120)
    
pylab.title('gauss fit con Cadmio')
pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
pylab.grid()

#pylab.savefig('fit_gaussiano_con_Cadmio')
pylab.show()

xmedio=[]
sigma=[]
photopeaks=[]
tension=np.array([4,9,11,12])
filenames = glob.glob('capacita*.txt')

for f in filenames:
    print(f)

    y = np.loadtxt(fname=f, delimiter=',',unpack=True)
    x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali
  
    plt.figure('Visuale a %s'%f.replace('.txt','')) #plot per vedere i dati
    plt.plot(x, y, color='blue',marker = 'o')
    plt.xlabel('chn')
    plt.title('Americio con %s'%f.replace('.txt',''))
    plt.ylabel('count')
    plt.xlim(0,250)
    plt.grid(True)
    plt.show()
    plt.savefig('Americio_con_%s.png'%f.replace('.txt',''))
    plt.close()


##fit gaussiano 
#la tecnica è la seguente: dal grafico precedente isolo ad occhio il fotopicco e vedo quali sono i dati che non sono nel fotopicco: dall'asse x vedo quali corrispondono e metto quegli elementi del vettore a zero, e poi faccio la stessa cosa agli elementi con gli stessi indici del vettore ordinata. Poi con una mask elimino quegli elementi
    data=y
    a=200
    b=500
    mean=350
    data[0:a]=-1
    data[b:2048]=-1
    x[0:a]=-1
    x[b:2048]=-1
    x=x[x>=0]
    data=data[data>=0]
    for i in range(len(data)):
        if data[i] == 0:
         
            x[i]=0

    data=data[data>0]
    x=x[x>0]

    photopeakcount=np.sum(data[data>=0])


    x1=np.linspace(0,2048,2048)
    ds=np.sqrt(data) 



    n = len(x)  #serve per i gradi di libertà                        
  

    def gaus(x,a,x0,sig):#funzione gaussiana per il fit
        return a*np.exp(-(x-x0)**2/(2*sig**2))

    popt,pcov = curve_fit(gaus,x,data,p0=[100,mean,30]) #trova i parametri ottimali (popt) e la matrice di covarianza(pcov).I parametri iniziali li ho stimati ad occhio

    DOF=n-4 #gradi di libertà
    chi2_1 = sum(((gaus(x,*popt)-data)/ds)**2) #calcolo chi quadro
    da,dx0,dsig= np.sqrt(pcov.diagonal()) #dalla diagonale della matrice di covarianza trovo le     incertezze al quadrato dei parametri ottimali
    chi2_1redux=chi2_1/DOF #chi quadro normalizzato con i gradi di libertà. deve essere più piccolo di 1 ma non eccessivamente!

    a=popt[0]#popt è un vettore con i parametri ottimali
    x0=popt[1]
    sig=popt[2]
   
    pvalue=1 - stats.chi2.cdf(chi2_1, DOF)#pvalue, deve essere maggiore di 0.005
    print('il fattore moltiplicativo per %s è %.3f, la media è %.2f, la sigma è %.2f' % (f.replace('.txt',''),a,x0,sig))
    print('il chi2 per %s è=%.3f, i DOF sono=%.3f' % (f.replace('.txt',''),chi2_1, DOF))
    print('il chi2 ridotto per %s è=%.3f '% (f.replace('.txt',''),chi2_1redux))
    print('il pvalue per %s è=%.3f'% (f.replace('.txt',''),pvalue))
    print('Area sotto il fotopicco per %s è : %.3f'%(f.replace('.txt',''),photopeakcount))
  
##plot 
    pylab.figure('fit gaussiano con %s'%f.replace('.txt','')) 


    pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

    pylab.xlabel('channel')
    pylab.ylabel('counts')
    #pylab.xlim(150,250)
    
    pylab.title('gauss fit con %s'%f.replace('.txt',''))
    pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
    pylab.grid()


    pylab.show()
    #plt.savefig('Fit_Gaussiano_con_%s.png'%f.replace('.txt',''))
    #plt.close()
    xmedio.append(x0)
    sigma.append(sig)
    photopeaks.append(photopeakcount)
fwhm1=[ 0.00028181,  0.0002922,   0.00032011 , 0.00027445]
fwhm=2.35*np.abs(sigma)
resolution=fwhm/xmedio
print('le fwhm sono:')
print(fwhm)
print('le risoluzioni energetiche  sono:')
print(resolution/90)

fwmh=resolution

y1=np.linspace(0,30,1000) 


capacity=[12.4,18.6,27.1,6.9]
cap=np.array([12.4,18.6,27.1,6.9])
fwhm=fwhm

ds=0.1*fwhm

def f1(x,m,q):

    y=m*x+q
    
    return y 
     

     
popt, pcov= curve_fit(f1, cap, fwhm, (0.,0.),ds,absolute_sigma=False)
DOF=len(cap)-3
chi2_1 = sum(((f1(cap,*popt)-fwhm)/ds)**2)
dm,dq= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

m=popt[0]
q=popt[1]
print(*popt)

print(pcov)

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('il coefficiente angolare per la calibrazione del segnale test è %.3f pm %.3f, la intercetta è %.3f pm %.3f' % (m,dm,q,dq))
print('il chi2 per la calibrazione del segnale test è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto per la calibrazione del segnale test è=%.3f '% (chi2_1redux))
print('il pvalue per la calibrazione del segnale test è=%.3f'% (pvalue))
print('la funzione per la calibrazione dei segnali test è ENERGY = %.2f * CHANNEL + %.2f -11'%(m,q))

pylab.figure('resvscap')


pylab.errorbar( cap, fwhm, ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('Chn')
pylab.ylabel('energy[KeV]')


pylab.title('calibration Test signals')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()
#pylab.xlim(70,350)
#pylab.ylim(20,100)

pylab.show()

y = np.loadtxt('incognitacapaicta.txt',unpack=True)
x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali
plt.figure('Visuale a Cadmio') #plot per vedere i dati
plt.plot(x, y, color='blue',marker = 'o')
plt.xlabel('chn')
plt.title('Cadmio')
plt.ylabel('count')
#plt.xlim(0,200)
plt.grid(True)
plt.show()
#plt.savefig('cadmio.png')

##fit gaussiano 
#per qualche motivo mi da un indexerror, ma non fa alcuna differenza al fine del fit
data=y
a=300
b=400
mean=350
data[0:a]=-1
data[b:2048]=-1
x[0:a]=-1
x[b:2048]=-1
x=x[x>=0]
data=data[data>=0]
for i in range(len(data)):
    if data[i] == 0:
         
        x[i]=0

data=data[data>0]
x=x[x>0]
photopeakcount=np.sum(data[data>=0])
    

x1=np.linspace(0,2048,2048)
ds=np.sqrt(data) 
    


n = len(x)  #serve per i gradi di libertà                        
  

def gaus(x,a,x0,sig):#funzione gaussiana per il fit
    return a*np.exp(-(x-x0)**2/(2*sig**2))
    
popt,pcov = curve_fit(gaus,x,data,p0=[100,mean,1]) #trova i parametri ottimali (popt) e la matrice di covarianza(pcov).I parametri iniziali li ho stimati ad occhio

DOF=n-4 #gradi di libertà
chi2_1 = sum(((gaus(x,*popt)-data)/ds)**2) #calcolo chi quadro
da,dx0,dsig= np.sqrt(pcov.diagonal()) #dalla diagonale della matrice di covarianza trovo le     incertezze al quadrato dei parametri ottimali
chi2_1redux=chi2_1/DOF #chi quadro normalizzato con i gradi di libertà. deve essere più piccolo di 1 ma non eccessivamente!

a=popt[0]#popt è un vettore con i parametri ottimali
x0=popt[1]
sig=popt[2]
    
pvalue=1 - stats.chi2.cdf(chi2_1, DOF)#pvalue, deve essere maggiore di 0.005
print('il fattore moltiplicativo per Cadmio è %.3f, la media è %.2f, la sigma è %.2f'%(a,x0,sig))
print('il chi2 per Cadmio è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto per Cadmio è=%.3f '% (chi2_1redux))
print('il pvalue per Cadmio è=%.3f'% (pvalue))
print('Area sotto il fotopicco per Cadmio è : %.3f'%(photopeakcount))
  
##plot 
pylab.figure('fit gaussiano con Cadmio') 


pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('channel')
pylab.ylabel('counts')
#pylab.xlim(40,120)
    
pylab.title('gauss fit con Cadmio')
pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
pylab.grid()

#pylab.savefig('fit_gaussiano_con_Cadmio')
pylab.show()


fwhminc=2.35*sig/x0
print(fwhminc)
fwhminc=fwhminc/90
print(fwhminc)
def incognitcapacity(x):
    return (x-q)/m
capacity1=incognitcapacity(fwhminc)
print('LA CAPACITà INCOGNITA è : %7.f'%(capacity1))
capacity.append(capacity1)
fwhm1.append(fwhminc)
pylab.figure('resvscap')


pylab.errorbar( cap, fwhm, ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('Chn')
pylab.ylabel('energy[KeV]')


pylab.title('calibration Test signals')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()
#pylab.xlim(70,350)
#pylab.ylim(20,100)

pylab.show()