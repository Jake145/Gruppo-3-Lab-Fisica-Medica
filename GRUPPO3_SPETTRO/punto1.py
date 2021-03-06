import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats


x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali
y=np.loadtxt('cesio_839stext.txt') #carica il txt delle acquisizioni
fondo=np.loadtxt('fondotext.txt') #carica il txt del fondo


plt.figure('cesio') #plot per vedere i dati
plt.plot(x, y, color='blue',marker = 'o')
plt.xlabel('chn')
plt.title('Cesio')
plt.ylabel('count')
plt.grid(True)
plt.show()



plt.figure('fondo') #plot per vedere il fondo normalizzato
plt.title('Fondo')
plt.plot(x, fondo/max(fondo), color='green',marker = 'o')
plt.xlabel('chn')

plt.ylabel('count')
plt.grid(True)
plt.show()

z=(y/max(y))-(fondo/max(fondo)) #vettore dei dati normalizzati con il fondo sottratto

z[z<0]=0 # una sorta di unit test che elimina eventuali dati negativi
data=z*max(y) #tolgo la normalizzazione
plt.figure('Americio senza fondo')#plot dati senza fondo
plt.title('Cesio senza fondo')

plt.plot(x, data, color='green',marker = 'o')
plt.xlabel('chn')

plt.ylabel('count')
plt.grid(True)
plt.show()

##fit gaussiano 
#la tecnica è la seguente: dal grafico precedente isolo ad occhio il fotopicco e vedo quali sono i dati che non sono nel fotopicco: dall'asse x vedo quali corrispondono e metto quegli elementi del vettore a zero, e poi faccio la stessa cosa agli elementi con gli stessi indici del vettore ordinata. Poi con una mask elimino quegli elementi
a=199 #estremi della gaussiana
b=210
mean=200
data[0:a]=-1
data[b:2048]=-1
x[0:a]=-1
x[b:2048]=-1
x=x[x>=0]
data=data[data>=0]
for i in range(0,len(data)-1):
    if data[i] > 0:
         data=data
         x=x
    else :
        data=np.delete(data,i)
        x=np.delete(x,i)
x1=np.linspace(0,2048,2048)
ds=np.sqrt(data) #errore poissoniano. 
n = len(x)  #serve per i gradi di libertà                        
  

def gaus(x,a,x0,sig):#funzione gaussiana per il fit
    return a*np.exp(-(x-x0)**2/(2*sig**2))

popt,pcov = curve_fit(gaus,x,data,p0=[10,mean,20]) #trova i parametri ottimali (popt) e la matrice di covarianza(pcov).I parametri iniziali li ho stimati ad occhio

DOF=n-4 #gradi di libertà
chi2_1 = sum(((gaus(x,*popt)-data)/ds)**2) #calcolo chi quadro
da,dx0,dsig= np.sqrt(pcov.diagonal()) #dalla diagonale della matrice di covarianza trovo le incertezze al quadrato dei parametri ottimali
chi2_1redux=chi2_1/DOF #chi quadro normalizzato con i gradi di libertà. deve essere più piccolo di 1 ma non eccessivamente!

a=popt[0]#popt è un vettore con i parametri ottimali
x0=popt[1]
sig=popt[2]
print(*popt)

print(pcov)

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)#pvalue, deve essere maggiore di 0.005
print('FIT GAUSSIANO il fattore moltiplicativo è %.3f, la media è %.2f, la sigma è %.2f' % (a,x0,sig))
print('il chi2 è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto è=%.3f '% (chi2_1redux))
print('il pvalue è=%.3f'% (pvalue))

##plot 
pylab.figure('fit gaussiano') #ho usato pylab anziché matplotlib.pyplot. Oramai è così :-)


pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('channel')
pylab.ylabel('counts')


pylab.title('Fit Gaussiano Fotopicco Cs')

pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()










##calibrazione

y1=np.linspace(0,3000,1000)    #genero una ascissa a caso per il fit
energy,counts=pylab.loadtxt('provadata.txt',unpack=True) 
Ds=2.56*27 #errore a caso, usa la fwhm dal fit gaussiano
def f1(x,m,q):

    y=m*x+q
    
    return y 
     

     
popt, pcov= curve_fit(f1, counts, energy, (0.,0.),Ds,absolute_sigma=False)
DOF=len(counts)-3
chi2_1 = sum(((f1(counts,*popt)-energy)/Ds)**2)
dm,dq= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

m=popt[0]
q=popt[1]
print(*popt)

print(pcov)

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('il coefficiente angolare è %.3f pm %.3f, la intercetta è %.3f pm %.3f' % (m,dm,q,dq))
print('il chi2 è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto è=%.3f '% (chi2_1redux))
print('il pvalue è=%.3f'% (pvalue))

##plot 
pylab.figure('calibrazione')


pylab.errorbar( counts, energy, Ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('Chn')
pylab.ylabel('energy[KeV]')


pylab.title('calibration')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()

