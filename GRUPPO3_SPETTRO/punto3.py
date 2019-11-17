
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats


x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali
y=np.loadtxt('mucalcCs300s400mm1Altext.txt') #carica il txt delle acquisizioni
fondo=np.loadtxt('fondotext.txt') #carica il txt del fondo



mu_Cu=np.array([933.72,931.77,931.15,930.70,930.76,929.70])#vettori dei valori che ho ottenuto 
sigma_Cu=np.array([28.12,28.82,28.92,28.33,29.00,29.33])    #sui vari file facendo il fit gaussiano
mu_Al=np.array([927.59,930.60,928.24,927.59,926.62])
sigma_Al=np.array([27.59,27.96,28.24,29.37,30.75])


print(len(y))
print(len(x))
plt.figure('Cesio') #plot per vedere i dati
plt.plot(x, y, color='blue',marker = 'o')
plt.xlabel('chn')

plt.ylabel('count')
plt.grid(True)
plt.show()



plt.figure('fondo') #plot per vedere il fondo normalizzato
plt.title('fondo')
plt.plot(x, fondo/max(fondo), color='green',marker = 'o')
plt.xlabel('chn')

plt.ylabel('count')
plt.grid(True)
plt.show()

z=(y/max(y))-(fondo/max(fondo)) #vettore dei dati normalizzati con il fondo sottratto

z[z<0]=0 # una sorta di unit test che elimina eventuali dati negativi
data=z*max(y) #tolgo la normalizzazione
plt.figure('Cesio senza fondo')#plot dati senza fondo
plt.title('Cesio senza fondo')

plt.plot(x, data, color='green',marker = 'o')
plt.xlabel('chn')

plt.ylabel('count')
plt.grid(True)
plt.show()
##fit gaussiano 
#la tecnica è la seguente: dal grafico precedente isolo ad occhio il fotopicco e vedo quali sono i dati che non sono nel fotopicco: dall'asse x vedo quali corrispondono e metto quegli elementi del vettore a zero, e poi faccio la stessa cosa agli elementi con gli stessi indici del vettore ordinata. Poi con una mask elimino quegli elementi
data[0:860]=0 
data[986:2048]=0
x[0:860]=0
x[986:2048]=0
x=x[x>0]
data=data[data>0]


photopeakcount=np.sum(data[data>0])



x1=np.linspace(0,2048,2048)
ds=np.sqrt(data) #errore poissoniano. Forse ho sbagliato la formula?
n = len(x)  #serve per i gradi di libertà                        
  

def gaus(x,a,x0,sig):#funzione gaussiana per il fit
    return a*np.exp(-(x-x0)**2/(2*sig**2))

popt,pcov = curve_fit(gaus,x,data,p0=[100,934,30]) #trova i parametri ottimali (popt) e la matrice di covarianza(pcov).I parametri iniziali li ho stimati ad occhio

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
print('il fattore moltiplicativo è %.3f, la media è %.2f, la sigma è %.2f' % (a,x0,sig))
print('il chi2 è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto è=%.3f '% (chi2_1redux))
print('il pvalue è=%.3f'% (pvalue))

##plot 
pylab.figure('fit gaussiano') #ho usato pylab anziché matplotlib.pyplot. Oramai è così :-)


pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('channel')
pylab.ylabel('counts')


pylab.title('gauss fit')
pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()




##Fit Rame 
y1=np.linspace(-10,300,1000)    #genero una ascissa a caso per il fit
w=0.25 #spessore moneta rame singolo in mm
spessori=np.array([0,w,2*w,3*w,4*w,5*w])
sigma_Cu=2.56*sigma_Cu
photopeaksCu=np.array([14035.35,12285.41,10528.31,9070.20,7839.17,6956.78])

 
def f1(x,mu,ch_0):

    y=ch_0*np.exp(-mu*x)
    
    return y 
     

     
popt, pcov= curve_fit(f1, spessori, photopeaksCu, (0.,0.),sigma_Cu,absolute_sigma=False)
DOF=len(spessori)-3
chi2_1 = sum(((f1(spessori,*popt)-photopeaksCu)/sigma_Cu)**2)
dmu,dch_0= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

mu=popt[0]
ch_0=popt[1]
print(*popt)

print(pcov)

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('il coefficiente di assorbimento del rame è %.3f pm %.3f, la costante moltiplicativa è %.3f pm %.3f' % (mu,dmu,ch_0,dch_0))
print('il chi2 è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto è=%.3f '% (chi2_1redux))
print('il pvalue è=%.3f'% (pvalue))

pylab.figure('Assorbimento Rame') 


pylab.errorbar( spessori, photopeaksCu, sigma_Cu , fmt= '.', ecolor= 'magenta')

pylab.xlabel('width[cm]')
pylab.ylabel('Photopeak Area')


pylab.title('Assorbimento Rame')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()
## Fit Alluminio
k=1.51 #spessore alluminio piccolo singolo in cm
h=2.01
spessoriAl=np.array([0,k,2*k,3*k,3*k+h])
photopeaksAl=np.array([10482.54,7938.11,6065.97,4180.16,2921.69])
sigma_Al=2.56*sigma_Al
popt, pcov= curve_fit(f1, spessoriAl, photopeaksAl, (0.,0.),sigma_Al,absolute_sigma=False)
DOF=len(spessoriAl)-3
chi2_1 = sum(((f1(spessoriAl,*popt)-photopeaksAl)/sigma_Al)**2)
dmu,dch_0= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

mu=popt[0]
ch_0=popt[1]
print(*popt)

print(pcov)

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('il coefficiente di assorbimento di Al è %.3f pm %.3f, la costante moltiplicativa è %.3f pm %.3f' % (mu,dmu,ch_0,dch_0))
print('il chi2 è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto è=%.3f '% (chi2_1redux))
print('il pvalue è=%.3f'% (pvalue))

pylab.figure('Assorbimento Alluminio') 


pylab.errorbar( spessoriAl, photopeaksAl, sigma_Al , fmt= '.', ecolor= 'magenta')

pylab.xlabel('width [cm]')
pylab.ylabel('Photpeak Area')


pylab.title('Assorbimento Alluminio')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()
print('Area sotto il fotopicco è %.2f' % (photopeakcount))
##Appunto per rimuovere outlier
#quello che si può fare è mettere le ordinate degli outlier uguali a zero, fare un ciclo for (purtroppo) che pone a zero le ascisse corrispondenti agli indici degli elementi delle ordinate uguali a zero e poi con una mask eliminare gli elementi uguali a zero
