import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats


x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali
y=np.loadtxt('Am10Cutext.txt') #carica il txt delle acquisizioni





print(len(y))
print(len(x))
plt.figure('Americio con 0 spessori') #plot per vedere i dati
plt.plot(x, y, color='blue',marker = 'o')
plt.xlabel('chn')

plt.ylabel('count')
plt.grid(True)
plt.show()



##fit gaussiano 
#la tecnica è la seguente: dal grafico precedente isolo ad occhio il fotopicco e vedo quali sono i dati che non sono nel fotopicco: dall'asse x vedo quali corrispondono e metto quegli elementi del vettore a zero, e poi faccio la stessa cosa agli elementi con gli stessi indici del vettore ordinata. Poi con una mask elimino quegli elementi
data=y
a=199
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

photopeakcount=np.sum(data[data>=0])


x1=np.linspace(0,2048,2048)
ds=np.sqrt(data) 



n = len(x)  #serve per i gradi di libertà                        
  

def gaus(x,a,x0,sig):#funzione gaussiana per il fit
    return a*np.exp(-(x-x0)**2/(2*sig**2))

popt,pcov = curve_fit(gaus,x,data,p0=[100,mean,30]) #trova i parametri ottimali (popt) e la matrice di covarianza(pcov).I parametri iniziali li ho stimati ad occhio

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
print('Area sotto il fotopicco è : %.3f'%(photopeakcount))
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
y1=np.linspace(-0,1,1000)    #genero una ascissa a caso per il fit
w=0.09 #spessore moneta rame singolo in mum
spessori=np.array([0,w,2*w,3*w,4*w,5*w,6*w,7*w,8*w,9*w,10*w])
sigma=np.array([8.24,5.38,4.68,4.38,4.11,4.19,4.39,3.90,4.40,5.09,2.36])
sigma_Cu=2.56*sigma
#photopeaksCu=np.array([47420,28430,15615,8016,4038,1959,942,439,220,109,46])
photopeaksCu=np.array([83313,27259,14953,7842,3967,1889,904,446,236,113,66])
 
def f1(x,mu,ch_0,q):

    y=ch_0*np.exp(-mu*x)+q
    
    return y 
     

     
popt, pcov= curve_fit(f1, spessori, photopeaksCu, (0.,0.,0.),sigma_Cu,absolute_sigma=False)
DOF=len(spessori)-4
chi2_1 = sum(((f1(spessori,*popt)-photopeaksCu)/sigma_Cu)**2)
dmu,dch_0,dq= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

mu=popt[0]
ch_0=popt[1]
q=popt[2]




pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('il coefficiente di assorbimento del rame è %.3f pm %.3f, la costante moltiplicativa è %.3f pm %.3f e il valore costante è %.3f pm %.3f'  % (mu,dmu,ch_0,dch_0,q,dq))
print('il chi2 è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto è=%.3f '% (chi2_1redux))
print('il pvalue è=%.3f'% (pvalue))

pylab.figure('Assorbimento Rame') 


pylab.errorbar( spessori, photopeaksCu, sigma_Cu , fmt= '.', ecolor= 'magenta')

pylab.xlabel('width [cm]')
pylab.ylabel('Photopeak Area')


pylab.title('Assorbimento Rame')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()
