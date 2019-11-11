
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats
##questa prima parte è identica al punto 1
x=np.linspace(0,2048,2048)
y=np.loadtxt('inverserootlawCs676mmtext.txt')
fondo=np.loadtxt('fondotext.txt')


plt.figure('Cesio a una certa distanza')
plt.plot(x, y, color='blue',marker = 'o')
plt.xlabel('chn')

z=(y/max(y))-(fondo/max(fondo))

z[z<0]=0
data=z*max(y)
plt.figure('Cesio senza fondo')
plt.plot(x, data, color='green',marker = 'o')
plt.xlabel('chn')

plt.ylabel('count')
plt.grid(True)
#plt.show()
##fit gaussiano 
#serve per la sigma, perla tecnica vedi punto1.py
data[0:880]=0 
data[980:2048]=0
x[0:880]=0
x[980:2048]=0
x=x[x>0]
data=data[data>0]
x1=np.linspace(0,2048,2048)
ds=np.sqrt(data) #errore poissoniano. 
n = len(x)  #serve per i gradi di libertà                        
  

def gaus(x,a,x0,sig):#funzione gaussiana per il fit
    return a*np.exp(-(x-x0)**2/(2*sig**2))

popt,pcov = curve_fit(gaus,x,data,p0=[10,940,30]) #trova i parametri ottimali (popt) e la matrice di covarianza(pcov).I parametri iniziali li ho stimati ad occhio

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
##calcolo area sotto il fotopicco
#tecnica simile al punto1, se volete potete aggiungerci il metodo di covell
data[0:840]=0
data[1000:2048]=0
photopeakcount=np.sum(data[data>0])

print(photopeakcount)
##fit converge sempre solo su root

sumphotopeak=np.array([89642.86,37152.39,19532.97,12149.04,13953.44,7728.73,5676.84,4271.79,3382.63,2859.33])
distance=np.array([266,316,366,400,416,476,526,576,626,676])
sigma=np.array([26.58,26.94,27.55,28.15,27.94,27.12,27.98,27.65,27.32,27.22])
sigma=2.56*sigma

y1=np.linspace(1,2000,1000)    #genero una ascissa a caso per il fit


def f1(x,a,q):

    y=a/x**2 + q
    
    return y 
     

     
popt, pcov= curve_fit(f1, distance, sumphotopeak, (6e9,2000),sigma,absolute_sigma=False)
DOF=len(distance)-3
chi2_1 = sum(((f1(distance,*popt)-sumphotopeak)/sigma)**2)
da,dq= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

a=popt[0]
q=popt[1]
print(*popt)

print(pcov)

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('il coefficiente moltiplicativo è %.2f, la costante da sommare è %.3f' % (a,q))
print('il chi2 è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto è=%.3f '% (chi2_1redux))
print('il pvalue è=%.3f'% (pvalue))

pylab.figure('andamento 1/r^2') 


pylab.errorbar( distance, sumphotopeak, sigma , fmt= '.', ecolor= 'magenta')

pylab.xlabel('width')
pylab.ylabel('channel')


pylab.title('andamento 1/r^2')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()

print(sigma)






