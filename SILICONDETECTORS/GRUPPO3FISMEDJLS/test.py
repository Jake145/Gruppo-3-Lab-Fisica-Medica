
import glob
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats
xmedio=[]
sigma=[]
photopeaks=[]
tension=np.array([4,9,11,12])
filenames = glob.glob('*V.txt')

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
    a=160
    b=240
    mean=200
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
