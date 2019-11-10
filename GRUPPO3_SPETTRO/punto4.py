import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats
d=4.76
a=5.94
solid_angle=2*np.pi*(1-d/(d**2+a**2))
geo_acceptance=4*np.pi/solid_angle
x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali
y=np.loadtxt('Am563stext.txt') #carica il txt delle acquisizioni
fondo=np.loadtxt('fondotext.txt') #carica il txt del fondo

z=(y/max(y))-(fondo/max(fondo)) #vettore dei dati normalizzati con il fondo sottratto

z[z<0]=0 # una sorta di unit test che elimina eventuali dati negativi
data=z*max(y) #tolgo la normalizzazione
plt.figure('Americio senza fondo')#plot dati senza fondo
plt.title('Americio senza fondo')
plt.plot(x, data, color='green',marker = 'o')
plt.xlabel('chn')
plt.ylabel('count')
plt.grid(True)
plt.show()

data[0:65]=0
data[109:2048]=0
photopeakcount=np.sum(data[data>0])
print(data[data>0])
print(photopeakcount)

