import numpy as np
from lmfit.models import GaussianModel,ExponentialModel,LinearModel
from lmfit import Model
import os
import matplotlib.pyplot as plt

import glob
bins=200
#cancelliamo file vecchi
old=glob.glob('hist*.png')
for oldfile in old:
    try:
        remove(oldfile)
    except:
        pass
#iniziamo il tutto caricando i dati
filenames=glob.glob('data*.txt')

###Ora definiamo la funzione di klein nishina

def kleinnishina(x,A,B):
    return A*((1+(np.cos(B)**2)/2))*(1/(1+(x**2)*(1-np.cos(B)))**2)*(1+(x*(1-np.cos(B))**2)/(((1+np.cos(B)**2)*(1+x*(1-np.cos(B))))))



###


for i in range(len(filenames)):
    f=filenames[i]

    print(f)

    data=np.loadtxt(f,unpack=True)
#qui definisco gli estremi dei fotopicchi
    if i==0:
        a=3
        b=4.7
    if i==1:
        a=2.29
        b=4
    if i==2:
        a=3.5
        b=6
    if i==3:
        a=1.76
        b=2.8
    if i==4:
        a=2.5
        b=3
    if i==5:
        a=1.75
        b=3.1
    if i==6:
        a=3.5
        b=4.6
    if i==7:
        a=2.8
        b=4
    if i==8:
        a=1.2
        b=1.4
    if i==9:
        a=0.97
        b=1.129
    if i==10:
        a=0.51
        b=0.63

    else:
        pass


#cominciamo la figura
    plt.figure('%s'%f.replace('.txt',''))
    bin_heights, bin_borders, _=plt.hist(data,bins,facecolor='g',ec='black',alpha=0.5,label='histogram data',density=True)
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
#qui creo il vettore per il fit sul fotopicco eliminando la parte che non interessa
    newcenters=[]
    newheights=[]
    for j in range(len(bin_heights)):
        if bin_centers[j]<b and bin_centers[j]>a:
            newcenters.append(bin_centers[j])
            newheights.append(bin_heights[j])
        else:
            pass
    y=np.array(newheights)
    x=np.array(newcenters)
#in base allo spettro fitto una gaussiana esponenzialmente corretta o una skewed
    if i!=3 and i!=5:

        text='Gaussian Model w/ linear background & Expn. Decay'
        peak=GaussianModel()

    elif i==3 or i==5:
        text='Double Gaussian Model w/ linear background & Expn. Decay'
        peak1=GaussianModel(prefix='peak1')
        peak2=GaussianModel(prefix='peak2')
        peak=peak1+peak2
#indovina i parametri iniziali
    noise=LinearModel()
    tails=ExponentialModel(prefix='exp')
    mod=peak + noise + tails
    if i!=3 and i!=5:
        parspeak=peak.guess(y,x=x)
    ###klein

        klein=Model(kleinnishina)
        mod2=peak + klein

        params = klein.make_params(A=1,B=np.pi)
        parss=params+parspeak
        out2=mod2.fit(y,parss,x=x)


    ###
    elif i==3 or i==5 :

        parspeak1=peak1.guess(y,x=x)
        parspeak2=peak2.guess(y,x=x)
        parspeak=parspeak1+parspeak2

    else:
        print('ERROR GUESS')
    parslinear=noise.guess(y,x=x)

    parstails=tails.guess(y,x=x)
    pars=parspeak+parslinear+parstails
#fit
    out = mod.fit(y, pars, x=x)


#ora calcoliamo le risoluzioni
    if i!=3 and i!=5:
        fwhm=out.params['fwhm'].value
        center=out.params['center'].value
        resolution=100*fwhm/center
        ##Klein resolution
        fwhmk=out2.params['fwhm'].value
        centerk=out2.params['center'].value
        resolutionk=100*fwhmk/centerk
        ##Klein resolution
    elif i==3 or i==5:
        fwhm1=out.params['peak1fwhm'].value
        fwhm2=out.params['peak2fwhm'].value
        center1=out.params['peak1center'].value
        center2=out.params['peak2center'].value
        resolution1=100*fwhm1/center1
        resolution2=100*fwhm2/center2
    else:
        print('error')
# salviamo i risultati del fit su un txt
    with open('fit_result%s.txt'%f.replace('.txt',''), 'w') as fh:
        fh.write(out.fit_report())
#figura


    plt.title('Histogram Resolution of %s '%f.replace('.txt',''))
    plt.xlabel('adc')
    plt.plot(x, out.best_fit, 'r-', label=text)
    plt.plot([], [], ' ', label='Linear Background')
    if i!=3 and i!=5:
        plt.plot([], [], ' ', label='Resolution: %.2f percent'%resolution)
        plt.plot([], [], ' ', label='Center of Photopeak: %.2f'%center)
        plt.plot(x, out2.best_fit, 'b--', label='Klein Nishina noise')
        plt.plot([], [], ' ', label='Resolution Klein: %.2f percent'%resolutionk)

    elif i==3 or i==5:
        plt.plot([], [], ' ', label='Resolution first peak: %.2f percent'%resolution1)
        plt.plot([], [], ' ', label='Center of first Photopeak: %.2f'%center1)
        plt.plot([], [], ' ', label='Resolution Second Photopeak: %.2f percent'%resolution2)
        plt.plot([], [], ' ', label='Center of second Photopeak: %.2f'%center2)


    plt.grid()
    plt.legend()
    plt.ylabel('frequency')
    plt.savefig('histklein%s.png'%f.replace('.txt',''))
    plt.show()