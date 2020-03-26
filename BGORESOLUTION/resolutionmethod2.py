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

m_electron_c2=500 #in Kev
energy=511 #in Kev
r_e=8e-30
alpha=1/137
def ferg(x,A,B):
    return m_electron_c2/energy+1-m_electron_c2/(x*A+B)
def Utility_1(x,A,B):
    return (1+ferg(x,A,B)**2)/2
def Utility_2(x,A,B):
    return (1/(1+alpha*(1-ferg(x,A,B))))**2
def Utility_3(x,A,B):
    return (1+((alpha**2)*((1-ferg(x,A,B))**2))/((1+ferg(x,A,B)**2)*(1+alpha*(1-ferg(x,A,B)))))

def kleinnishina(x,A,B,Z):
    return Z*r_e*Utility_1(x,A,B)*Utility_2(x,A,B)*Utility_3(x,A,B)

plt.figure('klein-nishina cross section')
plt.title('klein-nishina cross section')
xplot=np.linspace((5/11)*energy,energy,10000)
plt.plot(xplot,kleinnishina(xplot,1,0,1)/np.amax(kleinnishina(xplot,1,0,1)),'g-',label='Normalized Klein Nishina Cross Section')
plt.legend()
plt.xlabel('Residual Energy [KeV]')
plt.ylabel('Normalized Cross Section [u.a.]')

plt.grid()
plt.savefig('/Users/JakeHarold/Desktop/workplace/Gruppo-3-Lab-Fisica-Medica/BGORESOLUTION/latex/kleincrosssection.png')
plt.show()
###


for i in range(len(filenames)):
    f=filenames[i]
    pathsavefigure=('/Users/JakeHarold/Desktop/workplace/Gruppo-3-Lab-Fisica-Medica/BGORESOLUTION/latex/histklein%s.png'%f.replace('.txt',''))

    print(f)

    data=np.loadtxt(f,unpack=True)
#qui definisco gli estremi dei fotopicchi
    if i==0:
        a=2.9
        b=4.7
    if i==1:
        a=2.29
        b=4
    if i==2:
        a=3.5
        b=6.4
    if i==3:
        a=1.76
        b=2.8
    if i==4:
        a=2.3
        b=3.1
    if i==5:
        a=1.75
        b=3.1
    if i==6:
        a=3.2
        b=4.6
    if i==7:
        a=2.6
        b=4
    if i==8:
        a=1.2
        b=1.4
    if i==9:
        a=0.9
        b=1.2
    if i==10:
        a=0.481
        b=0.66

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

#in base allo spettro, identificato da un indice che va da 0 a 10, fitto una gaussiana o una doppia gaussiana
    if i!=3 and i!=5:

        text='Gaussian Model w/ linear background & Expn. Decay'
        peak=GaussianModel()

    elif i==3 or i==5:
        text='Double Gaussian Model w/ linear background & Expn. Decay'
        peak1=GaussianModel(prefix='peak1')
        peak2=GaussianModel(prefix='peak2')
        peak=peak1+peak2

    noise=LinearModel()
#indovina i parametri iniziali
##Code Esponenziali
    tails=ExponentialModel(prefix='exp')
    mod=peak + noise + tails
    if i!=3 and i!=5:
        parspeak=peak.guess(y,x=x)

    elif i==3 or i==5 :

        parspeak1=peak1.guess(y,x=x)
        parspeak2=peak2.guess(y,x=x)
        parspeak=parspeak1+parspeak2

    else:
        print('ERROR GUESS')
    parslinear=noise.guess(y,x=x)

    parstails=tails.guess(y,x=x)
    pars=parspeak+parslinear+parstails
    out = mod.fit(y, pars, x=x)

    ##klein-Nishina

    klein=Model(kleinnishina)
    parsklein=klein.make_params(A=1,B=0,Z=1)
    if i!=3 and i!=5:
        mod2=peak + klein +noise

        parss=parspeak+parslinear+parsklein
        out2=mod2.fit(y,parss,x=x)
    if i==3 or i==5:
        mod2=peak + klein +noise + tails
        parss=parspeak+parslinear+parstails+parsklein
        out2=mod2.fit(y,parss,x=x)









#ora calcoliamo le risoluzioni
###GAUSSIANA SINGOLA
    if i!=3 and i!=5:
        ##Exponential decay resolution
        fwhm=out.params['fwhm'].value
        center=out.params['center'].value
        resolution=100*fwhm/center
        ##Klein resolution
        fwhmk=out2.params['fwhm'].value
        centerk=out2.params['center'].value
        resolutionk=100*fwhmk/centerk
        A=out2.params['A'].value
        B=out2.params['B'].value
        Z=out2.params['A'].value
###DOPPIA GAUSSIANA
    elif i==3 or i==5:
        ##
        fwhm1=out.params['peak1fwhm'].value
        fwhm2=out.params['peak2fwhm'].value
        center1=out.params['peak1center'].value
        center2=out.params['peak2center'].value
        resolution1=100*fwhm1/center1
        resolution2=100*fwhm2/center2

        fwhm1k=out2.params['peak1fwhm'].value
        fwhm2k=out2.params['peak2fwhm'].value
        center1k=out2.params['peak1center'].value
        center2k=out2.params['peak2center'].value
        resolution1k=100*fwhm1k/center1k
        resolution2k=100*fwhm2k/center2k
        A=out2.params['A'].value

    else:
        print('error')
# salviamo i risultati del fit su un txt
    with open('fit_result%s.txt'%f.replace('.txt',''), 'w') as fh:
        fh.write(out.fit_report())

    with open('fit_resultKlein%s.txt'%f.replace('.txt',''), 'w') as fh:
        fh.write(out2.fit_report())
#figura


    plt.title('Histogram Resolution of %s '%f.replace('.txt',''))
    plt.xlabel('adc')
    plt.plot(x, out.best_fit, 'r-', label=text)

    if i!=3 and i!=5:
        plt.plot([], [], ' ', label='Resolution: %.2f percent'%resolution)
        plt.plot([], [], ' ', label='Center of Photopeak: %.2f'%center)
        plt.plot(x, out2.best_fit, 'b--', label='Klein Nishina noise')
        plt.plot([], [], ' ', label='Resolution Klein: %.2f percent'%resolutionk)
        plt.plot([], [], ' ', label='Center of Photopeak K-N: %.2f Scale factor: %.2f'%(centerk,Z))
    elif i==3 or i==5:
        plt.plot(x, out2.best_fit, 'b--', label='Klein Nishina noise \w Exponential decay')
        plt.plot([], [], ' ', label='Resolution first peak: %.3f percent'%resolution1)
        plt.plot([], [], ' ', label='Center of first Photopeak: %.3f'%center1)
        plt.plot([], [], ' ', label='Resolution Second Photopeak: %.3f percent'%resolution2)
        plt.plot([], [], ' ', label='Center of second Photopeak: %.3f'%center2)
        plt.plot([], [], ' ', label='Resolution Klein First Peak: %.3f percent'%resolution1k)
        plt.plot([], [], ' ', label='Center of Klein First Peak K-N: %.3f'%center1k)
        plt.plot([], [], ' ', label='Resolution Klein Second Peak: %.3f percent'%resolution2k)
        plt.plot([], [], ' ', label='Center of Klein Second Peak K-N: %.3f'%center2k)
        plt.plot([], [], ' ', label='Scale Factor: %.2f'%Z)

    plt.grid()
    plt.legend()
    plt.ylabel('frequency')
    if i==3 or i==5:
        plt.xlim(0,3)
    plt.savefig(pathsavefigure)
    plt.show()
