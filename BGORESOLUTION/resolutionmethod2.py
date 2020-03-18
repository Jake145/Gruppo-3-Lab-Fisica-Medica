import numpy as np
from lmfit.models import ExponentialGaussianModel,SkewedGaussianModel
from lmfit import Model
import os
import matplotlib.pyplot as plt

import glob
bins=200
#cancelliamo file vecchi
crap1=glob.glob('hist*.png')
for craps in crap1:
    try:
        remove(craps)
    except:
        pass
#iniziamo il tutto caricando i dati
filenames=glob.glob('data*.txt')



for i in range(len(filenames)):
    f=filenames[i]

    print(f)

    data=np.loadtxt(f,unpack=True)
#qui definisco gli estremi dei fotopicchi
    if i==0:
        a=2.3
        b=10
    if i==1:
        a=2.4
        b=10
    if i==2:
        a=3.5
        b=10
    if i==3:
        a=2.3
        b=4
    if i==4:
        a=2.2
        b=4
    if i==5:
        a=2.3
        b=10
    if i==6:
        a=1.25
        b=1.4
    if i==7:
        a=1
        b=1.2
    if i==8:
        a=0.47
        b=1.4
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
    if i<6 and i!=1:
        mod=ExponentialGaussianModel()
        text='Exponential Gaussian Fit'
    elif i==1 or i==6 or i>6:
        mod=SkewedGaussianModel()
        text='Skewed Gaussian Fit'
    else:
        print('ERROR')
#indovina i parametri iniziali
    pars=mod.guess(y,x=x)
#fit
    out = mod.fit(y, pars, x=x)
#ora calcoliamo le risoluzioni

    fwhm=out.params['fwhm'].value
    center=out.params['center'].value
    resolution=100*fwhm/center
# salviamo i risultati del fit su un txt
    with open('fit_result%s.txt'%f.replace('.txt',''), 'w') as fh:
        fh.write(out.fit_report())
#figura
    plt.title('Histogram Resolution of %s '%f.replace('.txt',''))
    plt.xlabel('adc')
    plt.plot(x, out.best_fit, 'r-', label=text)
    plt.plot([], [], ' ', label='Resolution: %.2f'%resolution)
    plt.grid()
    plt.legend()
    plt.ylabel('frequency')
    plt.savefig('hist%s.png'%f.replace('.txt',''))
    plt.show()