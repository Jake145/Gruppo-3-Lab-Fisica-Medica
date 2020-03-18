import numpy as np
from lmfit.models import ExponentialGaussianModel
from lmfit import Model
import os
import matplotlib.pyplot as plt

import glob

bins=200

crap1=glob.glob('hist*.png')
for craps in crap1:
    try:
        remove(craps)
    except:
        pass
filenames=glob.glob('dataBGO2.txt')
#def fit_func(x, l, s, m):
 #   return 0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*x))*sse.erfc((m+l*s*s-x)/(np.sqrt(2)*s))


for i in range(len(filenames)):
    f=filenames[i]

    print(f)

    data=np.loadtxt(f,unpack=True)



    plt.figure('%s'%f.replace('.txt',''))
    bin_heights, bin_borders, _=plt.hist(data,bins,facecolor='g',ec='black',alpha=0.5,label='histogram data',density=True)
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

    newcenters=[]
    newheights=[]
    for j in range(len(bin_heights)):
        if bin_centers[j]<10 and bin_centers[j]>2.3:
            newcenters.append(bin_centers[j])
            newheights.append(bin_heights[j])
        else:
            pass
    y=np.array(newheights)
    x=np.array(newcenters)

    mod=ExponentialGaussianModel()
    pars=mod.guess(y,x=x)

    out = mod.fit(y, pars, x=x)
    #popt,pcov=curve_fit(fit_func, x, y,p0)
    y1=np.linspace(np.min(x)-0.1*np.min(x),np.max(x)+0.3*np.max(x),1000)

    plt.title('Histogram Resolution of %s '%f.replace('.txt',''))
    plt.xlabel('adc')
    plt.plot(x, out.best_fit, 'r-', label='best fit')
    plt.grid()
    plt.legend()
    plt.ylabel('frequency')
    plt.savefig('hist%s.png'%f.replace('.txt',''))
    plt.show()