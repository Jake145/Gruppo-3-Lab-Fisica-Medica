import numpy as np
from lmfit.models import ExponentialGaussianModel,SkewedGaussianModel
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
filenames=glob.glob('data*.txt')



for i in range(len(filenames)):
    f=filenames[i]

    print(f)

    data=np.loadtxt(f,unpack=True)
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



    plt.figure('%s'%f.replace('.txt',''))
    bin_heights, bin_borders, _=plt.hist(data,bins,facecolor='g',ec='black',alpha=0.5,label='histogram data',density=True)
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

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
    if i<6 and i!=1:
        mod=ExponentialGaussianModel()
        text='Exponential Gaussian Fit'
    elif i==1 or i==6 or i>6:
        mod=SkewedGaussianModel()
        text='Skewed Gaussian Fit'
    else:
        print('ERROR')
    pars=mod.guess(y,x=x)

    out = mod.fit(y, pars, x=x)


    plt.title('Histogram Resolution of %s '%f.replace('.txt',''))
    plt.xlabel('adc')
    plt.plot(x, out.best_fit, 'r-', label=text)
    plt.grid()
    plt.legend()
    plt.ylabel('frequency')
    plt.savefig('hist%s.png'%f.replace('.txt',''))
    plt.show()