
import numpy as np
import glob
import numpy as np
import matplotlib.pyplot as plt

import os
from scipy.stats import norm


#from statistics import NormalDist



import shutil
photons=[]
tappeto=[] #tempi

t=50 #in picoseconds
oldcrap=glob.glob('new_acquisition_C*.txt')
for f in oldcrap:
    print(f)
    os.remove(f)

maxindexes=[]
filenames = glob.glob('C3*.txt')
for f in filenames:
    print(f)
    fin = open( f, "r" )
    data_list = fin.readlines()
    fin.close()
    del data_list[0:3]
    fout = open('new_acquisition_%s'%f.replace('C3--XX--',''), "w")
    fout.writelines(data_list)
    fout.close()
crap=glob.glob('new_acquisition_C*.txt')
for f in crap:
    os.remove(f)
print('done!')
newfiles=glob.glob('new_acquisition_0*.txt')
for f in newfiles:
    print(f)
    data=np.loadtxt(f,unpack=True)
    os.remove(f)
    x=np.linspace(0,len(data),len(data))*t

    p2=np.polyfit(x,data,2)
    yfit2=p2[0]*x**2+p2[1]*x + p2[2]
    yres2=data-yfit2
    SSresid2=sum(pow(yres2,2))
    SStotal2=len(data)*np.var(data)
    rsq2=1-SSresid2/SStotal2
    try:
        rsq2>0.8
    except:
        print('ATTENTION: THE FIT SUCKS')
    Vth=5
    def intersection(x):
        return (-p2[1]+np.sqrt(abs((p2[1]**2-4*p2[0]*(p2[2]-x)))))/(2*p2[0])

    tappeto.append(intersection(Vth))

    (mu, sigma) = norm.fit(data)



    #n, bins, patches = plt.hist(data, len(data),facecolor='g',alpha=0.75)
    #(mu, sigma) = norm.fit(data)
    if mu>957 and mu<1915:
        photons.append(0)
    elif mu>2923 and mu<3780:
        photons.append(1)
    elif mu>4838 and mu<5796:
        photons.append(2)
    elif mu>7006 and mu<7916:
        photons.append(3)
    elif mu>8921 and mu<10131:
        photons.append(4)
    elif mu>11139 and mu<12248:
        photons.append(5)
    elif mu>13356 and mu<14264:
        photons.append(6)
    elif mu>15675 and mu<16179:
        photons.append(7)
    else:
        photons.append(-1)

    maxindex=np.sum(data)
    maxindexes.append(maxindex)
    del data
c=[tappeto,photons]
with open("valori_tempo_e_numero_fotoni.txt", "w") as file:
    for x in zip(*c):
        file.write("{0}\t{1}\n".format(*x))


data=np.abs(maxindexes)*t
#print(data)

n, bins, patches = plt.hist(data, len(data),facecolor='g',alpha=0.75)
plt.xlabel('number of cells excited')
plt.ylabel('Amplitude of waveform ')
plt.title('Histogram of %s'%f.replace('new_aquisition.txt',''))
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.xlim(40, 160)
#plt.ylim(0, 0.03)
plt.savefig('acquisition_more')
plt.grid(True)
plt.show()
