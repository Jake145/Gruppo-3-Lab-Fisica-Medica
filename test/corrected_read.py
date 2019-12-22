
import numpy as np
import glob
import numpy as np
import matplotlib.pyplot as plt

import os
from scipy.stats import norm






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
newfiles=glob.glob('new_acquisition_*.txt')
for f in newfiles:
    print(f)
    data=np.loadtxt(f,unpack=True)
    os.remove(f)

    a=np.abs(data)
    a=a[a>0.03]
    max=np.argmax(a)
    for i in range(len(a)):
        if i > max:
            a[i]=0
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
    Vth=0.5*np.ndarray.max(np.abs(a))
    def intersection(x):
        return (-p2[1]+np.sqrt(abs((p2[1]**2-4*p2[0]*(p2[2]-x)))))/(2*p2[0])

    tappeto.append(intersection(Vth))

    #(maxindex, sigma) = norm.fit(data)
    maxindex=np.sum(abs(data))*t


    #n, bins, patches = plt.hist(data, len(data),facecolor='g',alpha=0.75)
    #(maxindex, sigma) = norm.fit(data)
    if maxindex>907 and maxindex<1965:
        photons.append(0)
    elif maxindex>3300 and maxindex<4385:
        photons.append(1)
    elif maxindex>5900 and maxindex<7006:
        photons.append(2)
    elif maxindex>8400 and maxindex<9500:
        photons.append(3)
    elif maxindex>10937 and maxindex<11945:
        photons.append(4)
    elif maxindex>13558 and maxindex<14560:
        photons.append(5)
    elif maxindex>16431 and maxindex<17050:
        photons.append(6)
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

n, bins, patches = plt.hist(data, int(round(abs(np.sqrt(len(a))))),facecolor='g',alpha=0.75)
plt.xlabel('area under waveform')
plt.ylabel('frequency of are under waveform ')
plt.title('Histogram of 55 V waveform')
#plt.text(60, .025, r'$\maxindex=100,\ \sigma=15$')
#plt.xlim(40, 160)
#plt.ylim(0, 0.03)
plt.savefig('acquisition_more')
plt.grid(True)
plt.show()
