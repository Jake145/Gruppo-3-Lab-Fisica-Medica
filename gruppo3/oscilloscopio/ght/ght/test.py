import glob
import numpy as np
import matplotlib.pyplot as plt

import os

import shutil


maxindexes=[]
filenames = glob.glob('C*.txt')
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
    
    maxindex=np.argmax(data )
    maxindexes.append(maxindex)
    

data=np.abs(maxindexes)
print(data)

n, bins, patches = plt.hist(data, range(min(data),max(data)),facecolor='g',alpha=0.75)
plt.xlabel('number of cells excited')
plt.ylabel('Amplitude of waveform ')
plt.title('Histogram of %s'%f.replace('new_aquisition.txt',''))
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.xlim(40, 160)
#plt.ylim(0, 0.03)
plt.grid(True)
plt.show()