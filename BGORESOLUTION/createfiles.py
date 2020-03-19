import pylab
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import urllib.request, zipfile, io
import os
import shutil
import glob

dataurl = 'https://docs.google.com/uc?export=download&id=183pCEe2KtC__kM9RO-hXJqFTc3qUGEsl'
files=['BGO1.F18.csv', 'BGO2.F18.csv', 'BGO3.F18.csv',
       'LSO1.F18.csv', 'LSO2.F18.csv', 'LSO3.F18.csv',
       'NaI1.Tc99m.csv', 'NaI2.Tc99m.csv', 'NaI3.Tc99m.csv',
       'LSO1.Ba133.csv', 'LSO2.Ba133.csv']
for f in files:
    try:
        os.remove(f)
    except:
        pass
old1=glob.glob('data*.txt')
for oldfiles in old1:
    try:
        remove(oldfiles)
    except:
        pass
old2=glob.glob('histdata*.txt')
for oldfiles2 in old2:
    try:
        remove(oldfiles2)
    except:
        pass
zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(dataurl).read())).extractall()

for i in range(len(files)):
    f=files[i]
    data=np.loadtxt(f,unpack=True)

    if i<6:
        name=f.replace('.F18.csv','F18')
        np.savetxt('data%s.txt'%(name),data)

    elif i==6 or i<9:
        name=f.replace('.Tc99m.csv','Tc99m')
        np.savetxt('data%s.txt'%(name),data)

    elif i==9 or i==10:
        name=f.replace('.Ba133.csv','Ba133')
        np.savetxt('data%s.txt'%(name),data)

    else:
        pass
for f in files:
    os.remove(f)

dir_path = '/Users/JakeHarold/Desktop/workplace/Gruppo-3-Lab-Fisica-Medica/BGORESOLUTION/__MACOSX'

try:
    shutil.rmtree(dir_path)
except OSError as e:
    print("Error: %s : %s" % (dir_path, e.strerror))
