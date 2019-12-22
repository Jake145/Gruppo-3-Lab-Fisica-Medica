
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats


##CALIBRAZIONE SEGNALI TEST

tension=np.array([1.04,2.16,3.68,4.16]) #in mv
print('le tensioni sono')
print(tension)
gain=np.array([43,38,33.5,32.5]) #in db
chn=[]
sigma=[]
photopeaks=[]


import glob
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats

xmedio=[]
sigma=[]
photopeaks=[]

filenames = glob.glob('Calibratione*.txt')

for f in filenames:
    print(f)

    y = np.loadtxt(fname=f, delimiter=',',unpack=True)
    x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali

    plt.figure('Visuale a %s'%f.replace('.txt','')) #plot per vedere i dati
    plt.plot(x, y, color='blue',marker = 'o')
    plt.xlabel('chn')
    plt.title('Segnale test con %s'%f.replace('.txt',''))
    plt.ylabel('count')
    #plt.xlim(0,250)
    plt.grid(True)
    plt.show()
    data=y
    a=74
    b=360
    mean=200
    data[0:a]=-1
    data[b:2048]=-1
    x[0:a]=-1
    x[b:2048]=-1
    x=x[x>=0]
    data=data[data>=0]
    for i in range(len(data)):
        if data[i] == 0:

            x[i]=0

    data=data[data>0]
    x=x[x>0]

    photopeakcount=np.sum(data[data>=0])


    x1=np.linspace(0,2048,2048)
    ds=np.sqrt(data)



    n = len(x)  #serve per i gradi di libertà


    def gaus(x,a,x0,sig):#funzione gaussiana per il fit
        return a*np.exp(-(x-x0)**2/(2*sig**2))

    popt,pcov = curve_fit(gaus,x,data,p0=[100,mean,30]) #trova i parametri ottimali (popt) e la matrice di covarianza(pcov).I parametri iniziali li ho stimati ad occhio

    DOF=n-4 #gradi di libertà
    chi2_1 = sum(((gaus(x,*popt)-data)/ds)**2) #calcolo chi quadro
    da,dx0,dsig= np.sqrt(pcov.diagonal()) #dalla diagonale della matrice di covarianza trovo le     incertezze al quadrato dei parametri ottimali
    chi2_1redux=chi2_1/DOF #chi quadro normalizzato con i gradi di libertà. deve essere più piccolo di 1 ma non eccessivamente!

    a=popt[0]#popt è un vettore con i parametri ottimali
    x0=popt[1]
    sig=popt[2]

    pvalue=1 - stats.chi2.cdf(chi2_1, DOF)#pvalue, deve essere maggiore di 0.005
    #print('il fattore moltiplicativo per %s è %.3f, la media è %.2f, la sigma è %.2f' % (f.replace('.txt',''),a,x0,sig))
    #print('il chi2 per %s è=%.3f, i DOF sono=%.3f' % (f.replace('.txt',''),chi2_1, DOF))
   # print('il chi2 ridotto per %s è=%.3f '% (f.replace('.txt',''),chi2_1redux))
   # print('il pvalue per %s è=%.3f'% (f.replace('.txt',''),pvalue))
   # print('Area sotto il fotopicco per %s è : %.3f'%(f.replace('.txt',''),photopeakcount))

##plot
    pylab.figure('fit gaussiano con %s'%f.replace('.txt',''))


    pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

    pylab.xlabel('channel')
    pylab.ylabel('counts')
    #pylab.xlim(150,250)

    pylab.title('gauss fit con %s'%f.replace('.txt',''))
    pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
    pylab.grid()


    pylab.show()
    plt.savefig('Fit_Gaussiano_con_%s.png'%f.replace('.txt',''))
    #plt.close()
    chn.append(x0)
    sigma.append(sig)
    photopeaks.append(photopeakcount)



f= 'ECalibrationes416mv120v2.txt'
y=np.loadtxt(fname=f, delimiter=',',unpack=True)
x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali

plt.figure('Visuale a %s'%f.replace('.txt','')) #plot per vedere i dati
plt.plot(x, y, color='blue',marker = 'o')
plt.xlabel('chn')
plt.title('Segnale test con %s'%f.replace('.txt',''))
plt.ylabel('count')
#plt.xlim(0,250)
plt.grid(True)
plt.show()
data=y
a=74
b=360
mean=350
data[0:a]=-1
data[b:2048]=-1
x[0:a]=-1
x[b:2048]=-1
x=x[x>=0]
data=data[data>=0]
for i in range(len(data)):
    if data[i] == 0:

        x[i]=0

data=data[data>0]
x=x[x>0]

photopeakcount=np.sum(data[data>=0])


x1=np.linspace(0,2048,2048)
ds=np.sqrt(data)



n = len(x)  #serve per i gradi di libertà


def gaus(x,a,x0,sig):#funzione gaussiana per il fit
    return a*np.exp(-(x-x0)**2/(2*sig**2))

popt,pcov = curve_fit(gaus,x,data,p0=[100,mean,30]) #trova i parametri ottimali (popt) e la matrice di covarianza(pcov).I parametri iniziali li ho stimati ad occhio

DOF=n-4 #gradi di libertà
chi2_1 = sum(((gaus(x,*popt)-data)/ds)**2) #calcolo chi quadro
da,dx0,dsig= np.sqrt(pcov.diagonal()) #dalla diagonale della matrice di covarianza trovo le     incertezze al quadrato dei parametri ottimali
chi2_1redux=chi2_1/DOF #chi quadro normalizzato con i gradi di libertà. deve essere più piccolo di 1 ma non eccessivamente!

a=popt[0]#popt è un vettore con i parametri ottimali
x0=popt[1]
sig=popt[2]

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)#pvalue, deve essere maggiore di 0.005
#print('il fattore moltiplicativo per %s è %.3f, la media è %.2f, la sigma è %.2f' % (f.replace('.txt',''),a,x0,sig))
#print('il chi2 per %s è=%.3f, i DOF sono=%.3f' % (f.replace('.txt',''),chi2_1, DOF))
#print('il chi2 ridotto per %s è=%.3f '% (f.replace('.txt',''),chi2_1redux))
#print('il pvalue per %s è=%.3f'% (f.replace('.txt',''),pvalue))
#print('Area sotto il fotopicco per %s è : %.3f'%(f.replace('.txt',''),photopeakcount))

##plot
pylab.figure('fit gaussiano con %s'%f.replace('.txt',''))


pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('channel')
pylab.ylabel('counts')
    #pylab.xlim(150,250)

pylab.title('gauss fit con %s'%f.replace('.txt',''))
pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()
plt.savefig('Fit_Gaussiano_con_%s.png'%f.replace('.txt',''))
    #plt.close()
chn.append(x0)
sigma.append(sig)
photopeaks.append(photopeakcount)

fwhm=2.35*np.abs(sigma)
resolution=np.abs(fwhm)/np.abs(chn)
print('i canali medi sono:')
print(chn)
print('le fwhm sono:')
print(fwhm)
print('le risoluzioni energetiche  sono:')
print(resolution)

##
#chn=np.array([85.58,176.57,305.20,342.69])
#fwhm=np.array([1.69,1.69,1.66,1.73])

capacity=1e-12 #in farad
Ee=3.6*1e-3 #in kev
e=1.6*1e-19 #in coulomb
print('le tensioni sono')
print(tension)
def energymvmvfrommv(x):
    return capacity*x*0.001*Ee/e
print('le energie predette sono:')

energymvmv=energymvmvfrommv(tension)

print(energymvmv)
print('le attenuazioni sono:')
print(gain)

y1=np.linspace(0,3000,1000)

def f1(x,m,q):

    y=m*np.abs(x)+q

    return y



popt, pcov= curve_fit(f1, np.abs(chn), tension, (0.,0.),fwhm,absolute_sigma=False)
DOF=len(chn)-3
chi2_1 = sum(((f1(np.abs(chn),*popt)-np.abs(tension))/np.abs(fwhm))**2)
dm,dq= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

m=popt[0]
q=popt[1]
print(*popt)

print(pcov)

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('il coefficiente angolare per la calibrazione del segnale test MILLIVOLT VS CHANNEL è %.3f pm %.3f, la intercetta è %.3f pm %.3f' % (m,dm,q,dq))
print('il chi2 per la calibrazione  MILLIVOLT VS CHANNEL del segnale test è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto per la calibrazione  MILLIVOLT VS CHANNEL del segnale test è=%.3f '% (chi2_1redux))
print('il pvalue per la calibrazione  MILLIVOLT VS CHANNEL del segnale test è=%.3f'% (pvalue))
print('la funzione per la calibrazione  MILLIVOLT VS CHANNEL dei segnali test è ENERGY = %.2f * CHANNEL + %.2f -11'%(m,q))
##plot
pylab.figure('calibrazioneMVvsCHN')


pylab.errorbar( chn, tension, 0 , fmt= '.', ecolor= 'magenta')

pylab.xlabel('Chn')
pylab.ylabel('Tension[mv]')


pylab.title('calibration Test signals')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()
pylab.xlim(0,420)
pylab.ylim(0,6)
pylab.savefig('calibration_test_signals.png')
pylab.show()
#pylab.close()
##calibration for the corresponding energy of test signal
popt, pcov= curve_fit(f1, chn, energymvmv, (0.,0.),fwhm,absolute_sigma=False)
DOF=len(chn)-3
chi2_1 = sum(((f1(chn,*popt)-energymvmv)/fwhm)**2)
dm,dq= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

m=popt[0]
q=popt[1]
print(*popt)

print(pcov)

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('il coefficiente angolare per la calibrazione del segnale ENERGY VS CHANNEL test è %.3f pm %.3f, la intercetta è %.3f pm %.3f' % (m,dm,q,dq))
print('il chi2 per la calibrazione del segnale test è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto per la calibrazione ENERGY VS CHANNEL del segnale test è=%.3f '% (chi2_1redux))
print('il pvalue per la calibrazione ENERGY VS CHANNEL del segnale test è=%.3f'% (pvalue))
print('la funzione per la calibrazione ENERGY VS CHANNEL dei segnali test è ENERGY = %.2f * CHANNEL + %.2f -11'%(m,q))

pylab.figure('calibrazione EvsCHN')


pylab.errorbar( chn, energymvmv, fwhm , fmt= '.', ecolor= 'magenta')

pylab.xlabel('Chn')
pylab.ylabel('Energy [KeV]')


pylab.title('calibration Test signals Energy')
pylab.plot(y1,f1(y1,*popt), color='red', label="fit")
pylab.grid()
pylab.xlim(0,400)
pylab.ylim(0,102)
pylab.savefig('calibration_test_signals_energy.png')
pylab.show()
#pylab.close()
###Questa parte riguarda la risoluzine vs attenuazione

##RISOLUZIONE ENERGETICA VS GAIN
tension=np.array([1.04,2.16,3.68,4.16]) #in mv
gain=np.array([43,38,33.5,32.5]) #in db

energymvmv=energymvmvfrommv(tension)
energres=resolution
y1=np.linspace(0,50,1000)
p1=np.polyfit(gain,energres,1)
yfit1=p1[0]*gain+p1[1]
yres1=energres-yfit1
SSresid1=sum(pow(yres1,2))
SStotal1=len(energres)*np.var(energres)
rsq1=1-SSresid1/SStotal1
print('Per il fit lineare per la risoluzione vs gain il coefficiente del primo grado è %.3f, del termine costante è %.3f, R value è %.3f' % (p1[0],p1[1],rsq1))

p2=np.polyfit(gain,energres,2)
yfit2=p2[0]*gain**2+p2[1]*gain +p2[2]
yres2=energres-yfit2
SSresid2=sum(pow(yres2,2))
SStotal2=len(energres)*np.var(energres)
rsq2=1-SSresid2/SStotal2
print('Per il fit quadratico per la risoluzione vs gain il coefficiente del secondo grado è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p2[0],p2[1],p2[2],rsq2))

p3=np.polyfit(gain,energres,3)
yfit3=p3[0]*gain**3+p3[1]*gain**2 + p3[2]*gain +p3[3]
yres3=energres-yfit3
SSresid3=sum(pow(yres3,2))
SStotal3=len(energres)*np.var(energres)
rsq3=1-SSresid3/SStotal3
print('Per il fit cubico per la risoluzione vs gain il coefficiente del terzo grado è %.3f, del second è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p3[0],p3[1],p3[2],p3[3],rsq3))

p4=np.polyfit(gain,energres,4)
yfit4=p4[0]*gain**4+p4[1]*gain**3 + p4[2]*gain**2 +p4[3]*gain+p4[4]
yres4=energres-yfit4
SSresid4=sum(pow(yres4,2))
SStotal4=len(energres)*np.var(energres)
rsq4=1-SSresid4/SStotal4
print('Per il fit quartico per la risoluzione vs gain il coefficiente del quarto grado è %.6f, del terzo è %.3f,del secondo è %.3f, del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p4[0],p4[1],p4[2],p4[3],p4[4],rsq4))



pylab.figure('resvsgain')


pylab.errorbar( gain, energres, 0 , fmt= '.', ecolor= 'magenta',markersize=10)

pylab.xlabel('gain [db]')
pylab.ylabel('energy resolution ')


pylab.title('Energy Resolution vs Attenuation')

pylab.plot(y1, np.polyval(p1,y1),'g--',label="linear fit")
pylab.plot(y1, np.polyval(p2,y1),'b--',label="square fit")
pylab.plot(y1, np.polyval(p3,y1),'r-',label="cubic fit")

pylab.legend()
pylab.grid()
pylab.xlim(32,44)
pylab.ylim(-0.1,0.3)
pylab.savefig('resvsgain.png')

pylab.show()
#pylab.close()
print('le risoluzioni energetiche dei segnali test sono:')
print(energres)
##FWHM VS CAPACITY

y1=np.linspace(0,30,1000)



cap,fwhm=np.loadtxt('1capacita.txt',unpack=True)
fwhm=fwhm/energymvmvfrommv(4.16)

ds=0.1*fwhm

p1=np.polyfit(cap,fwhm,1)
yfit1=p1[0]*cap+p1[1]
yres1=fwhm-yfit1
SSresid1=sum(pow(yres1,2))
SStotal1=len(fwhm)*np.var(fwhm)
rsq1=1-SSresid1/SStotal1
print('Per il fit lineare di RESOLUTION VS CAPACITY il coefficiente del primo grado è %.3f, del termine costante è %.3f, R value è %.3f' % (p1[0],p1[1],rsq1))

p2=np.polyfit(cap,fwhm,2)
yfit2=p2[0]*cap**2+p2[1]*cap +p2[2]
yres2=fwhm-yfit2
SSresid2=sum(pow(yres2,2))
SStotal2=len(fwhm)*np.var(fwhm)
rsq2=1-SSresid2/SStotal2
print('Per il fit quadratico di RESOLUTIO VS CAPACITY il coefficiente del secondo grado è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p2[0],p2[1],p2[2],rsq2))

p3=np.polyfit(cap,fwhm,3)
yfit3=p3[0]*cap**3+p3[1]*cap**2 + p3[2]*cap +p3[3]
yres3=fwhm-yfit3
SSresid3=sum(pow(yres3,2))
SStotal3=len(fwhm)*np.var(fwhm)
rsq3=1-SSresid3/SStotal3
print('Per il fit cubico il di RESOLUTIO VS CAPACITY coefficiente del terzo grado è %.3f, del second è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p3[0],p3[1],p3[2],p3[3],rsq3))

p4=np.polyfit(cap,fwhm,4)
yfit4=p4[0]*cap**4+p4[1]*cap**3 + p4[2]*cap**2 +p4[3]*cap+p4[4]
yres4=fwhm-yfit4
SSresid4=sum(pow(yres4,2))
SStotal4=len(fwhm)*np.var(fwhm)
rsq4=1-SSresid4/SStotal4
print('Per il fit quartico di RESOLUTIO VS CAPACITY il coefficiente del quarto grado è %.6f, del terzo è %.3f,del secondo è %.3f, del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p4[0],p4[1],p4[2],p4[3],p4[4],rsq4))



pylab.figure('resvscap')


pylab.errorbar( cap, fwhm, ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('cap [pf]')
pylab.ylabel('energy resolution')
#pylab.ylim(0.015,0.030)

pylab.title('Energy resolution vs capacity')

pylab.plot(y1, np.polyval(p1,y1),'g--',label="linear fit")
pylab.plot(y1, np.polyval(p2,y1),'b-',label="square fit")
pylab.plot(y1, np.polyval(p3,y1),'m--',label="cubic fit")
pylab.plot(y1, np.polyval(p4,y1),'r--', label="quartic fit")
pylab.legend()
pylab.grid()
pylab.savefig('resvvscapacity')

pylab.show()
#pylab.close()
## trova la capacità incognita
#Calcola la fwhm
fwhminc=1.84
fwhminc=fwhminc/energymvmvfrommv(4.16)
def incognitcapacity(x):
    return (-p2[1]+np.sqrt(p2[1]**2+4*p2[0]*(x-p2[2])))/(2*p2[0])
capacity=incognitcapacity(fwhminc)
print('LA CAPACITà INCOGNITA è : %.3f'%(capacity))
