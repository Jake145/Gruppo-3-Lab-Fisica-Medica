import glob
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
dchn=[]
sigma=[]
dsigma=[]
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
    plt.xlim(0,250)
    plt.grid(True)
    plt.show()
    #plt.close()
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
    print('il fattore moltiplicativo per %s è %.3f, la media è %.2f, la sigma è %.2f' % (f.replace('.txt',''),a,x0,sig))
    print('il chi2 per %s è=%.3f, i DOF sono=%.3f' % (f.replace('.txt',''),chi2_1, DOF))
    print('il chi2 ridotto per %s è=%.3f '% (f.replace('.txt',''),chi2_1redux))
    print('il pvalue per %s è=%.3f'% (f.replace('.txt',''),pvalue))
    print('Area sotto il fotopicco per %s è : %.3f'%(f.replace('.txt',''),photopeakcount))

##plot
    pylab.figure('fit gaussiano con %s'%f.replace('.txt',''))


    pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

    pylab.xlabel('channel')
    pylab.ylabel('counts')
    pylab.xlim(300,400)

    pylab.title('gauss fit con %s'%f.replace('.txt',''))
    pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
    pylab.grid()


    pylab.show()
    plt.savefig('Fit_Gaussiano_con_%s.png'%f.replace('.txt',''))
    #plt.close()
    chn.append(x0)
    sigma.append(sig)
    photopeaks.append(photopeakcount)
    dchn.append(dx0)
    dsigma.append(dsig)

f= 'ECalibrationes416mv120v2.txt'
y=np.loadtxt(fname=f, delimiter=',',unpack=True)
x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali

plt.figure('Visuale a %s'%f.replace('.txt','')) #plot per vedere i dati
plt.plot(x, y, color='blue',marker = 'o')
plt.xlabel('chn')
plt.title('Segnale test con %s'%f.replace('.txt',''))
plt.ylabel('count')
plt.xlim(0,500)
plt.grid(True)
plt.show()
plt.savefig('Visualecon_%s.png'%f.replace('.txt',''))
plt.close()
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
print('il fattore moltiplicativo per %s è %.3f, la media è %.2f, la sigma è %.2f' % (f.replace('.txt',''),a,x0,sig))
print('il chi2 per %s è=%.3f, i DOF sono=%.3f' % (f.replace('.txt',''),chi2_1, DOF))
print('il chi2 ridotto per %s è=%.3f '% (f.replace('.txt',''),chi2_1redux))
print('il pvalue per %s è=%.3f'% (f.replace('.txt',''),pvalue))
print('Area sotto il fotopicco per %s è : %.3f'%(f.replace('.txt',''),photopeakcount))

##plot
pylab.figure('fit gaussiano con %s'%f.replace('.txt',''))


pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('channel')
pylab.ylabel('counts')
pylab.xlim(300,400)

pylab.title('gauss fit con %s'%f.replace('.txt',''))
pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()
plt.savefig('Fit_Gaussiano_con_%s.png'%f.replace('.txt',''))
plt.close()
chn.append(x0)
sigma.append(sig)
dsigma.append(dsig)
photopeaks.append(photopeakcount)
dchn.append(dx0)
fwhm=2.35*np.abs(sigma)
dfwhm=2.35*np.abs(dsigma)
resolution=np.abs(fwhm)/np.abs(chn)
def propagateerrorfwhm(chn,dchn,fwhm,dfwhm):
    return np.sqrt(((1/chn)*dfwhm)**2+((fwhm/chn**2)*dchn)**2)
print('PER I SEGNALI TEST I SEGUENTI VALORI CON INCERTEZZE:')
for i in range(len(chn)):
    print('Per il segnale %.3f abbiamo il canale medio %.3f +- %.3f abbiamo la fwhm %.3f +- %.3f e una risoluzione energetica %.3f +- %.3f '%(tension[i],chn[i],dchn[i],fwhm[i],dfwhm[i],resolution[i],propagateerrorfwhm(chn[i],dchn[i],fwhm[i],dfwhm[i])))




##CALIBRAZIONE MILLIVOT VERSUS CANALE
#chn=np.array([85.58,176.57,305.20,342.69])
#fwhm=np.array([1.69,1.69,1.66,1.73])

capacity=1e-12 #in farad
Ee=3.6*1e-3 #in kev
e=1.6*1e-19 #in coulomb

def energymvmvfrommv(x):
    return capacity*x*0.001*Ee/e

energymvmv=energymvmvfrommv(tension)
denergymvmv= 0.01/tension * energymvmv
for i in range(len(tension)):
    print('PER LA TENSIONE %.3f HO UNA ENERGIA DI %3.f +- %.3f e UNA ATTENUAZIONE DI %.3f '%(tension[i],energymvmv[i],denergymvmv[i],gain[i]))




y1=np.linspace(0,3000,1000)

def f1(x,m,q):

    y=m*np.abs(x)+q

    return y



popt, pcov= curve_fit(f1, np.abs(chn)-11, tension, (0.,0.),fwhm,absolute_sigma=False)
DOF=len(chn)-3
chi2_1 = sum(((f1(np.abs(chn)-11,*popt)-np.abs(tension))/np.abs(fwhm))**2)
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
print('la funzione per la calibrazione  MILLIVOLT VS CHANNEL dei segnali test è ENERGY = %.2f * CHANNEL + %.2f '%(m,q))
##plot
pylab.figure('calibrazioneMVvsCHN')


pylab.errorbar( np.abs(chn)-11, tension, 0 , fmt= '.', ecolor= 'magenta')

pylab.xlabel('Chn')
pylab.ylabel('Tension[mv]')


pylab.title('calibration Test signals')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()
pylab.xlim(0,420)
pylab.ylim(0,6)
pylab.savefig('calibration_test_signals.png')
pylab.show()
pylab.close()
##calibration for the corresponding energy of test signal
popt, pcov= curve_fit(f1, np.abs(chn)-11, energymvmv, (0.,0.),resolution*energymvmv,absolute_sigma=False)
DOF=len(chn)-3
chi2_1 = sum(((f1(np.abs(chn)-11,*popt)-energymvmv)/resolution*energymvmv)**2)
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
print('la funzione per la calibrazione ENERGY VS CHANNEL dei segnali test è ENERGY = %.2f * CHANNEL + %.2f '%(m,q))

pylab.figure('calibrazione EvsCHN')


pylab.errorbar( np.abs(chn)-11, energymvmv, resolution*energymvmv , fmt= '.', ecolor= 'magenta')

pylab.xlabel('Chn')
pylab.ylabel('Energy [KeV]')


pylab.title('calibration Test signals Energy')
pylab.plot(y1,f1(y1,*popt), color='red', label="fit")
pylab.grid()
pylab.xlim(0,400)
pylab.ylim(0,102)
pylab.savefig('calibration_test_signals_energy.png')
pylab.show()
pylab.close()
###CALCOLO INCERTEZZE SULL'ENERGIA

print('QUESTA PARTE RIGUARDA IL CALCOLO DELLE INCERTEZZE SULL ENERGIA DELLLA CALIBRAZIONE CON I SEGNALI TEST')

def calcoloincertezzaparte1(x):
    return np.sqrt((np.abs(m)*x)**2 +(x*dm)**2+dsig**2)
dE=calcoloincertezzaparte1(np.abs(dchn))
for i in range(len(chn)):
    print('per il canale %.3f incertezza sulla relativa energia è %.3f'%(chn[i],dE[i]))

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
pylab.close()

##FWHM VS CAPACITY

###fit gaussiani per trovare le fwhm delle capacità

chn=[]
sigma=[]
photopeaks=[]
dchn=[]
dsigma=[]
filenames = glob.glob('capacita*.txt')

for f in filenames:
    print(f)

    y = np.loadtxt(fname=f, delimiter=',',unpack=True)
    x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali

    plt.figure('Visuale a %s'%f.replace('.txt','')) #plot per vedere i dati
    plt.plot(x, y, color='blue',marker = 'o')
    plt.xlabel('chn')
    plt.title('Segnale test con %s'%f.replace('.txt',''))
    plt.ylabel('count')
    plt.xlim(0,500)
    plt.grid(True)
    plt.show()
    plt.close()
    data=y
    a=300
    b=400
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
    print('il fattore moltiplicativo per %s è %.3f, la media è %.2f, la sigma è %.2f' % (f.replace('.txt',''),a,x0,sig))
    print('il chi2 per %s è=%.3f, i DOF sono=%.3f' % (f.replace('.txt',''),chi2_1, DOF))
    print('il chi2 ridotto per %s è=%.3f '% (f.replace('.txt',''),chi2_1redux))
    print('il pvalue per %s è=%.3f'% (f.replace('.txt',''),pvalue))
    print('Area sotto il fotopicco per %s è : %.3f'%(f.replace('.txt',''),photopeakcount))

##plot
    pylab.figure('fit gaussiano con %s'%f.replace('.txt',''))


    pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

    pylab.xlabel('channel')
    pylab.ylabel('counts')
    pylab.xlim(0,500)

    pylab.title('gauss fit con %s'%f.replace('.txt',''))
    pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
    pylab.grid()


    pylab.show()
    plt.savefig('Fit_Gaussiano_con_%s.png'%f.replace('.txt',''))
    plt.close()
    chn.append(x0)
    sigma.append(sig)
    dchn.append(dx0)
    dsigma.append(dsig)
    photopeaks.append(photopeakcount)





###fit per la risoluzione vs capacità
y1=np.linspace(0,30,1000)



cap,shit=np.loadtxt('1capacita.txt',unpack=True)
fwhm=2.35*np.abs(sigma)
resolution=fwhm/chn
dfwhm=2.35*np.abs(dsigma)
chn=np.abs(chn)
dchn=np.abs(dchn)
def propagateerrorfwhm(chn,dchn,fwhm,dfwhm):
    return np.sqrt(((1/chn)*dfwhm)**2+((fwhm/chn**2)*dchn)**2)
print('PER Il SEGNALE DA 4.16mv HO I SEGUENTI VALORI CON INCERTEZZE:')
for i in range(len(chn)):
    print('Per il segnale con %.3f abbiamo il canale medio %.3f +- %.3f abbiamo la fwhm %.3f +- %.3f e una risoluzione energetica %.3f +- %.3f '%(cap[i],chn[i],dchn[i],fwhm[i],dfwhm[i],resolution[i],propagateerrorfwhm(chn[i],dchn[i],fwhm[i],dfwhm[i])))

fwhm=resolution
ds=propagateerrorfwhm(chn,dchn,fwhm,dfwhm)

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
pylab.close()
## trova la capacità incognita
#Calcola la fwhm
print('questi risultati sul fit gaussiano riguardano la capacità incognita')
f= 'incognitacapaicta.txt'
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
plt.close()
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
print('il fattore moltiplicativo per %s è %.3f, la media è %.2f, la sigma è %.2f' % (f.replace('.txt',''),a,x0,sig))
print('il chi2 per %s è=%.3f, i DOF sono=%.3f' % (f.replace('.txt',''),chi2_1, DOF))
print('il chi2 ridotto per %s è=%.3f '% (f.replace('.txt',''),chi2_1redux))
print('il pvalue per %s è=%.3f'% (f.replace('.txt',''),pvalue))
print('Area sotto il fotopicco per %s è : %.3f'%(f.replace('.txt',''),photopeakcount))

##plot
pylab.figure('fit gaussiano con %s'%f.replace('.txt',''))


pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('channel')
pylab.ylabel('counts')
pylab.xlim(0,500)

pylab.title('gauss fit con %s'%f.replace('.txt',''))
pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
pylab.grid()


pylab.show()
pylab.close()
plt.savefig('Fit_Gaussiano_con_%s.png'%f.replace('.txt',''))

##calcolo capacità incognita


fwhminc=2.35*np.abs(sig)
fwhminc=fwhminc/x0
def incognitcapacity(x):
    return (-p2[1]+np.sqrt(p2[1]**2+4*p2[0]*(x-p2[2])))/(2*p2[0])
capacity=incognitcapacity(fwhminc)
dcapacity=dsig*2.35/fwhminc * capacity
print('LA CAPACITà INCOGNITA è : %.3f +- %.3f'%(capacity,dcapacity))



## questa parte serve per la parte di FWHM e Xmedio Vs TENSIONE  per AMERICIO
xmedio=[]
dxmedio=[]
sigma=[]
photopeaks=[]
dsigma=[]
dxmedio=[]
tension=np.array([4,9,11,12])
filenames = glob.glob('??V.txt')

for f in filenames:
    print(f)

    y = np.loadtxt(fname=f, delimiter=',',unpack=True)
    x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali

    plt.figure('Visuale a %s'%f.replace('.txt','')) #plot per vedere i dati
    plt.plot(x, y, color='blue',marker = 'o')
    plt.xlabel('chn')
    plt.title('Americio con %s'%f.replace('.txt',''))
    plt.ylabel('count')
    plt.xlim(0,250)
    plt.grid(True)
    plt.show()
    plt.savefig('Americio_con_%s.png'%f.replace('.txt',''))
    plt.close()


##fit gaussiano
#la tecnica è la seguente: dal grafico precedente isolo ad occhio il fotopicco e vedo quali sono i dati che non sono nel fotopicco: dall'asse x vedo quali corrispondono e metto quegli elementi del vettore a zero, e poi faccio la stessa cosa agli elementi con gli stessi indici del vettore ordinata. Poi con una mask elimino quegli elementi
    data=y
    a=180
    b=240
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
    print('il fattore moltiplicativo per %s è %.3f, la media è %.2f, la sigma è %.2f' % (f.replace('.txt',''),a,x0,sig))
    print('il chi2 per %s è=%.3f, i DOF sono=%.3f' % (f.replace('.txt',''),chi2_1, DOF))
    print('il chi2 ridotto per %s è=%.3f '% (f.replace('.txt',''),chi2_1redux))
    print('il pvalue per %s è=%.3f'% (f.replace('.txt',''),pvalue))
    print('Area sotto il fotopicco per %s è : %.3f'%(f.replace('.txt',''),photopeakcount))

##plot
    pylab.figure('fit gaussiano con %s'%f.replace('.txt',''))


    pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

    pylab.xlabel('channel')
    pylab.ylabel('counts')
    pylab.xlim(0,500)

    pylab.title('gauss fit con %s'%f.replace('.txt',''))
    pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
    pylab.grid()


    pylab.show()
    plt.savefig('Fit_Gaussiano_con_%s.png'%f.replace('.txt',''))
    #plt.close()
    xmedio.append(x0)
    sigma.append(sig)
    photopeaks.append(photopeakcount)
    dxmedio.append(dx0)
    dsigma.append(dsig)


fwhm=2.35*np.abs(sigma)
resolution=fwhm/xmedio
xmedio=np.abs(xmedio)
dxmedio=np.abs(dxmedio)
dfwhm=2.35*np.abs(dsigma)

def propagateerrorfwhm(chn,dchn,fwhm,dfwhm):
    return np.sqrt(((1/chn)*dfwhm)**2+((fwhm/chn**2)*dchn)**2)
print('PER I SEGNALI con AMERICIO A VARIE TENSIONI I SEGUENTI VALORI CON INCERTEZZE:')
for i in range(len(chn)):
    print('Per il segnale %s abbiamo il canale medio %.3f +- %.3f abbiamo la fwhm %.3f +- %.3f e una risoluzione energetica %.3f +- %.3f '%(filenames[i],xmedio[i],dxmedio[i],fwhm[i],dfwhm[i],resolution[i],propagateerrorfwhm(xmedio[i],dxmedio[i],fwhm[i],dfwhm[i])))




## Questa parte serve per il fit di FWHM vs Tensione e Xmedio vs Tensione

##Xmedio
ds=resolution
y1=np.linspace(2,22,1000)
p1=np.polyfit(tension,xmedio,1)
yfit1=p1[0]*tension+p1[1]
yres1=xmedio-yfit1
SSresid1=sum(pow(yres1,2))
SStotal1=len(xmedio)*np.sqrt(np.var(xmedio)**2 +(max(ds))**2)
rsq1=1-SSresid1/SStotal1
print('Per il fit lineare di Xmedio vs Tensione il coefficiente del primo grado è %.3f, del termine costante è %.3f, R value è %.3f' % (p1[0],p1[1],rsq1))


p2=np.polyfit(tension,xmedio,2)
yfit2=p2[0]*tension**2+p2[1]*tension +p2[2]
yres2=xmedio-yfit2
SSresid2=sum(pow(yres2,2))
SStotal2=len(xmedio)*np.var(xmedio)
rsq2=1-SSresid2/SStotal2
print('Per il fit quadratico di Xmedio vs Tensione il coefficiente del secondo grado è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p2[0],p2[1],p2[2],rsq2))

p3=np.polyfit(tension,xmedio,3)
yfit3=p3[0]*tension**3+p3[1]*tension**2 + p3[2]*tension +p3[3]
yres3=xmedio-yfit3
SSresid3=sum(pow(yres3,2))
SStotal3=len(xmedio)*np.var(xmedio)
rsq3=1-SSresid3/SStotal3
print('Per il fit cubico di Xmedio vs Tensione il coefficiente del terzo grado è %.3f, del second è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p3[0],p3[1],p3[2],p3[3],rsq3))

p4=np.polyfit(tension,xmedio,4)
yfit4=p4[0]*tension**4+p4[1]*tension**3 + p4[2]*tension**2 +p4[3]*tension+p4[4]
yres4=xmedio-yfit4
SSresid4=sum(pow(yres4,2))
SStotal4=len(xmedio)*np.var(xmedio)
rsq4=1-SSresid4/SStotal4
print('Per il fit quartico di Xmedio vs Tensione il coefficiente del quarto grado è %.6f, del terzo è %.3f,del secondo è %.3f, del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p4[0],p4[1],p4[2],p4[3],p4[4],rsq4))
#NB:il fit quartico da un rankwarning, cioè se i risultati hanno senso va bene, altrimenti stare attenti: in questo caso ho scelto il fit cubico come il migliore

pylab.figure('<x> vs tension')


pylab.errorbar( tension, xmedio, ds , fmt= '.', ecolor= 'magenta',)

pylab.xlabel('tensione [mv]')
pylab.ylabel('<x>')


pylab.title('<x> vs Tensione')
pylab.xlim(3,12)
pylab.ylim(194,203)
pylab.plot(y1, np.polyval(p1,y1),'g--',label="linear fit")
pylab.plot(y1, np.polyval(p2,y1),'b--',label="square fit")
pylab.plot(y1, np.polyval(p3,y1),'r-',label="cubic fit")
pylab.plot(y1, np.polyval(p4,y1),'m--',label="quartic fit")
pylab.legend()
pylab.grid()


pylab.show()
pylab.savefig('x_vs_Tensione.png')
pylab.close()
##FWHM


p1=np.polyfit(tension,fwhm,1)
yfit1=p1[0]*tension+p1[1]
yres1=fwhm-yfit1
SSresid1=sum(pow(yres1,2))
SStotal1=len(fwhm)*np.var(fwhm)
rsq1=1-SSresid1/SStotal1
print('Per il fit lineare di fwhm vs Tensione il coefficiente del primo grado è %.3f, del termine costante è %.3f, R value è %.3f' % (p1[0],p1[1],rsq1))


p2=np.polyfit(tension,fwhm,2)
yfit2=p2[0]*tension**2+p2[1]*tension +p2[2]
yres2=fwhm-yfit2
SSresid2=sum(pow(yres2,2))
SStotal2=len(fwhm)*np.var(fwhm)
rsq2=1-SSresid2/SStotal2
print('Per il fit quadratico di fwhm vs Tensione il coefficiente del secondo grado è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p2[0],p2[1],p2[2],rsq2))

p3=np.polyfit(tension,fwhm,3)
yfit3=p3[0]*tension**3+p3[1]*tension**2 + p3[2]*tension +p3[3]
yres3=fwhm-yfit3
SSresid3=sum(pow(yres3,2))
SStotal3=len(fwhm)*np.var(fwhm)
rsq3=1-SSresid3/SStotal3
print('Per il fit cubico di fwhm vs Tensione il coefficiente del terzo grado è %.3f, del second è %.3f,del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p3[0],p3[1],p3[2],p3[3],rsq3))

p4=np.polyfit(tension,fwhm,4)
yfit4=p4[0]*tension**4+p4[1]*tension**3 + p4[2]*tension**2 +p4[3]*tension+p4[4]
yres4=fwhm-yfit4
SSresid4=sum(pow(yres4,2))
SStotal4=len(fwhm)*np.var(fwhm)
rsq4=1-SSresid4/SStotal4
print('Per il fit quartico di fwhm vs Tensione il coefficiente del quarto grado è %.6f, del terzo è %.3f,del secondo è %.3f, del primo è %.3f, del termine costante è %.3f, R value è %.3f' % (p4[0],p4[1],p4[2],p4[3],p4[4],rsq4))
#NB:il fit quartico da un rankwarning, cioè se i risultati hanno senso va bene, altrimenti stare attenti: in questo caso ho scelto il fit cubico come il migliore

pylab.figure('fwhm vs tension')


pylab.errorbar( tension, fwhm, ds , fmt= '.', ecolor= 'magenta',)

pylab.xlabel('tensione [mv]')
pylab.ylabel('FWHM')


pylab.title('FWHM vs Tensione')
pylab.xlim(3,13)
pylab.ylim(10,30)
pylab.plot(y1, np.polyval(p1,y1),'g--',label="linear fit")
pylab.plot(y1, np.polyval(p2,y1),'b--',label="square fit")
pylab.plot(y1, np.polyval(p3,y1),'r-',label="cubic fit")
pylab.plot(y1, np.polyval(p4,y1),'m--',label="quartic fit")
pylab.legend()
pylab.grid()

pylab.savefig('FWHM_vs_Tensione.png')
pylab.show()
pylab.close()

##QUESTA PARTE E' PER LA CALIBRAZIONE ENERGETICA DEGLI ISOTOPI

filenames = glob.glob('calibrationAmericio*.txt')
xmedio=[]
sigma=[]
dxmedio=[]
dsigma=[]
for f in filenames:
    print(filenames)
    y = np.loadtxt(fname=f, delimiter=',',unpack=True)
    x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali

    plt.figure('Visuale a %s'%f.replace('.txt','')) #plot per vedere i dati
    plt.plot(x, y, color='blue',marker = 'o')
    plt.xlabel('chn')
    plt.xlim(0,250)
    plt.title('Americio con %s'%f.replace('.txt',''))
    plt.ylabel('count')
    plt.grid(True)
    plt.show()
    plt.savefig('Americio_con_%s.png'%f.replace('.txt',''))
    plt.close()
##fit gaussiano
#per qualche motivo mi da un indexerror, ma non fa alcuna differenza al fine del fit
    data=y
    a=180
    b=240
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




    popt,pcov = curve_fit(gaus,x,data,p0=[100,mean,30]) #trova i parametri ottimali (popt) e la matrice di covarianza(pcov).I parametri iniziali li ho stimati ad occhio

    DOF=n-4 #gradi di libertà
    chi2_1 = sum(((gaus(x,*popt)-data)/ds)**2) #calcolo chi quadro
    da,dx0,dsig= np.sqrt(pcov.diagonal()) #dalla diagonale della matrice di covarianza trovo le     incertezze al quadrato dei parametri ottimali
    chi2_1redux=chi2_1/DOF #chi quadro normalizzato con i gradi di libertà. deve essere più piccolo di 1 ma non eccessivamente!

    a=popt[0]#popt è un vettore con i parametri ottimali
    x0=popt[1]
    sig=popt[2]

    pvalue=1 - stats.chi2.cdf(chi2_1, DOF)#pvalue, deve essere maggiore di 0.005
    print('il fattore moltiplicativo per %s è %.3f, la media è %.2f, la sigma è %.2f' % (f.replace('.txt',''),a,x0,sig))
    print('il chi2 per %s è=%.3f, i DOF sono=%.3f' % (f.replace('.txt',''),chi2_1, DOF))
    print('il chi2 ridotto per %s è=%.3f '% (f.replace('.txt',''),chi2_1redux))
    print('il pvalue per %s è=%.3f'% (f.replace('.txt',''),pvalue))
    print('Area sotto il fotopicco per %s è : %.3f'%(f.replace('.txt',''),photopeakcount))

##plot
    pylab.figure('fit gaussiano con %s'%f.replace('.txt',''))


    pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

    pylab.xlabel('channel')
    pylab.ylabel('counts')
    pylab.xlim(150,250)

    pylab.title('gauss fit con %s'%f.replace('.txt',''))
    pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
    pylab.grid()

    pylab.savefig('gauss_fit_con_%s.png'%f.replace('.txt',''))
    pylab.show()
    plt.close()
    xmedio.append(x0)
    sigma.append(sig)
    dxmedio.append(dx0)
    dsigma.append(dsig)

##qui il fit gaussiano col cadmio
y = np.loadtxt('calibrationCadmio.txt',unpack=True)
x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali
plt.figure('Visuale a Cadmio') #plot per vedere i dati
plt.plot(x, y, color='blue',marker = 'o')
plt.xlabel('chn')
plt.title('Cadmio')
plt.ylabel('count')
plt.xlim(0,200)
plt.grid(True)
plt.show()
plt.savefig('cadmio.png')
plt.close()
##fit gaussiano
#per qualche motivo mi da un indexerror, ma non fa alcuna differenza al fine del fit
data=y
a=65
b=85
mean=75
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




popt,pcov = curve_fit(gaus,x,data,p0=[100,mean,30]) #trova i parametri ottimali (popt) e la matrice di covarianza(pcov).I parametri iniziali li ho stimati ad occhio

DOF=n-4 #gradi di libertà
chi2_1 = sum(((gaus(x,*popt)-data)/ds)**2) #calcolo chi quadro
da,dx0,dsig= np.sqrt(pcov.diagonal()) #dalla diagonale della matrice di covarianza trovo le     incertezze al quadrato dei parametri ottimali
chi2_1redux=chi2_1/DOF #chi quadro normalizzato con i gradi di libertà. deve essere più piccolo di 1 ma non eccessivamente!

a=popt[0]#popt è un vettore con i parametri ottimali
x0=popt[1]
sig=popt[2]

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)#pvalue, deve essere maggiore di 0.005
print('il fattore moltiplicativo per Cadmio è %.3f, la media è %.2f, la sigma è %.2f'%(a,x0,sig))
print('il chi2 per Cadmio è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto per Cadmio è=%.3f '% (chi2_1redux))
print('il pvalue per Cadmio è=%.3f'% (pvalue))
print('Area sotto il fotopicco per Cadmio è : %.3f'%(photopeakcount))

##plot
pylab.figure('fit gaussiano con Cadmio')


pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('channel')
pylab.ylabel('counts')
pylab.xlim(40,120)

pylab.title('gauss fit con Cadmio')
pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
pylab.grid()

pylab.savefig('fit_gaussiano_con_Cadmio')
pylab.show()
pylab.close()
xmedio.append(x0)
sigma.append(sig)
dxmedio.append(dx0)
dsigma.append(dsig)
##fit gaussiano per il picco del gadolinio
y = np.loadtxt('calibrationAmericioGDpure.txt',unpack=True)
x=np.linspace(0,2048,2048) #crea il vettore del numero dei canali


data=y
a=180
b=300
mean=220
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




popt,pcov = curve_fit(gaus,x,data,p0=[100,mean,30]) #trova i parametri ottimali (popt) e la matrice di covarianza(pcov).I parametri iniziali li ho stimati ad occhio

DOF=n-4 #gradi di libertà
chi2_1 = sum(((gaus(x,*popt)-data)/ds)**2) #calcolo chi quadro
da,dx0,dsig= np.sqrt(pcov.diagonal()) #dalla diagonale della matrice di covarianza trovo le     incertezze al quadrato dei parametri ottimali
chi2_1redux=chi2_1/DOF #chi quadro normalizzato con i gradi di libertà. deve essere più piccolo di 1 ma non eccessivamente!

a=popt[0]#popt è un vettore con i parametri ottimali
x0=popt[1]
sig=popt[2]

pvalue=1 - stats.chi2.cdf(chi2_1, DOF)#pvalue, deve essere maggiore di 0.005
print('il fattore moltiplicativo per GADOLINIO è %.3f, la media è %.2f, la sigma è %.2f'%(a,x0,sig))
print('il chi2 per GADOLINIO è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto per GADOLINIO è=%.3f '% (chi2_1redux))
print('il pvalue per GADOLINIO è=%.3f'% (pvalue))
print('Area sotto il fotopicco per GADOLINIO è : %.3f'%(photopeakcount))

##plot
pylab.figure('fit gaussiano del picco del Gadolinio')


pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

pylab.xlabel('channel')
pylab.ylabel('counts')
pylab.xlim(0,500)

pylab.title('gauss fit Americio con Gadolinio Puro ')
pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
pylab.grid()

pylab.savefig('fit_gaussiano_del_picco_del_Gadolinio.png')
pylab.show()
pylab.close()
xmedio.append(x0)
sigma.append(sig)
dxmedio.append(dx0)
dsigma.append(dsig)
##calibrazione energetica isotopi
del xmedio[1:3]
del sigma[1:3]

energy=[60,22,42]
ds=2.35*np.abs(sigma)/np.abs(xmedio)
y1=np.linspace(0,250,1000)
p1=np.polyfit(xmedio,energy,1)
yfit1=p1[0]*np.abs(xmedio)+p1[1]
yres1=energy-yfit1
SSresid1=sum(pow(yres1,2))
SStotal1=len(energy)*np.var(energy)
rsq1=1-SSresid1/SStotal1
print('Per il fit lineare per la calibrazione energetica degli isotopi il coefficiente del primo grado è %.3f, del termine costante è %.3f, R value è %.3f' % (p1[0],p1[1],rsq1))


pylab.figure('Calibrazione Energetica Isotopi')


pylab.errorbar( xmedio, energy, ds , fmt= '.', ecolor= 'magenta',)

pylab.xlabel('Canale')
pylab.ylabel('Energia [keV]')


pylab.title('Calibrazione Energetica Isotopi')

pylab.plot(y1, np.polyval(p1,y1),'g--',label="linear fit")

pylab.grid()

pylab.savefig('calibrazione_energetica_isotopi.png')
pylab.show()
pylab.close()

print('la funzione di calibrazione è ENERGIA= %.2f * CANALE + %.2f - 15'%(p1[0],p1[1]))
def enrgeticcalibration(x):
    return p1[0]*x+p1[0]-15


###CALCOLO INCERTEZZE SULL ENERGIA PER CALIBRAZIONE CON SORGENTI


print('QUESTA PARTE RIGUARDA IL CALCOLO DELLE INCERTEZZE SULL ENERGIA DELLLA CALIBRAZIONE CON I SEGNALI DELLA SORGENTE')
xmedio=np.abs(xmedio)
def calcoloincertezzaparte1(x):
    return np.sqrt((np.abs(m)*x)**2 +(x*dm)**2+dsig**2)
dE=calcoloincertezzaparte1(np.abs(dxmedio))
for i in range(len(xmedio)):
    print('per il canale %.3f incertezza sulla relativa energia è %.3f'%(xmedio[i],dE[i]))

##QUESTA PARTE RIGUARDA L'ASSORBIMENTO DEL RAME NB:automatizzato così fa un'approssimazione forzata, in questo caso è più preciso analizzare singolarmente gli spettri e cambiare l'intervallo a mano
filenames = glob.glob('Am*.txt')
xmedio=[]
sigma=[]
photopeaks=[]
for f in filenames:
    print(f)
    x=np.linspace(0,2048,2048)
    y = np.loadtxt(fname=f, delimiter=',',unpack=True)
    plt.figure('Visuale a %s'%f.replace('text.txt','')) #plot per vedere i dati
    plt.plot(x, y, color='blue',marker = 'o')
    plt.xlabel('chn')
    plt.title('Visuale a %s'%f.replace('text.txt',''))
    plt.ylabel('count')
    plt.xlim(0,250)
    plt.grid(True)
    plt.show()
    plt.savefig('visuale_abs_%s.png'%f.replace('text.txt',''))
    plt.close()
    data=y


    a=176
    b=220
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
    popt,pcov = curve_fit(gaus,x,data,p0=[100,mean,30]) #trova i parametri ottimali (popt) e la matrice di covarianza(pcov).I parametri iniziali li ho stimati ad occhio

    DOF=n-4 #gradi di libertà
    chi2_1 = sum(((gaus(x,*popt)-data)/ds)**2) #calcolo chi quadro
    da,dx0,dsig= np.sqrt(pcov.diagonal()) #dalla diagonale della matrice di covarianza trovo le     incertezze al quadrato dei parametri ottimali
    chi2_1redux=chi2_1/DOF #chi quadro normalizzato con i gradi di libertà. deve essere più piccolo di 1 ma non eccessivamente!

    a=popt[0]#popt è un vettore con i parametri ottimali
    x0=popt[1]
    sig=popt[2]


    pvalue=1 - stats.chi2.cdf(chi2_1, DOF)#pvalue, deve essere maggiore di 0.005
    print('il fattore moltiplicativo per %s spessori di Rame è %.3f, la media è %.2f, la sigma è %.2f' % (f.replace('text.txt',''),a,x0,sig))
    print('il chi2 per %s spessori di Rame è=%.3f, i DOF sono=%.3f' % ((f.replace('text.txt',''),chi2_1, DOF)))
    print('il chi2 ridotto per %s spessori di Rame è=%.3f '% ((f.replace('text.txt',''),chi2_1redux)))
    print('il pvalue per %s spessori di Rame è=%.3f'% ((f.replace('text.tx',''),pvalue)))
    print('Area sotto il fotopicco per %s spessori di Rame è : %.3f'%((f.replace('text.txt',''),photopeakcount)))
    pylab.figure('fit gaussiano con %s'%f.replace('text.txt',''))


    pylab.errorbar( x, data, ds , fmt= '.', ecolor= 'magenta')

    pylab.xlabel('channel')
    pylab.ylabel('counts')
    pylab.xlim(150,250)

    pylab.title('gauss fit con %s'%f.replace('text.txt',''))
    pylab.plot(x1,gaus(x1,*popt), color='green', label="fit")
    pylab.grid()

    pylab.savefig('gauss_fit_con_%s.png'%f.replace('.txt',''))
    pylab.show()
    plt.close()
    xmedio.append(x0)
    sigma.append(sig)

    photopeaks.append(photopeakcount)

fwhm=2.35*np.abs(sigma)
#print(len(fwhm),len(photopeaks))
sigma_Cu=np.sqrt(fwhm**2 + np.sqrt(np.abs(photopeaks)**2))

y1=np.linspace(-0,1,1000)    #genero una ascissa a caso per il fit
w=0.06 #spessore moneta rame singolo in cm
spessori=np.array([0,10*w,w,2*w,3*w,4*w,5*w,6*w,7*w,8*w,9*w]) #glob me li ordina così

def f1(x,mu,ch_0,q):

    y=ch_0*np.exp(-mu*x)+q

    return y



popt, pcov= curve_fit(f1, spessori, photopeaks, (1,10,0.),sigma_Cu,absolute_sigma=False)
DOF=len(spessori)-4
chi2_1 = sum(((f1(spessori,*popt)-photopeaks)/sigma_Cu)**2)
dmu,dch_0,dq= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

mu=popt[0]
ch_0=popt[1]
q=popt[2]




pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('il coefficiente di assorbimento del rame è %.3f pm %.3f, la costante moltiplicativa è %.3f pm %.3f e il valore costante è %.3f pm %.3f'  % (mu,dmu,ch_0,dch_0,q,dq))
print('il chi2 è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto è=%.3f '% (chi2_1redux))
print('il pvalue è=%.3f'% (pvalue))

pylab.figure('Assorbimento Rame')


pylab.errorbar( spessori, photopeaks, sigma_Cu , fmt= '.', ecolor= 'magenta')

pylab.xlabel('width [cm]')
pylab.ylabel('Photopeak Area')


pylab.title('Assorbimento Rame')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()

pylab.savefig('assorbimento_rame.png')
pylab.show()
pylab.close()
I_0=np.abs(photopeaks[0])
## QUESTA PARTE RIGUARDA LE ATTENUAZIONI DEI SINGOLI MATERIALI
chn=[]
sigma=[]
photopeaks=[]
def absorptioncoeff(x,y):
    return (1/x)*np.log(I_0/y)

filenames = glob.glob('muvarie*.txt')

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
    plt.savefig('Visuale a %s'%f.replace('.txt',''))
    #plt.close()
    data=y
    a=130
    b=500
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
    print('per i vari materiali con lamericio')
    pvalue=1 - stats.chi2.cdf(chi2_1, DOF)#pvalue, deve essere maggiore di 0.005
    print('il fattore moltiplicativo per %s è %.3f, la media è %.2f, la sigma è %.2f' % (f.replace('.txt',''),a,x0,sig))
    print('il chi2 per %s è=%.3f, i DOF sono=%.3f' % (f.replace('.txt',''),chi2_1, DOF))
    print('il chi2 ridotto per %s è=%.3f '% (f.replace('.txt',''),chi2_1redux))
    print('il pvalue per %s è=%.3f'% (f.replace('.txt',''),pvalue))
    print('Area sotto il fotopicco per %s è : %.3f'%(f.replace('.txt',''),photopeakcount))

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
    plt.close()
    chn.append(x0)
    sigma.append(sig)
    photopeaks.append(photopeakcount)

##CALCOLO VERO COEFFICIENTI DI ASSORBIMENTO
lenght=np.array([100,125,50,50])
densities=np.array([10.3,7.3,10.3,7.3])
del photopeaks[2]

photopeaks=np.abs(photopeaks)
absorptions=absorptioncoeff(lenght,photopeaks)
print('I COEFFICIENTI DI ASSORBIMENTO DEI VARI METERIALI SONO:')
print(absorptions)
print('I COEFFICIENTI DIVISI PER LA DENSITA SONO IN gr/cm^2:')
print(absorptions/densities)


##Discriminatore

filenames = glob.glob('Discriminatore*.txt')

for f in filenames:
    print(f)
    signal,counts = np.loadtxt(fname=f, unpack=True)
    dc=np.sqrt(counts)

    y1=np.linspace(0,250,1000)

    def f1(x,mu,sigma,A1,A2):

        y=A2+(A1-A2)/(1+np.exp((x-mu)/sigma))

        return y



    popt, pcov= curve_fit(f1, signal, counts, (170,100,5000,0.),dc,absolute_sigma=False)
    DOF=len(signal)-3
    chi2_1 = sum(((f1(signal,*popt)-counts)/dc)**2)
    dmu,dsigma,dA1,dA2= np.sqrt(pcov.diagonal())
    chi2_1redux=chi2_1/DOF

    mu=popt[0]
    sigma=popt[1]
    A1=popt[2]
    A2=popt[3]

    def derivative(x):
        return ((popt[2]-popt[1])/popt[1])*(1/(1+np.exp((x-popt[0])/popt[1])))

    def xmaxy():
        return popt[1]*np.log(np.abs((popt[2]-popt[3]))/(np.abs((np.max(y)/2)-popt[3])))+popt[0]


    pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
    print('la media per %s è %.3f pm %.3f, la sigma è %.3f pm %.3f , Ampiezza superiore è %.3f pm %.3f , Ampiezza2 è %.3f pm %.3f'  % ((f.replace('Discriminatore.txt',''),mu,dmu,sigma,dsigma,A1,dA1,A2,dA2)))
    print('il chi2 per %s è=%.3f, i DOF sono=%.3f' % (f.replace('Discriminatore.txt',''),chi2_1, DOF))
    print('il chi2 ridotto per %s è=%.3f '% (f.replace('Discriminatore.txt',''),chi2_1redux))
    print('il pvalue per %s è=%.3f'% (f.replace('Discriminatore.txt',''),pvalue))
    print('il valore dellcascissa a metà conteggio è %.3f'%(xmaxy()))
    print('il valore della derivata calcolata nell ascissa a metà conteggio è %.3f'%(derivative(xmaxy())))
    pylab.figure('Discriminatore con %s'%(f.replace('Discriminatore.txt','')))


    pylab.errorbar( signal  , counts, dc , fmt= '.', ecolor= 'magenta')

    pylab.xlabel('Amplitude [mV]')
    pylab.ylabel('Counts')
    pylab.xlim(100,200)

    pylab.title('Discriminatore con C=%s *10^-1 pf'%f.replace('CDiscriminatore.txt',''))
    pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
    pylab.grid()

    pylab.savefig('discriminatore_%s.png'%f.replace('Discriminatore.txt',''))
    pylab.show()
    #pylab.close()

##ULTIMO GIORNO
x=np.array([133.6,140.0,150.3,161.4,171.5,181.3,195.6,156.1,166.2])
y=np.array([50818,50658,45013,25507,10344,1453,10,40628,21081])
dc=np.sqrt(y)
popt, pcov= curve_fit(f1, x, y, (170,1721,50000,0.),dc,absolute_sigma=False)
DOF=len(signal)-3
chi2_1 = sum(((f1(x,*popt)-y)/dc)**2)
dmu,dsigma,dA1,dA2= np.sqrt(pcov.diagonal())
chi2_1redux=chi2_1/DOF

mu=popt[0]
sigma=popt[1]
A1=popt[2]
A2=popt[3]


def derivative(x):
    return ((popt[2]-popt[1])/popt[1])*(1/(1+np.exp((x-popt[0])/popt[1])))

def xmaxy():
    return popt[1]*np.log((popt[2]-popt[3])/((np.max(y)/2)-popt[3]))+popt[0]



resolution=2.35*sigma/mu
print('la risoluzione energetica per il discriminatore con segnale da 20 Kev è %.3f'%(resolution))
pvalue=1 - stats.chi2.cdf(chi2_1, DOF)
print('la media per il segnale da 20 kev col discriminatore è %.3f pm %.3f, la sigma è %.3f pm %.3f , Ampiezza superiore è %.3f pm %.3f , Ampiezza2 è %.3f pm %.3f'  % (mu,dmu,sigma,dsigma,A1,dA1,A2,dA2))
print('il chi2 per il segnale da 20 kev col discriminatore è=%.3f, i DOF sono=%.3f' % (chi2_1, DOF))
print('il chi2 ridotto per il segnale da 20 kev col discriminatore  è=%.3f '% (chi2_1redux))
print('il valore dellcascissa a metà conteggio è %.3f'%(xmaxy()))
print('il valore della derivata calcolata nell ascissa a metà conteggio è %.3f'%(derivative(xmaxy())))
pylab.figure('Discriminatore segnale da 20 KeV')


pylab.errorbar( x  , y, dc , fmt= '.', ecolor= 'magenta')

pylab.xlabel('Amplitude [mV]')
pylab.ylabel('Counts')
pylab.xlim(100,200)
pylab.errorbar( x  , y, dc , fmt= '.', ecolor= 'magenta')
pylab.title('Discriminatore segnale da 20 KeV')
pylab.plot(y1,f1(y1,*popt), color='green', label="fit")
pylab.grid()
pylab.show()
pylab.savefig('discriminatore_20_Kev.png')
### Calibrazione coi kpicchi


