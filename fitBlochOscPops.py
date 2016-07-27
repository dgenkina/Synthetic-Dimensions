# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:02:33 2016

@author: dng5
"""

import numpy as np
from numpy import linalg as LA
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt


def RamanLatHamiltonian(k, omega, delta, epsilon, U, n):
    n=np.int(n)
    if n%2==0:
        print "Number of lattice orders n must be odd!"
        
    c=1.064/0.79
    H=np.zeros((3*n,3*n))
    for i in np.arange(3*n):
        spinI=i%3-1
        latI=np.divide(i,3)-np.divide(n,2)
        for j in np.arange(3*n):
            spinJ=j%3-1
            if i==j:
                spinI=np.float(spinI)
                latI=np.float(latI)
                H[i,j]=(k-2.0*spinI*c-2.0*latI)**2.0+spinI*delta-(1-np.abs(spinI))*epsilon
            if np.abs(i-j)==3:
                H[i,j]=U/4.0
            if ((np.abs(i-j)==1) and (np.abs(spinI-spinJ)==1)):
                H[i,j]=omega/2.0
            
    return H
    
def adiabaticPops(k,omega,delta,epsilon,U,n):
    k=np.array(k)
    pop0=np.zeros(k.size)
    popM=np.zeros(k.size)
    popP=np.zeros(k.size)
    for i,kLoc in enumerate(k):
        H=RamanLatHamiltonian(kLoc, omega, delta, epsilon, U, n)
        Energies, eigenstates = LA.eig(H)
        sort = np.argsort(Energies)
        Esorted, eVsorted = Energies[sort], eigenstates[:,sort]
        ev1=eVsorted[:,0].reshape(n,3)
        popM[i]=np.dot(ev1[:,0],ev1[:,0])
        pop0[i]=np.dot(ev1[:,1],ev1[:,1])
        popP[i]=np.dot(ev1[:,2],ev1[:,2])
    return np.append(popM,np.append(pop0,popP))
    
dataFile=np.load('17May2016_files_161-193.npz')
signalGood=dataFile['signalGood']
imbal=dataFile['imbalArray']
cutoff=0.1
fieldGoodArray=((np.abs(imbal)<cutoff) & (signalGood))
fractionP=dataFile['fractionP'][fieldGoodArray]
fraction0=dataFile['fraction0'][fieldGoodArray]
fractionM=dataFile['fractionM'][fieldGoodArray]
fractions=np.append(fractionM,np.append(fraction0,fractionP))#.reshape(3,fractionP.size).transpose()
time=dataFile['tlist'][fieldGoodArray]
qlist=dataFile['qlist'][fieldGoodArray]
kLtoT=0.004015
kList=time/kLtoT


n=9
c=1.064/0.79
k=0.0
epsilon=0.048
U=5.74
omega=0.6
def constrainedAdPops(omega, epsilon,U, n):
    return lambda k, delta: np.array(adiabaticPops(k, omega, delta, epsilon, U, n))
PopsOfK=constrainedAdPops(omega,epsilon,U, n)

popt,pcov = optimize.curve_fit(PopsOfK,kList,fractions, p0=(0.03))
print popt, np.sqrt(np.diag(pcov))
kForFit=np.linspace(np.min(kList),np.max(kList),100)
pops_fitted=PopsOfK(kForFit,*popt)
#sort=np.argsort(tRecoils)
#tSorted=tRecoils[sort]
pop0 = pops_fitted.reshape(3,kForFit.size)[1]
popM = pops_fitted.reshape(3,kForFit.size)[0]
popP = pops_fitted.reshape(3,kForFit.size)[2] 

figure=plt.figure()
panel=figure.add_subplot(1,1,1)
#panel.set_title( r'$\Omega$ = ' + str(np.round(popt[0],2)) + r' $E_L/h$, $\delta$ = '+str(np.round(popt[1],3))+r' $E_L/h$, U= '+str(np.round(popt[2],3))+r'$E_L$')
panel.plot(time*1.0e3,fractionP,'bo', label=r'$m_F$=+1')
panel.plot(time*1.0e3,fraction0,'go', label=r'$m_F$=0')
panel.plot(time*1.0e3,fractionM,'ro', label=r'$m_F$=-1')
panel.plot(kForFit*kLtoT*1.0e3,pop0,'g-')
panel.plot(kForFit*kLtoT*1.0e3,popP,'b-')
panel.plot(kForFit*kLtoT*1.0e3,popM,'r-')
panel.set_xlabel('Oscillation time [ms]')
legend()