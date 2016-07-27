# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 18:44:46 2016

@author: dng5
"""

import numpy as np
from numpy import linalg as LA
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt


hbar = 1.0545718e-34 # reduced Planck constant m^2 kg/s
mRb =1.44467e-25 #mass of rubidium in kg
lambdaR = 790e-9 # Raman wavelength in m
lambdaL = 1.064e-6 #lattice wavelength in m
Erecoil = (2.0*np.pi*hbar)**2.0/(2.0*mRb*lambdaL**2.0) #recoil energy

def RamanLatHamiltonian(k, omega, delta, epsilon, U, n):
    n=np.int(n)
    if n%2==0:
        print "Number of lattice orders n must be odd!"
        
    c=4.0/3.0
    H=np.zeros((3*n,3*n))
    for i in np.arange(3*n):
        spinI=i%3-1
        latI=np.divide(i,3)-np.divide(n,2)
        for j in np.arange(3*n):
            spinJ=j%3-1
            if i==j:
                spinI=np.float(spinI)
                latI=np.float(latI)
                H[i,j]=(k-2.0*spinI*c-2.0*latI)**2.0+spinI*delta+(spinI**2.0)*epsilon
            if np.abs(i-j)==3:
                H[i,j]=U/4.0
            if ((np.abs(i-j)==1) and (np.abs(spinI-spinJ)==1)):
                H[i,j]=omega/2.0
            
    return H


def adiabaticPops(k,omega,delta,epsilon,U,n):
    H=RamanLatHamiltonian(k, omega, delta, epsilon, U, n)
    Energies, eigenstates = LA.eig(H)
    sort = np.argsort(Energies)
    Esorted, eVsorted = Energies[sort], eigenstates[:,sort]
    ev1=eVsorted[:,0].reshape(n,3)
    m1pops=np.array([np.dot(ev1[:,0],ev1[:,0]),np.dot(ev1[:,1],ev1[:,1]),np.dot(ev1[:,2],ev1[:,2])])
    return m1pops
    
def plotAdiabatiPopsOfDelta(deltaMin,deltaMax,deltaStep,k,omega,epsilon,U,n,rampOnt=0.02*Erecoil/hbar):
    dlist=np.arange(deltaMin,deltaMax,deltaStep)
    fP=np.zeros(dlist.size)
    f0=np.zeros(dlist.size)
    fM=np.zeros(dlist.size)
    fP2=np.zeros(dlist.size)
    f02=np.zeros(dlist.size)
    fM2=np.zeros(dlist.size)

    for ind,d in enumerate(dlist):
        pops=adiabaticPops(k,omega,d,epsilon,U,n)
        fP[ind]=pops[0]
        f0[ind]=pops[1]
        fM[ind]=pops[2]
        pops2=rampedOnPops(k,omega,d,epsilon,U,n,rampOnt)
        fP2[ind]=pops2[0]
        f02[ind]=pops2[1]
        fM2[ind]=pops2[2]
    figure=plt.figure()
    pan=figure.add_subplot(1,1,1)
    pan.plot(dlist,fP,'b-', label='mF=+1')
    pan.plot(dlist,f0,'g-', label='mF=0')
    pan.plot(dlist,fM,'r-', label='mF=-1')
    pan.plot(dlist,fP2,'bo')
    pan.plot(dlist,f02,'go')
    pan.plot(dlist,fM2,'ro')
    pan.set_xlabel(r'Detuning [$E_r/h$]')
    pan.set_ylabel('Fractional population')
    pan.set_title(r'$\Omega$=%.3f, U=%.3f,$\epsilon$=%.2f,k=%.2f,n=%.0f,$\tau_{Raman}$=%.3f'%(omega,U,epsilon,k,n,rampOnt*hbar/Erecoil))
    legend()
    return dlist,fP,f0,fM
    
def rampedOnPops(k,omega,delta,epsilon,U,n,rampOnt):
    tlist=np.linspace(0.0,rampOnt,80)  
    dt=tlist[1]-tlist[0]
    Energy1, V1 = LA.eig(RamanLatHamiltonian(0.0, 0.0, 0.0, 0.0, U, n))
    sort=np.argsort(Energy1)
    V1sorted=V1[:,sort]
    psi0=V1sorted[:,0]
    omegalist=omega*tlist/rampOnt
    
    
    for t in tlist:
        omegaLoc=omega*t/rampOnt
        Energy,V=LA.eig(RamanLatHamiltonian(k,omegaLoc,delta,epsilon, U, n))
        V=V+0.0j
        Vinv=np.conj(V.transpose())
        psi0=np.dot(Vinv,psi0)
        teo=np.diag(np.exp(-1.0j*Energy*dt))
        psi0=np.dot(teo,psi0)
        psi0=np.dot(V,psi0)
        
    psi0=psi0.reshape(n,3).transpose()
    pops=np.array([np.dot(psi0[0],np.conj(psi0[0])),np.dot(psi0[1],np.conj(psi0[1])),np.dot(psi0[2],np.conj(psi0[2]))])
    return pops
    
def popsForFit(freqList,resFreq,omega,U,wP,wM):
    epsilon=0.048
    n=11
    k=0.0
 #   U=6.5
 #   wP=1.0
 #   wM=1.0
    weights=np.array([wP,1.0,wM])
    deltaList=(resFreq-freqList)
    rampOnt=0.02*Erecoil/hbar
    popsW=np.zeros([deltaList.size,3])
    for ind,delta in enumerate(deltaList):
        pop=rampedOnPops(k,omega,delta,epsilon,U,n,rampOnt)
        totW=np.dot(pop,weights)
        popsW[ind]=pop*weights/totW
        
    return popsW.flatten()
    
def popsForFitPenalized(freqList,resFreq,omega,wP,wM,U):
    popsW=popsForFit(freqList,resFreq,omega,wP,wM,U)
    if ((wP<1.3) & (wP>0.7)):
        pen1=0
    else:
        pen1=1.0
    if ((wM<1.3) & (wM>0.7)):
        pen2=0
    else:
        pen2=1.0
    if ((U<8.0)&(U>4.5)):
        pen3=0.0
    else:
        pen3=1.0
    return popsW-pen1-pen2-pen3

    
#    
#filename='18Jul2016_Detuning.npz'
#datafile=np.load(filename)
#freqList=datafile['tlist'][2:14]
#freqRecoils = freqList*1.0e6*(hbar*2.0*pi)/Erecoil
#rfResGuess=0.81721*1.0e6*(hbar*2.0*pi)/Erecoil
#fractionP=datafile['fractionP'][2:14]
#fraction0=datafile['fraction0'][2:14]
#fractionM=datafile['fractionM'][2:14]
#fractions=np.array([fractionP,fraction0,fractionM]).transpose()
#
#popt,pcov = optimize.curve_fit(popsForFitPenalized,freqRecoils,fractions.flatten(), p0=(rfResGuess,0.5,6.0,1.0,1.0))
#print popt,pcov
#freqForFit=np.linspace(np.min(freqRecoils),np.max(freqRecoils),200)
#pops_fitted=popsForFit(freqForFit,*popt).reshape(freqForFit.size,3).transpose()
#pop0 =pops_fitted[0]
#pop1 = pops_fitted[1]
#pop2 = pops_fitted[2]
#
#
#
#figure=plt.figure()
#panel=figure.add_subplot(1,1,1)
#panel.set_title(r'$\Omega$ = ' + str(np.round(popt[1],2)) + ' $E_L/\hbar$, U ='+str(np.round(popt[2],3))+', resonance at '+str(np.round(popt[0]*Erecoil*1e-6/(hbar*2.0*pi),5))+' MHz, wP = ' + str(np.round(popt[3],3))+ ', wM ='+str(np.round(popt[4],3)))
#panel.plot(freqList,fractionP,'bo', label='mF=+1')
#panel.plot(freqList,fraction0,'go', label='mF=0')
#panel.plot(freqList,fractionM,'ro', label='mF=-1')
#panel.plot(freqForFit*Erecoil*1e-6/(hbar*2.0*pi),pop0,'b-')
#panel.plot(freqForFit*Erecoil*1e-6/(hbar*2.0*pi),pop1,'g-')
#panel.plot(freqForFit*Erecoil*1e-6/(hbar*2.0*pi),pop2,'r-')
#panel.set_xlabel('Raman difference frequency [MHz]')