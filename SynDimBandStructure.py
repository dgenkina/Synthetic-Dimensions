# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:16:11 2016

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
        
    c=0.0#4.0/3.0
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
def adiabatiPopsOfDelta(deltaMin,deltaMax,deltaStep,k,omega,epsilon,U,n):
    dlist=np.arange(deltaMin,deltaMax,deltaStep)
    fP=np.zeros(dlist.size)
    f0=np.zeros(dlist.size)
    fM=np.zeros(dlist.size)
    for ind,d in enumerate(dlist):
        pops=adiabaticPops(k,omega,d,epsilon,U,n)
        fP[ind]=pops[0]
        f0[ind]=pops[1]
        fM[ind]=pops[2]
    figure=plt.figure()
    pan=figure.add_subplot(1,1,1)
    pan.plot(dlist,fP,'b-', label='mF=+1')
    pan.plot(dlist,f0,'g-', label='mF=0')
    pan.plot(dlist,fM,'r-', label='mF=-1')
    pan.set_xlabel(r'Detuning [$E_r/h$]')
    pan.set_ylabel('Fractional population')
    pan.set_title(r'$\Omega$=%.3f, U=%.3f,$\epsilon$=%.2f,k=%.2f,n=%.0f'%(omega,U,epsilon,k,n))
    legend()
    return dlist,fP,f0,fM
    
    
    
def adiabaticPopsVect(kList,omega,delta,epsilon,U,n,band):  
    Energies=np.zeros((kList.size,3*n))
    E1=np.zeros(kList.size)
    E2=np.zeros(kList.size)
    E3=np.zeros(kList.size)
    m1=np.zeros(kList.size)
    m2=np.zeros(kList.size)
    m3=np.zeros(kList.size)
    m1pops=np.zeros((kList.size,3))
    m2pops=np.zeros((kList.size,3))
    m3pops=np.zeros((kList.size,3))
    i=0
    for k in kList:
        H=RamanLatHamiltonian(k, omega, delta, epsilon, U, n)
        Energies, eigenstates = LA.eig(H)
        sort = np.argsort(Energies)
        Esorted, eVsorted = Energies[sort], eigenstates[:,sort]
        ev1=eVsorted[:,band].reshape(n,3)
        m1[i]=np.dot(ev1[:,0],ev1[:,0])-np.dot(ev1[:,2],ev1[:,2])
        m1pops[i]=np.array([np.dot(ev1[:,0],ev1[:,0]),np.dot(ev1[:,1],ev1[:,1]),np.dot(ev1[:,2],ev1[:,2])])
        ev2=eVsorted[:,1].reshape(n,3)
        m2[i]=np.dot(ev2[:,0],ev2[:,0])-np.dot(ev2[:,2],ev2[:,2])
        m2pops[i]=np.array([np.dot(ev2[:,0],ev2[:,0]),np.dot(ev2[:,1],ev2[:,1]),np.dot(ev2[:,2],ev2[:,2])])
        ev3=eVsorted[:,2].reshape(n,3)
        m3[i]=np.dot(ev3[:,0],ev3[:,0])-np.dot(ev3[:,2],ev3[:,2])
        m3pops[i]=np.array([np.dot(ev3[:,0],ev3[:,0]),np.dot(ev3[:,1],ev1[:,1]),np.dot(ev3[:,2],ev1[:,2])])
        E1[i]=Esorted[0]
        E2[i]=Esorted[1]
        E3[i]=Esorted[2]
        i=i+1
    return m1pops
#H=RamanLatHamiltonian(1.0, omega, delta, epsilon, U, n)
#Energies, eigenstates = LA.eig(H)
#sort = np.argsort(Energies)
#Esorted, eVsorted = Energies[sort], eigenstates[:,sort]
#ev1=eVsorted[:,0].reshape(n,3)
#print np.dot(ev1[:,0],ev1[:,0])
#print np.dot(ev1[:,1],ev1[:,1])
#
#print np.dot(ev1[:,2],ev1[:,2])


def plotSynDimBandStructure(omega, delta, epsilon, U, n, kList=np.linspace(-1.0,1.0,600)):
    i=0    
    E1=np.zeros(kList.size)
    E2=np.zeros(kList.size)
    E3=np.zeros(kList.size)
    m1=np.zeros(kList.size)
    m2=np.zeros(kList.size)
    m3=np.zeros(kList.size)
    for k in kList:
        H=RamanLatHamiltonian(k, omega, delta, epsilon, U, n)
        Energies, eigenstates = LA.eig(H)
        sort = np.argsort(Energies)
        Esorted, eVsorted = Energies[sort], eigenstates[:,sort]
        ev1=eVsorted[:,0].reshape(n,3)
        m1[i]=np.dot(ev1[:,0],ev1[:,0])-np.dot(ev1[:,2],ev1[:,2])
       
        ev2=eVsorted[:,1].reshape(n,3)
        m2[i]=np.dot(ev2[:,0],ev2[:,0])-np.dot(ev2[:,2],ev2[:,2])
        
        ev3=eVsorted[:,2].reshape(n,3)
        m3[i]=np.dot(ev3[:,0],ev3[:,0])-np.dot(ev3[:,2],ev3[:,2])
        
        E1[i]=Esorted[0]
        E2[i]=Esorted[1]
        E3[i]=Esorted[2]
        i=i+1
    figure=plt.figure()
    panel=figure.add_subplot(1,1,1)
    panel.set_title(r'$\Omega$ = '+str(np.round(omega,2))+r'$E_L$, $\delta$ = '+str(np.round(delta,3))+r'$E_L$, $\epsilon$ = '+str(np.round(epsilon,3))+r'$E_L$, U = '+str(np.round(U,2))+r'$E_L$')
    p1=panel.scatter(kList,E1,c=m1,vmin=-1.0,vmax=1.0, marker='_')
    panel.scatter(kList,E2,c=m2,vmin=-1.0,vmax=1.0,marker='_')
    panel.scatter(kList,E3,c=m3,vmin=-1.0,vmax=1.0,marker='_')
    panel.set_xlabel(r'$k/k_L$')
    plt.colorbar(p1)
#
#fig2=plt.figure()
#pan2=fig2.add_subplot(1,1,1)
#pan2.plot(kList,m1,'bo')
#pan2.set_xlabel(r'$k/k_L$')
##pan2.set_ylabel('Magnetization')
#    
#filename='30Jun2016_files_282-331.npz'
#kick='neg'
#f2=0.0
#f1=1.0-f2
#dataFile=np.load(filename)
#imbal=dataFile['imbalArray']
#signalGood=dataFile['signalGood']
#cutoff=0.2
#fieldGoodArray=((imbal<cutoff) & signalGood)
#fractionP=dataFile['fractionP'][fieldGoodArray]
#fraction0=dataFile['fraction0'][fieldGoodArray]
#fractionM=dataFile['fractionM'][fieldGoodArray]
#time=dataFile['tlist'][fieldGoodArray]
#qlist=dataFile['qlist'][fieldGoodArray]
#
#qlist=dataFile['qlist'][fieldGoodArray]
#sort=np.argsort(time)
#time=time[sort]
#qlist=qlist[sort]
#for i in range(qlist.size-1):
#    if kick == 'neg':        
#        while qlist[i+1]<qlist[i]-0.8:
#            qlist[i+1]=qlist[i+1]+2.0
#    if kick == 'pos':
#        while qlist[i+1]>qlist[i]+0.8:
#            qlist[i+1]=qlist[i+1]-2.0
#        
#A, B = lineFit(time[:20]*1.0e3,qlist[:20],'time[ms]',r'quasimomentum [$k_L$]')
#
#kList = np.linspace(np.min(qlist),np.max(qlist),600)
#    
n=13
c=4.0/3.0
omega =0.5
delta =0.0
epsilon = 0.048
U=6.0
kList=np.linspace(0.0,2.0,600)
m1pops = adiabaticPopsVect(kList,omega,delta,epsilon,U,n,0)
#m2pops = adiabaticPopsVect(kList,omega,delta,epsilon,U,n,1)
#kLtoT=1.0e-3/A
##uncrossed = time<kLtoT
fig3=plt.figure()
#fig3.suptitle(filename)
pan3=fig3.add_subplot(1,1,1)
##pan3.plot(kList ,f1*m1pops[:,0]+f2*m2pops[:,0],'b-.')
##pan3.plot(kList ,f1*m1pops[:,1]+f2*m2pops[:,1],'g-.')
##pan3.plot(kList ,f1*m1pops[:,2]+f2*m2pops[:,2],'r-.')
pan3.plot(kList ,m1pops[:,0],'b-')
pan3.plot(kList ,m1pops[:,1],'g-')
pan3.plot(kList ,m1pops[:,2],'r-')
#pan3.plot(qlist,fractionP[sort],'bo', label='mF=+1')
#pan3.plot(qlist,fraction0[sort],'go', label='mF=0')
#pan3.plot(qlist,fractionM[sort],'ro', label='mF=-1')
#pan3.set_xlabel(r'Quasimomentum [$k_L$]')
#pan3.set_ylabel('Spin populations')
#pan3.set_title(r'$ms/k_L$,$\Omega$ = '+str(np.round(omega,2))+r'$E_L$, $\delta$ = '+str(np.round(delta,3))+r'$E_L$, $\epsilon$ = '+str(np.round(epsilon,3))+r'$E_L$, U = '+str(np.round(U,2))+r'$E_L$')

#panel.plot(kList, E1, 'bo')
#panel.plot(kList, E2,'g-') 
#panel.plot(kList, E3, 'r.')   

