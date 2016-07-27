# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:34:49 2016

@author: dng5
"""

import numpy as np
from numpy import linalg as LA
from scipy import optimize
from scipy import linalg as sLA
from scipy import stats
import matplotlib.pyplot as plt
import readIgor

hbar = 1.0545718e-34 # reduced Planck constant m^2 kg/s
mRb =1.44467e-25 #mass of rubidium in kg
lambdaR = 790e-9 # Raman wavelength in m
lambdaL = 1.064e-6 #lattice wavelength in m
Erecoil = (2.0*np.pi*hbar)**2.0/(2.0*mRb*lambdaL**2.0) #recoil energy

def Fz(S):
    a=np.arange(np.float(-S),np.float(S+1))
    F=np.diag(a)
    return F
    
def Fx(S):
    F=np.zeros((2*S+1,2*S+1))
    for i in range(int(2*S+1)):
        for j in range(int(2*S+1)):
            if np.abs(i-j)==1:
                F[i,j]=(1.0/2.0)*np.sqrt(S*(S+1)-(i-S)*(j-S))
    return F
    
def FxFlat(S):
    F=np.zeros((2*S+1,2*S+1))
    for i in range(2*S+1):
        for j in range(2*S+1):
            if np.abs(i-j)==1:
                F[i,j]=1.0/np.sqrt(2)
    return F

def RamanLatHam(k, omega, delta, epsilon, U, n, S):
    c=4.0/3.0
    Nlat=2*n+1
    Ntot=Nlat*(2*S+1)
    Kinetic=np.zeros(Ntot)

    for i in range(Ntot):
        spinI=np.float(i%(2*S+1)-S)
        latI=np.float(np.divide(i,2*S+1)-n)
        Kinetic[i]=(k-2.0*spinI*c-2.0*latI)**2.0
    H=np.diag(Kinetic)
    H+=delta*sLA.block_diag(*[Fz(S)]*Nlat)
    H+=((-1.0)**(S+1))*epsilon*sLA.block_diag(*[Fz(S)**2.0]*Nlat)
    H+=(np.sqrt(2.0)/2.0)*omega*sLA.block_diag(*[Fx(S)]*Nlat)    
        
    for i in range(Ntot):
        for j in range(Ntot):
            if np.abs(i-j)==(2*S+1):
                H[i,j]=U/4.0         
    return H
    
def plotSynDimBandStructGen(omega, delta, epsilon, U, n,S, kList=np.linspace(-1.0,1.0,600)):
    i=0  
    s=2*S+1
    E1=np.zeros(kList.size)
    E2=np.zeros(kList.size)
    E3=np.zeros(kList.size)
    m1=np.zeros(kList.size)
    m2=np.zeros(kList.size)
    m3=np.zeros(kList.size)
    m1pops=np.zeros((kList.size,s))
    m2pops=np.zeros((kList.size,s))
    m3pops=np.zeros((kList.size,s))
    for k in kList:
        H=RamanLatHam(k, omega, delta, epsilon, U, n,S)
        Energies, eigenstates = LA.eig(H)
        sort = np.argsort(Energies)
        Esorted, eVsorted = Energies[sort], eigenstates[:,sort]
        ev1=eVsorted[:,0].reshape(2*n+1,s)
        m1[i]=np.sum(np.array([-np.float(j-S)*np.dot(ev1[:,j],ev1[:,j]) for j in range(s)]))
        m1pops[i]=np.array([np.dot(ev1[:,j],ev1[:,j]) for j in range(s)])
        ev2=eVsorted[:,1].reshape(2*n+1,s)
        m2[i]=np.sum(np.array([-np.float(j-S)*np.dot(ev2[:,j],ev2[:,j]) for j in range(s)]))
        m2pops[i]=np.array([np.dot(ev2[:,j],ev2[:,j]) for j in range(s)])
        ev3=eVsorted[:,2].reshape(2*n+1,s)
        m3[i]=np.sum(np.array([-np.float(j-S)*np.dot(ev3[:,j],ev3[:,j]) for j in range(s)]))
        m3pops[i]=np.array([np.dot(ev3[:,j],ev3[:,j]) for j in range(s)])
        E1[i]=Esorted[0]
        E2[i]=Esorted[1]
        E3[i]=Esorted[2]
        i=i+1
    figure=plt.figure()
    panel=figure.add_subplot(1,1,1)
  #  panel.set_title(r'$\Omega$ = '+str(np.round(omega,2))+r'$E_L$, $\delta$ = '+str(np.round(delta,3))+r'$E_L$, $\epsilon$ = '+str(np.round(epsilon,3))+r'$E_L$, U = '+str(np.round(U,2))+r'$E_L$')
    p1=panel.scatter(kList,E1,c=m1,vmin=-S,vmax=S, marker='_')
    panel.scatter(kList,E2,c=m2,vmin=-S,vmax=S,marker='_')
    panel.scatter(kList,E3,c=m3,vmin=-S,vmax=S,marker='_')
    panel.set_xlabel(r'$q/k_L$')
    panel.set_ylabel(r'$E/E_L$')
    plt.colorbar(p1)
    
    fig3=plt.figure()
    fig3.suptitle(r'$\Omega$ = '+str(np.round(omega,2))+r'$E_L$, $\delta$ = '+str(np.round(delta,3))+r'$E_L$, $\epsilon$ = '+str(np.round(epsilon,3))+r'$E_L$, U = '+str(np.round(U,2))+r'$E_L$')
    pan3=fig3.add_subplot(3,1,1)    
 #   pan3.set_title('Lowest band')    
    for i in range(s):    
        pan3.scatter(kList,[i for j in range(kList.size)],c=m1pops[:,i],vmin=0,vmax=1.0, cmap='Blues', marker='_',linewidths=10)
    
    pan3.set_ylabel('Synthetic lattice site')
    pan3.set_xlabel(r'Crystal momentum [$k_L$]')
    pan4=fig3.add_subplot(3,1,2)    
 #   pan4.set_title('Second band')    
    for i in range(s):    
        pan4.scatter(kList,[i for j in range(kList.size)],c=m2pops[:,i],vmin=0,vmax=1.0, cmap='Blues', marker='_',linewidths=10)
    pan4.set_ylabel('Synthetic lattice site')
    pan5=fig3.add_subplot(3,1,3)    
    pan5.set_title('Average of First and Second band')    
    for i in range(s):    
        pan5.scatter(kList,[i for j in range(kList.size)],c=(m2pops[:,i]+m1pops[:,i])/2.0,vmin=0,vmax=1.0,cmap='Blues', marker='_',linewidths=10)
    pan5.set_ylabel('Synthetic lattice site')
    pan5.set_xlabel(r'$k/k_L$')
    
    fig4=plt.figure()
    pan4=fig4.add_subplot(1,1,1)
    pan4.plot(kList,m1,'b-')
    pan4.set_xlabel(r'Crystal momentum [$k_L$]')
    pan4.set_ylabel('Magnetization')
    