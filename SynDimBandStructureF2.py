# -*- coding: utf-8 -*-
"""
Created on Mon May 02 14:59:41 2016

@author: dng5
"""

import numpy as np
from numpy import linalg as LA
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt


def RamanLatHamiltonianF2(k, omega, delta, epsilon, U, n):
    n=np.int(n)
    if n%2==0:
        print "Number of lattice orders n must be odd!"
        
    c=1.064/0.79
    H=np.zeros((5*n,5*n))
    for i in np.arange(5*n):
        spinI=i%5-2
        latI=np.divide(i,5)-np.divide(n,2)
        for j in np.arange(5*n):
            spinJ=j%5-2
            if i==j:
                spinI=np.float(spinI)
                latI=np.float(latI)
                H[i,j]=(k-2.0*spinI*c-2.0*latI)**2.0+spinI*delta-(spinI**2.0)*epsilon
            if np.abs(i-j)==5:
                H[i,j]=U/4.0
            if ((np.abs(i-j)==1) and (np.abs(spinI+spinJ)==1)):                
                H[i,j]=np.sqrt(3.0)*omega/2.0
            if ((np.abs(i-j)==1) and (np.abs(spinI+spinJ)==3)): 
                H[i,j]=np.sqrt(2.0)*omega/2.0
                
    return H
    
n=9
c=4.0/3.0#1.064/0.79
omega =0.66
delta =0.018
epsilon = 0.027*c*c
U=5.48

def adiabaticPopsF2(k,omega,delta,epsilon,U,n):
    H=RamanLatHamiltonianF2(k, omega, delta, epsilon, U, n)
    Energies, eigenstates = LA.eig(H)
    sort = np.argsort(Energies)
    Esorted, eVsorted = Energies[sort], eigenstates[:,sort]
    ev1=eVsorted[:,0].reshape(n,5)
    m1pops=np.array([np.dot(ev1[:,i],ev1[:,i]) for i in range(5)])
    return m1pops
    
def plotSynDimBandStructure(omega, delta, epsilon, U, n, kList=np.linspace(-1.0,1.0,600)):
    i=0  
    s=5
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
        H=RamanLatHamiltonianF2(k, omega, delta, epsilon, U, n)
        Energies, eigenstates = LA.eig(H)
        sort = np.argsort(Energies)
        Esorted, eVsorted = Energies[sort], eigenstates[:,sort]
        ev1=eVsorted[:,0].reshape(n,5)
        m1[i]=np.sum(np.array([-np.float(j-2)*np.dot(ev1[:,j],ev1[:,j]) for j in range(s)]))
        m1pops[i]=np.array([np.dot(ev1[:,j],ev1[:,j]) for j in range(s)])
        ev2=eVsorted[:,1].reshape(n,5)
        m2[i]=np.sum(np.array([-np.float(j-2)*np.dot(ev2[:,j],ev2[:,j]) for j in range(s)]))
        m2pops[i]=np.array([np.dot(ev2[:,j],ev2[:,j]) for j in range(s)])
        ev3=eVsorted[:,2].reshape(n,5)
        m3[i]=np.sum(np.array([-np.float(j-2)*np.dot(ev3[:,j],ev3[:,j]) for j in range(s)]))
        m3pops[i]=np.array([np.dot(ev3[:,j],ev3[:,j]) for j in range(s)])
        E1[i]=Esorted[0]
        E2[i]=Esorted[1]
        E3[i]=Esorted[2]
        i=i+1
    figure=plt.figure()
    panel=figure.add_subplot(1,1,1)
    panel.set_title(r'$\Omega$ = '+str(np.round(omega,2))+r'$E_L$, $\delta$ = '+str(np.round(delta,3))+r'$E_L$, $\epsilon$ = '+str(np.round(epsilon,3))+r'$E_L$, U = '+str(np.round(U,2))+r'$E_L$')
    p1=panel.scatter(kList,E1,c=m1,vmin=-2.0,vmax=2.0, marker='_')
    panel.scatter(kList,E2,c=m2,vmin=-2.0,vmax=2.0,marker='_')
    panel.scatter(kList,E3,c=m3,vmin=-2.0,vmax=2.0,marker='_')
    panel.set_xlabel(r'$k/k_L$')
    plt.colorbar(p1)
    
    fig3=plt.figure()
    fig3.suptitle(r'$\Omega$ = '+str(np.round(omega,2))+r'$E_L$, $\delta$ = '+str(np.round(delta,3))+r'$E_L$, $\epsilon$ = '+str(np.round(epsilon,3))+r'$E_L$, U = '+str(np.round(U,2))+r'$E_L$')
    pan3=fig3.add_subplot(3,1,1)    
    pan3.set_title('Lowest band')    
    for i in range(s):    
        pan3.scatter(kList,[i for j in range(kList.size)],c=m1pops[:,i],vmin=0,vmax=1.0, cmap='Blues', marker='_',linewidths=10)
    
    pan3.set_ylabel('Synthetic lattice site')
    pan4=fig3.add_subplot(3,1,2)    
    pan4.set_title('Second band')    
    for i in range(s):    
        pan4.scatter(kList,[i for j in range(kList.size)],c=m2pops[:,i],vmin=0,vmax=1.0, cmap='Blues', marker='_',linewidths=10)
    pan4.set_ylabel('Synthetic lattice site')
    pan5=fig3.add_subplot(3,1,3)    
    pan5.set_title('Average of First and Second band')    
    for i in range(s):    
        pan5.scatter(kList,[i for j in range(kList.size)],c=(m2pops[:,i]+m1pops[:,i])/2.0,vmin=0,vmax=1.0,cmap='Blues', marker='_',linewidths=10)
    pan5.set_ylabel('Synthetic lattice site')
    pan5.set_xlabel(r'$k/k_L$')

    
    fig2=plt.figure()
    fig2.suptitle(r'$\Omega$ = '+str(np.round(omega,2))+r'$E_L$, $\delta$ = '+str(np.round(delta,3))+r'$E_L$, $\epsilon$ = '+str(np.round(epsilon,3))+r'$E_L$, U = '+str(np.round(U,2))+r'$E_L$')
    pan2=fig2.add_subplot(1,2,1)
    pan2.set_title('Second band')
    pan2.plot(kList,m2pops[:,0],'c-', label=r'$m_F$=+2')
    pan2.plot(kList,m2pops[:,1],'b-', label=r'$m_F$=+1')
    pan2.plot(kList,m2pops[:,2],'g-', label=r'$m_F$=0')
    pan2.plot(kList,m2pops[:,3],'r-', label=r'$m_F$=-1')
    pan2.plot(kList,m2pops[:,4],'m-', label=r'$m_F$=-2')
    pan2.set_xlabel(r'$k/k_L$')
    pan2.set_ylabel('Spin populations')
    legend()
    pan1=fig2.add_subplot(1,2,2)
    pan1.set_title('Lowest band')
    pan1.plot(kList,m1pops[:,0],'c-', label=r'$m_F$=+2')
    pan1.plot(kList,m1pops[:,1],'b-', label=r'$m_F$=+1')
    pan1.plot(kList,m1pops[:,2],'g-', label=r'$m_F$=0')
    pan1.plot(kList,m1pops[:,3],'r-', label=r'$m_F$=-1')
    pan1.plot(kList,m1pops[:,4],'m-', label=r'$m_F$=-2')
    pan1.set_xlabel(r'$k/k_L$')
    pan1.set_ylabel('Spin populations')
    legend()