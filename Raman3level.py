# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:49:38 2016

@author: dng5
"""

import numpy as np
from numpy import linalg as LA
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt

def RamanHamiltonian(k, omega, delta, epsilon):
    H = np.array([[(k+2.0)**2.0 - delta, omega/2.0, 0.0],[omega/2.0, k**2.0-epsilon,omega/2.0],[0.0,omega/2.0,(k-2.0)**2.0 + delta]])
    return H
    
def propagateHamiltonian(t, psi0, k, omega, delta, epsilon):
    H=RamanHamiltonian(k,omega,delta,epsilon)
    #print H
    Energy, V = LA.eig(H)

    V=V+1j*0.0
 #   print V[:,0]

    Vinv = np.conjugate(np.transpose(V))
   # print Vinv
    #print np.dot(Vinv[1,:],(np.dot(H,V[:,1])))
    #print np.transpose(V)
    U=np.diag(np.exp(1j*np.array(Energy)*t))

    a=np.dot(Vinv,psi0)
   # print a
    b=np.dot(U,a)
    #print b
    psi = np.dot(V,b)
    #print psi
    pop0=np.absolute(psi[0])**2.0
    pop1=np.absolute(psi[1])**2.0
    pop2=np.absolute(psi[2])**2.0
    return np.array([pop0,pop1,pop2])
    
kList = np.linspace(-2.0,2.0,300)
Energies=np.zeros((kList.size,3))
delta = 0.0
epsilon = 0.0
omega=1.0
i=0
for k in kList:
    H=RamanHamiltonian(k, omega, delta, epsilon)
    Energies[i], eigenstates = LA.eig(H)
    i=i+1

figure=plt.figure()
panel=figure.add_subplot(1,1,1)
panel.plot(kList, Energies, 'bo')
  


def plotPulsedRaman(tf, step, psi0 = [0.0,1.0,0.0], k=0.0, omega=4.0, delta=0.0, epsilon=0.0):
    tlist = np.arange(0,tf,step)
    pop0 = np.zeros(tlist.size)
    pop1 = np.zeros(tlist.size)
    pop2 = np.zeros(tlist.size)

    i=0    
    for t in tlist:
        p0,p1,p2=propagateHamiltonian(t, psi0,k,omega,delta,epsilon)
     
        pop0[i]=p0
        pop1[i]=p1
        pop2[i]=p2
        i+=1
        
    figure=plt.figure()
    pan=figure.add_subplot(1,1,1)
    pan.plot(tlist,pop0,'b-')
    pan.plot(tlist,pop1,'g-')
    pan.plot(tlist,pop2,'r-')
    return 
    