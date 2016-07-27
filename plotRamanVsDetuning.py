# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:38:36 2016

@author: dng5
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize 
from numpy import linalg as LA
from scipy.linalg import block_diag


hbar = 1.0545718e-34 # reduced Planck constant m^2 kg/s
mRb =1.44467e-25 #mass of rubidium in kg
lambdaR = 790e-9 # Raman wavelength in nm
lambdaL = 1064e-9 # Raman wavelength in nm
Eraman = (2.0*np.pi*hbar)**2.0/(2.0*mRb*lambdaR**2.0) #Raman recoil energy
Elattice = (2.0*np.pi*hbar)**2.0/(2.0*mRb*lambdaL**2.0) #Lattice recoil energy

def RamanHamiltonian(k, omega, delta, epsilon):
    H = np.array([[(k+2.0)**2.0 - delta, omega/2.0, 0.0],[omega/2.0, k**2.0-epsilon,omega/2.0],[0.0,omega/2.0,(k-2.0)**2.0 + delta]])
    return H
    
    

def propagateHamiltonianTestSO(t, omega, delta,k,epsilon):  
    t=np.array(t)    

    psi0 = np.array([0+1j*0.0, 1.0+1j*0.0, 0.0+1j*0.0])
    H = RamanHamiltonian(k, omega, delta ,epsilon)
    Energy, V = LA.eig(H)

    V = V + 1j*0.0
    Vinv = np.conjugate(np.transpose(V))

    # np.outer(t, Energy).flatten() creates a matrix for all t
    U = np.diag(np.exp(-1j*np.outer(t, Energy).flatten()))  

    a = np.dot(Vinv, psi0)
    # This repeats a so that the shape is consitent with U
    aa = np.outer(np.ones(t.size),a).flatten()
                      
    # Have to add the transpose to make shapes match 
    b = np.dot(U, aa)                                     
    # Same block diagonal trick for eigenvector matrix
    VV = block_diag(*([V]*t.size))                          
    psi = np.dot(VV, b)
    
    pops=np.absolute(psi)**2.0                     
    # Since you want the first value, need to take every 3rd row 
    # and extract the values you want from the diagonal
    return pops
    
def plotPulsedRmanvsDetuning(deltaMax, step, psi0 = [0.0,1.0,0.0], k=0.0, omega=4.0, t=0.0, epsilon=0.0):
    deltaList = np.arange(-deltaMax,deltaMax,step)
    pop0 = np.zeros(deltaList.size)
    pop1 = np.zeros(deltaList.size)
    pop2 = np.zeros(deltaList.size)
    t=np.array(t)
    
    i=0    
    for delta in deltaList:
        p0,p1,p2=propagateHamiltonianTestSO(t,omega,delta,k,epsilon)
     
        pop0[i]=p0
        pop1[i]=p1
        pop2[i]=p2
        i+=1
        
    figure=plt.figure()
    pan=figure.add_subplot(1,1,1)
    pan.set_title(r'$\Omega$ = ' + str(omega) + r' $E_r/\hbar$, pulse time = '+str(np.round(t*hbar*1e6/Eraman,3))+r'$\mu s$, $\epsilon$ = ' + str(epsilon)+ r' $E_r$')
    pan.plot(deltaList,pop0,'b-', label='mF=+1')
    pan.plot(deltaList,pop1,'g-', label='mF=0')
    pan.plot(deltaList,pop2,'r-', label='mF=-1')
    pan.set_xlabel(r'$\delta$ [$E_r$]')
    legend()
    return 
    
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
    
def propagateRLHamiltonian(t, k, omega, delta, epsilon, U, n):  
    t=np.array(t)    

    Energy1, V1 = LA.eig(RamanLatHamiltonian(0.0, 0.0, 0.0, 0.0, U, n))
    sort=np.argsort(Energy1)
    V1sorted=V1[:,sort]
    psi0=V1sorted[:,0]
   # psi0[np.divide(3*n,2)]=1.0+0.0*1j
    H = RamanLatHamiltonian(k, omega, delta ,epsilon,U,n)
    Energy, V = LA.eig(H)

    V = V + 1j*0.0
    Vinv = np.conjugate(np.transpose(V))

    # np.outer(t, Energy).flatten() creates a matrix for all t
    U = np.diag(np.exp(-1j*np.outer(t, Energy).flatten()))  

    a = np.dot(Vinv, psi0)
    # This repeats a so that the shape is consitent with U
    aa = np.outer(np.ones(t.size),a).flatten()
                      
    # Have to add the transpose to make shapes match 
    b = np.dot(U, aa)                                     
    # Same block diagonal trick for eigenvector matrix
    VV = sLA.block_diag(*([V]*t.size))                          
    psi = np.dot(VV, b)
    
    pops=np.absolute(psi)**2.0                     
    # Since you want the first value, need to take every 3rd row 
    # and extract the values you want from the diagonal
    latPops=np.sum(pops.reshape(t.size,n,3)[:,np.divide(n,2)-1:np.divide(n,2)+2,:],axis=2).flatten() 
    #populations in the -2k_L, 0, and +2k_L lattice sites, summed over spin sites,in time step blocks
    spinPops=np.sum(pops.reshape(t.size,n,3),axis=1).flatten() 
    #populations in each spin state, summed over lattice sites, in time step blocks 
    return spinPops
    
def plotPulsedRLvsDetuning(deltaMax, step, k=0.0, omega=4.0, t=0.0, epsilon=0.0, U=6.0, n=11):
    deltaList = np.arange(-deltaMax,deltaMax,step)
    pop0 = np.zeros(deltaList.size)
    pop1 = np.zeros(deltaList.size)
    pop2 = np.zeros(deltaList.size)
    t=np.array(t)
    
    i=0    
    for delta in deltaList:
        p0,p1,p2=propagateRLHamiltonian(t,k,omega,delta,epsilon,U,n)
     
        pop0[i]=p0
        pop1[i]=p1
        pop2[i]=p2
        i+=1
        
    figure=plt.figure()
    pan=figure.add_subplot(1,1,1)
    pan.set_title(r'$\Omega$ = ' + str(omega) + r' $E_L/\hbar$, pulse time = '+str(np.round(t*hbar*1e6/Elattice,3))+r'$\mu s$, $\epsilon$ = ' + str(epsilon)+ r' $E_L$'+'U = '+str(np.round(U,3))+r'$E_L$')
    pan.plot(deltaList,pop0,'b-', label='mF=+1')
    pan.plot(deltaList,pop1,'g-', label='mF=0')
    pan.plot(deltaList,pop2,'r-', label='mF=-1')
    pan.set_xlabel(r'$\delta$ [$E_L$]')
    legend()
    return 
    
def plotPulsedRamanLat(tf, step, k=0.0, omega=4.0, delta=0.0, epsilon=0.0, U=6.0,n=11):
    tlist = np.arange(0,tf,step)
    pop0 = np.zeros(tlist.size)
    pop1 = np.zeros(tlist.size)
    pop2 = np.zeros(tlist.size)

    i=0    
    for t in tlist:
        p0,p1,p2=propagateRLHamiltonian(t,k,omega,delta,epsilon,U,n)
     
        pop0[i]=p0
        pop1[i]=p1
        pop2[i]=p2
        i+=1
        
    figure=plt.figure()
    pan=figure.add_subplot(1,1,1)
    pan.set_title(r'$\Omega$ = ' + str(np.round(omega,3)) + r' $E_L/\hbar$, $\delta$ = '+str(np.round(delta,3))+r'$E_L$, $\epsilon$ = ' + str(np.round(epsilon,3))+ r' $E_L$'+'U = '+str(np.round(U,3))+r'$E_L$')
    pan.plot(tlist*1e3*hbar/Elattice,pop0,'b-', label='mF=+1')
    pan.plot(tlist*1e3*hbar/Elattice,pop1,'g-', label='mF=0')
    pan.plot(tlist*1e3*hbar/Elattice,pop2,'r-', label='mF=-1')
    pan.set_xlabel('Pulse time [ms]')
    legend()
    return 
    
def RfLatHamiltonian(k, omega, delta, epsilon, U, n):
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
                H[i,j]=(k-2.0*latI)**2.0+spinI*delta-(1-np.abs(spinI))*epsilon
            if np.abs(i-j)==3:
                H[i,j]=U/4.0
            if ((np.abs(i-j)==1) and (np.abs(spinI-spinJ)==1)):
                H[i,j]=omega/2.0
            
    return H
    
def propagateRfLatHamiltonian(t, k, omega, delta, epsilon, U, n):  
    t=np.array(t)    

    Energy1, V1 = LA.eig(RfLatHamiltonian(0.0, 0.0, 0.0, epsilon, U, n))
    sort=np.argsort(Energy1)
    V1sorted=V1[:,sort]
    psi0=V1sorted[:,0]
#    pops0=np.absolute(psi0)**2.0 
 #   print np.sum(pops0.reshape(t.size,n,3),axis=1).flatten() 
   # psi0[np.divide(3*n,2)]=1.0+0.0*1j
    H = RfLatHamiltonian(k, omega, delta ,epsilon,U,n)
    Energy, V = LA.eig(H)

    V = V + 1j*0.0
    Vinv = np.conjugate(np.transpose(V))

    # np.outer(t, Energy).flatten() creates a matrix for all t
    U = np.diag(np.exp(-1j*np.outer(t, Energy).flatten()))  

    a = np.dot(Vinv, psi0)
    # This repeats a so that the shape is consitent with U
    aa = np.outer(np.ones(t.size),a).flatten()
                      
    # Have to add the transpose to make shapes match 
    b = np.dot(U, aa)                                     
    # Same block diagonal trick for eigenvector matrix
    VV = sLA.block_diag(*([V]*t.size))                          
    psi = np.dot(VV, b)
    
    pops=np.absolute(psi)**2.0                     
    # Since you want the first value, need to take every 3rd row 
    # and extract the values you want from the diagonal
    latPops=np.sum(pops.reshape(t.size,n,3)[:,np.divide(n,2)-1:np.divide(n,2)+2,:],axis=2).flatten() 
    #populations in the -2k_L, 0, and +2k_L lattice sites, summed over spin sites,in time step blocks
    spinPops=np.sum(pops.reshape(t.size,n,3),axis=1).flatten() 
    #populations in each spin state, summed over lattice sites, in time step blocks 
    return spinPops
    
def plotPulsedRfLatvsDetuning(deltaMax, step, k=0.0, omega=4.0, t=0.0, epsilon=0.0, U=6.0, n=11):
    deltaList = np.arange(-deltaMax,deltaMax,step)
    pop0 = np.zeros(deltaList.size)
    pop1 = np.zeros(deltaList.size)
    pop2 = np.zeros(deltaList.size)
    t=np.array(t)
    
    i=0    
    for delta in deltaList:
        p0,p1,p2=propagateRfLatHamiltonian(t,k,omega,delta,epsilon,U,n)
     
        pop0[i]=p0
        pop1[i]=p1
        pop2[i]=p2
        i+=1
        
    figure=plt.figure()
    pan=figure.add_subplot(1,1,1)
    pan.set_title(r'$\Omega$ = ' + str(omega) + r' $E_L/\hbar$, pulse time = '+str(np.round(t*hbar*1e6/Elattice,3))+r'$\mu s$, $\epsilon$ = ' + str(epsilon)+ r' $E_L$'+'U = '+str(np.round(U,3))+r'$E_L$')
    pan.plot(deltaList,pop0,'b-', label='mF=+1')
    pan.plot(deltaList,pop1,'g-', label='mF=0')
    pan.plot(deltaList,pop2,'r-', label='mF=-1')
    pan.set_xlabel(r'$\delta$ [$E_L$]')
    legend()
    return 
    
  
def plotPulsedRfLat(tf, step, k=0.0, omega=4.0, delta=0.0, epsilon=0.0, U=6.0,n=11):
    tlist = np.arange(0,tf,step)
    pop0 = np.zeros(tlist.size)
    pop1 = np.zeros(tlist.size)
    pop2 = np.zeros(tlist.size)

    i=0    
    for t in tlist:
        p0,p1,p2=propagateRfLatHamiltonian(t,k,omega,delta,epsilon,U,n)
     
        pop0[i]=p0
        pop1[i]=p1
        pop2[i]=p2
        i+=1
        
    figure=plt.figure()
    pan=figure.add_subplot(1,1,1)
    pan.set_title(r'$\Omega$ = ' + str(np.round(omega,3)) + r' $E_L/\hbar$, $\delta$ = '+str(np.round(delta,3))+r'$E_L$, $\epsilon$ = ' + str(np.round(epsilon,3))+ r' $E_L$'+'U = '+str(np.round(U,3))+r'$E_L$')
    pan.plot(tlist*1e3*hbar/Elattice,pop0,'b-', label='mF=+1')
    pan.plot(tlist*1e3*hbar/Elattice,pop1,'g-', label='mF=0')
    pan.plot(tlist*1e3*hbar/Elattice,pop2,'r-', label='mF=-1')
    pan.set_xlabel('Pulse time [ms]')
    legend()
    return 