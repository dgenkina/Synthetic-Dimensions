# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 12:22:33 2016

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
    
def params(t, Vmax, omegaMax):
    latticeRampOnt=0.2*Erecoil/hbar
    ramanRampOnt=0.02*Erecoil/hbar
    if t<latticeRampOnt:
        omega=0.0
        V0=Vmax*t/latticeRampOnt
    elif t<latticeRampOnt+ramanRampOnt:
        omega=omegaMax*(t-latticeRampOnt)/ramanRampOnt
        V0=Vmax
    else:
        V0=Vmax
        omega=omegaMax
    return V0,omega
    
    