# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 15:07:47 2016

@author: dng5
"""

import readIgor
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize 
from numpy import linalg as LA

hbar = 1.0545718e-34 # reduced Planck constant m^2 kg/s
mRb =1.44467e-25 #mass of rubidium in kg
lambdaR = 790e-9 # Raman wavelength in nm
lambdaL = 1064.0e-9 # Lattice wavelenght in nm
Erecoil = (2.0*np.pi*hbar)**2.0/(2.0*mRb*lambdaL**2.0) #recoil energy

def RfHamiltonian(k, omega, delta, epsilon):
    H = np.array([[(k)**2.0 - delta, omega/2.0, 0.0],[omega/2.0, k**2.0-epsilon,omega/2.0],[0.0,omega/2.0,(k)**2.0 + delta]])
    return H
    
from scipy.linalg import block_diag

def propagateRfHamiltonian(t, omega, delta):  
    k = 0.0
    epsilon=0.048
    psi0 = np.array([0+1j*0.0, 1.0+1j*0.0, 0.0+1j*0.0])
    H = RfHamiltonian(k, omega, delta ,epsilon)
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

def propagateRfHofDelta(rfFreq,resRfFreq,omega):  
    k = 0.0
    epsilon=0.0265
    t =260.0e-6*Erecoil/hbar
    delta=-(resRfFreq-rfFreq)
    psi0 = np.array([0+1j*0.0, 1.0+1j*0.0, 0.0+1j*0.0])

    pops=np.zeros([delta.size,3])
    i=0
    for delt in delta:
        H = RfHamiltonian(k, omega, delt ,epsilon)
        Energy, V = LA.eig(H)
        
        V = V + 1j*0.0
        Vinv = np.conjugate(np.transpose(V))
    
        # np.outer(t, Energy).flatten() creates a matrix for all t
        U = np.diag(np.exp(-1j*t*Energy))  
    
        a = np.dot(Vinv, psi0)
        b = np.dot(U, a)                                                            
        psi = np.dot(V, b)
    
        pops[i]=np.absolute(psi)**2.0 
        i+=1                    
    # Since you want the first value, need to take every 3rd row 
    # and extract the values you want from the diagonal

    return pops.flatten()
  
def plotPulsedRFvsDetuning(deltaMax, step, psi0 = [0.0,1.0,0.0], k=0.0, omega=4.0, t=0.0, epsilon=0.0):
    deltaList = np.arange(-deltaMax,deltaMax,step)
    pop0 = np.zeros(deltaList.size)
    pop1 = np.zeros(deltaList.size)
    pop2 = np.zeros(deltaList.size)
    t=np.array(t)
    
    i=0    
    for delta in deltaList:
        p0,p1,p2=propagateRfHamiltonian(t,omega,delta)
     
        pop0[i]=p0
        pop1[i]=p1
        pop2[i]=p2
        i+=1
        
    figure=plt.figure()
    pan=figure.add_subplot(1,1,1)
    pan.set_title(r'$\Omega$ = ' + str(omega) + r' $E_r/\hbar$, pulse time = '+str(t)+r' recoils, $\epsilon$ = ' + str(epsilon)+ r' $E_r$')
    pan.plot(deltaList,pop0,'b-', label='mF=+1')
    pan.plot(deltaList,pop1,'g-', label='mF=0')
    pan.plot(deltaList,pop2,'r-', label='mF=-1')
    pan.set_xlabel(r'\delta [$E_r/\hbar$]')
    legend()
    return 
roiList=[[505, 535, 430, 470], [500, 530, 475, 515], [495, 525, 525, 565]]

filestart=4
filestop=36
fileroot = 'X:/2016/June/29/PIXIS_29Jun2016' 
counts,fractions,waveDict = readIgor.batchCountMultipleROIs(fileroot,filestart,filestop,roiList,bgndLoc='top')                      
rfFreqList=waveDict['rfFreq']
rfFreqRecoils = rfFreqList*1.0e6*(hbar*2.0*pi)/Erecoil
fractions=np.array(fractions)
rfResGuess=0.819*1.0e6*(hbar*2.0*pi)/Erecoil
a=np.array(rfFreqRecoils)

popt,pcov = optimize.curve_fit(propagateRfHofDelta,a,fractions.flatten(), p0=(rfResGuess,0.9))
print popt,pcov
rfForFit=np.linspace(np.min(rfFreqRecoils),np.max(rfFreqRecoils),200)
pops_fitted=propagateRfHofDelta(rfForFit,*popt)
pop0 = np.array([pops_fitted[i*3] for i in range(np.int(pops_fitted.size/3))])
pop1 = np.array([pops_fitted[i*3+1] for i in range(rfForFit.size)]) 
pop2 = np.array([pops_fitted[i*3+2] for i in range(rfForFit.size)]) 

weightGuess=np.array([26.0/21.0,1.0,26.0/31.0])
countsW=weightGuess*counts
total=np.sum(countsW, axis=1)
fractionsW=np.zeros_like(fractions)
for i in range(total.size):
    fractionsW[i] = countsW[i]/total[i]

figure=plt.figure()
panel=figure.add_subplot(1,1,1)
panel.set_title('Omega = ' + str(np.round(popt[1],2)) + ' Er/hbar, resonance at '+str(np.round(popt[0]*Erecoil*1e-6/(2.0*pi*hbar),5))+' MHz')#, epsilon = ' + str(np.round(popt[2],3))+ ' Er')
panel.plot(rfFreqList,fractions[:,0],'bo', label='mF=+1')
panel.plot(rfFreqList,fractions[:,1],'go', label='mF=0')
panel.plot(rfFreqList,fractions[:,2],'ro', label='mF=-1')
panel.plot(rfForFit*Erecoil*1e-6/(2.0*pi*hbar),pop0,'b-')
panel.plot(rfForFit*Erecoil*1e-6/(2.0*pi*hbar),pop1,'g-')
panel.plot(rfForFit*Erecoil*1e-6/(2.0*pi*hbar),pop2,'r-')
panel.set_xlabel('Rf frequency [MHz]')
