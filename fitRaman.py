# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:42:26 2016

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
Erecoil = (2.0*np.pi*hbar)**2.0/(2.0*mRb*lambdaR**2.0) #recoil energy

def RamanHamiltonian(k, omega, delta, epsilon):
    H = np.array([[(k+2.0)**2.0 - delta, omega/2.0, 0.0],[omega/2.0, k**2.0-epsilon,omega/2.0],[0.0,omega/2.0,(k-2.0)**2.0 + delta]])
    return H
    
#def propagateHamiltonianTest(t, omega, delta, epsilon):
def propagateHamiltonianTest(t, *p0):
   # print t   
    omega, delta, epsilon = p0
    k=0.0
    psi0=np.array([0+1j*0.0,1.0+1j*0.0,0.0+1j*0.0])
    H=RamanHamiltonian(k,omega,delta,epsilon)
    #print H
    Energy, V = LA.eig(H)

    V=V+1j*0.0
 #   print V[:,0]

    Vinv = np.conjugate(np.transpose(V))
   # print Vinv
    #print np.dot(Vinv[1,:],(np.dot(H,V[:,1])))
    #print np.transpose(V)
  #  print 'test 1'
    const =- t*1j
   # print const.shape

    energyarray = []
    for idx, entry in enumerate(Energy):
        energyarray = np.append(energyarray, np.exp(entry*const))
    
#    U=np.diag(np.exp(np.array(Energy)*const))
    U = np.diag(energyarray)
   # print U
    a=np.dot(Vinv,psi0)
   # print a
    b=np.dot(U,a)
    #print b
    psi = np.dot(V,b)
    #print psi
    pop0=np.absolute(psi[0])**2.0
    pop1=np.absolute(psi[1])**2.0
    pop2=np.absolute(psi[2])**2.0
    return pop0
from scipy.linalg import block_diag

def propagateHamiltonianTestSO(t, omega, delta):  
    k = 0.0
    epsilon=0.0265
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
    
roiList=np.array([[630, 690, 550, 590], 
         [545,605, 515, 550],
         [440,515,470,520]])
roiFlea=([240,275,185,210])

filestart=84
filestop=123
fileroot = 'X:/2016/May/12/PIXIS_12May2016' 
filerootFlea='X:/2016/May/12/Flea3_12May2016'
#counts,fractions,waveDict = readIgor.batchCountMultipleROIs(fileroot,filestart,filestop,roiList,bgndLoc="left")                      
tList=waveDict['ramanDelay']
tRecoils = tList*Erecoil/hbar
fractions=np.array(fractions)
frac0=np.array(fractions[:,0])
checkField=False
if checkField:
    fileList=np.arange(filestart,filestop+1)
    imbal=np.zeros(fileList.size)
    for ind, filenum in enumerate(fileList):
        filenameFlea=filerootFlea+"_"+ str(filenum).zfill(4) + ".ibw"
        dictFlea =readIgor.processIBW(filenameFlea, angle=-37)
        od1=dictFlea['od1'][roiFlea[0]:roiFlea[1],roiFlea[2]:roiFlea[3]]
        od2=dictFlea['od2'][roiFlea[0]:roiFlea[1],roiFlea[2]:roiFlea[3]]   
        num1=np.sum(od1)
        num2=np.sum(od2)
        imbal[ind]=np.abs(num1-num2)/(num1+num2)
    cutoff=1.0
    fieldGoodArray=((imbal<cutoff) & (imbal>0.0))
#print tRecoils
#psi0=np.array([0+1j*0.0,1.0+1j*0.0,0.0+1j*0.0])
#k=0.0
#def constrainedHamiltonian(psi0,k):
#    return lambda t, omega, delta, epsilon: np.array(Raman.propagateHamiltonian(t, psi0, k, omega, delta, epsilon))
#H=constrainedHamiltonian(psi0,k)
#print H(1.0,4.0,0.0,0.0)
a=np.array(tRecoils)
b=np.array(frac0)
popt,pcov = optimize.curve_fit(propagateHamiltonianTestSO,a,fractions.flatten(), p0=(0.1,4.0))
print popt
print np.diag(np.sqrt(pcov))
tForFit=np.linspace(np.min(tRecoils),np.max(tRecoils),700)
pops_fitted=propagateHamiltonianTestSO(tForFit,*popt)
sort=np.argsort(tRecoils)
tSorted=tRecoils[sort]
pop0 = np.array([pops_fitted[i*3] for i in range(tForFit.size)])
pop1 = np.array([pops_fitted[i*3+1] for i in range(tForFit.size)]) 
pop2 = np.array([pops_fitted[i*3+2] for i in range(tForFit.size)]) 

figure=plt.figure()
panel=figure.add_subplot(1,1,1)
panel.set_title('Omega = ' + str(np.round(popt[0],3)) + ' Er/hbar, delta = '+str(np.round(popt[1],3))+' Er/hbar')
panel.plot(tRecoils*hbar*1.0e6/Erecoil,fractions[:,0],'bo', label='mF=+1')
panel.plot(tRecoils*hbar*1.0e6/Erecoil,fractions[:,1],'go', label='mF=0')
panel.plot(tRecoils*hbar*1.0e6/Erecoil,fractions[:,2],'ro', label='mF=-1')
panel.plot(tForFit*hbar*1.0e6/Erecoil,pop0,'b-')
panel.plot(tForFit*hbar*1.0e6/Erecoil,pop1,'g-')
panel.plot(tForFit*hbar*1.0e6/Erecoil,pop2,'r-')
panel.set_xlabel('Raman hold time [us]')
legend()

