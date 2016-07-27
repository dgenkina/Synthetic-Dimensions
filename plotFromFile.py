# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:31:29 2016

@author: dng5
"""
import numpy as np

import matplotlib.pyplot as plt

filename='14Apr2016_files_119-163.npz'
kick='pos'

dataFile=np.load(filename)
imbal=dataFile['imbalArray']
signalGood=dataFile['signalGood']
cutoff=0.35
fieldGoodArray=((imbal<cutoff) & signalGood)
fractionP=dataFile['fractionP'][fieldGoodArray]
fraction0=dataFile['fraction0'][fieldGoodArray]
fractionM=dataFile['fractionM'][fieldGoodArray]
time=dataFile['tlist'][fieldGoodArray]
qlist=dataFile['qlist'][fieldGoodArray]

sort=np.argsort(time)
time=time[sort]
qlist=qlist[sort]
for i in range(qlist.size-1):
    if kick == 'neg':        
        if qlist[i+1]<qlist[i]-0.8:
            qlist[i+1]=qlist[i+1]+2.0
    if kick == 'pos':
        if qlist[i+1]>qlist[i]+0.8:
            qlist[i+1]=qlist[i+1]-2.0
        
A, B = lineFit(time*1.0e3,qlist,'time[ms]',r'quasimomentum [$k_L$]')

qlistMod=np.mod(qlist,2.0)-1.0
kList = np.linspace(np.min(qlistMod),np.max(qlistMod),600)
    
n=9
c=1.064/0.79
omega =0.43
delta =0.018
epsilon = 0.048
U=6.8
m1pops = adiabaticPopsVect(kList+1.0,omega,delta,epsilon,U,n)
kLtoT=1.0e-3/A

fig3=plt.figure()
#fig3.suptitle(filename)
pan3=fig3.add_subplot(1,1,1)
#pan3.plot(kList ,f1*m1pops[:,0]+f2*m2pops[:,0],'b-.')
#pan3.plot(kList ,f1*m1pops[:,1]+f2*m2pops[:,1],'g-.')
#pan3.plot(kList ,f1*m1pops[:,2]+f2*m2pops[:,2],'r-.')
pan3.plot(kList ,m1pops[:,0]-m1pops[:,2],'b-')
#pan3.plot(kList ,m1pops[:,1],'g-')
#pan3.plot(kList ,m1pops[:,2],'r-')
pan3.plot(qlistMod,fractionP[sort]-fractionM[sort],'bo', label='mF=+1')
#pan3.plot(qlist,fraction0[sort],'go', label='mF=0')
#pan3.plot(qlist,fractionM[sort],'ro', label='mF=-1')
pan3.set_xlabel(r'Crystal momentum [$k_L$]')
pan3.set_ylabel('Magnetization')
pan3.set_title(r'$\tau$ = '+np.str(np.round(kLtoT*1e3,3))+' $ms/k_L$,$\Omega$ = '+str(np.round(omega,2))+r'$E_L$, $\delta$ = '+str(np.round(delta,3))+r'$E_L$, $\epsilon$ = '+str(np.round(epsilon,3))+r'$E_L$, U = '+str(np.round(U,2))+r'$E_L$')
