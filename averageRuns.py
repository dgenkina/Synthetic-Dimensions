# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:20:49 2016

@author: dng5
"""
import numpy as np

import matplotlib.pyplot as plt
from scipy import optimize


dataFileList = np.array(['21Jul2016_files_54-102.npz','21Jul2016_files_104-153.npz','21Jul2016_files_154-203.npz',
                         '21Jul2016_files_204-253.npz','21Jul2016_files_254-303.npz'])
saveFileName='21Jul2016_Arp'
tAll=[]
fP=[]
fM=[]
f0=[]  
num1=[]
num2=[]
num3=[]
q=[]
kick='pos'
for fileName in dataFileList:
    dataFile=np.load(fileName)
    imbal=dataFile['imbalArray']
    cutoff=0.2
    fieldGoodArray=((imbal<cutoff) & (imbal>0.0))
    times=dataFile['tlist'][fieldGoodArray]
    sort=np.argsort(times)
    times=times[sort]
    qlist=dataFile['qlist'][fieldGoodArray]
    qlist=qlist[sort]
    if kick=='neg':
        for i in range(qlist.size-1):
            if qlist[i+1]<qlist[i]-0.8:
                qlist[i+1]=qlist[i+1]+2.0
                
    if kick=='pos':
        for i in range(qlist.size-1):
            if qlist[i+1]>qlist[i]+0.8:
                qlist[i+1]=qlist[i+1]+2.0
    fractionP=dataFile['fractionP'][fieldGoodArray][sort]
    fraction0=dataFile['fraction0'][fieldGoodArray][sort]
    fractionM=dataFile['fractionM'][fieldGoodArray][sort]
    num1loc=dataFile['num1'][fieldGoodArray][sort]
    num2loc=dataFile['num2'][fieldGoodArray][sort]
    num3loc=dataFile['num3'][fieldGoodArray][sort]

    for i,time in enumerate(times):
        try:
            ind = tAll.index(time)
            fP[ind].append(fractionP[i])
            f0[ind].append(fraction0[i])
            fM[ind].append(fractionM[i])
            q[ind].append(qlist[i])
            num1[ind].append(num1loc[i])
            num2[ind].append(num2loc[i])
            num3[ind].append(num3loc[i])
        except ValueError:
            tAll.append(time)
            fP.append([fractionP[i]])
            f0.append([fraction0[i]])
            fM.append([fractionM[i]])  
            q.append([qlist[i]])
            num1.append([num1loc[i]])
            num2.append([num2loc[i]])
            num3.append([num3loc[i]])
fractionP=np.array([np.average(fP[i]) for i in range(len(fP))]) 
sigmaP= np.array([np.std(fP[i]) for i in range(len(fP))]) 
fraction0=np.array([np.average(f0[i]) for i in range(len(f0))]) 
sigma0=np.array([np.std(f0[i]) for i in range(len(f0))]) 
fractionM=np.array([np.average(fM[i]) for i in range(len(fM))])  
sigmaM=np.array([np.std(fM[i]) for i in range(len(fM))])
qlist=np.array([np.average(q[i]) for i in range(len(q))])  
sigmaQ=np.array([np.std(q[i]) for i in range(len(q))])
time=np.array(tAll)   
numPoints=np.array([len(q[i]) for i in range(len(q))])
num1Tot=np.array([np.average(num1[i]) for i in range(len(num1))])  
sigma1Tot=np.array([np.std(num1[i]) for i in range(len(num1))])
num2Tot=np.array([np.average(num2[i]) for i in range(len(num2))])  
sigma2Tot=np.array([np.std(num2[i]) for i in range(len(num2))])
num3Tot=np.array([np.average(num3[i]) for i in range(len(num3))])  
sigma3Tot=np.array([np.std(num3[i]) for i in range(len(num3))])
np.savez(saveFileName,fractionP=fractionP,fraction0=fraction0,fractionM=fractionM,
         qlist=qlist,sigmaQ=sigmaQ,tlist=time,sigmaP=sigmaP,sigma0=sigma0,sigmaM=sigmaM,numPoints=numPoints,
         num1Tot=num1Tot,num2Tot=num2Tot,num3Tot=num3Tot, sigma1Tot=sigma1Tot,
         sigma2Tot=sigma2Tot,sigma3Tot=sigma3Tot)


(A,x0,Gamma), cov=optimize.curve_fit(lor,time,fraction0)
xrangefit=np.linspace(np.min(time),np.max(time),600)
data_fitted=lor(xrangefit,*(A,x0,Gamma))
    

fig1=plt.figure()
pan1=fig1.add_subplot(1,1,1)
pan1.set_title(r'x0='+str(np.round(x0,6))+r', $\Gamma$='+str(np.round(Gamma,3))) 
pan1.errorbar(time*1e3,fractionP,yerr=sigmaP, fmt='bo', label=r'$m_F$=+1')
pan1.errorbar(time*1e3,fraction0,yerr=sigma0, fmt='go', label=r'$m_F$=0')
pan1.errorbar(time*1e3,fractionM,yerr=sigmaM, fmt='ro', label=r'$m_F$=-1')
pan1.plot(xrangefit*1e3,data_fitted,'g-')
pan1.set_xlabel('Arp length [A]')
pan1.set_ylabel('Fractional populations')


fig2=plt.figure()
pan2=fig2.add_subplot(1,1,1)
pan2.errorbar(time*1e3,qlist,yerr=sigmaQ, fmt='bo')
pan2.set_xlabel('Difference frequency [kHz]')
pan2.set_ylabel(r'Quasimomentum [$k_L$]')
stdGood = numPoints>1
#lineFit(time[stdGood]*1e3,qlist[stdGood],'Oscillation time [ms]',r'Quasimomentum [$k_L$]',errorbars=True,yerr=sigmaQ[stdGood],absolute_sigma=True,maxfev=10000)

fig3=plt.figure()
pan3=fig3.add_subplot(1,1,1)
pan3.errorbar(time*1e3, num1Tot,yerr=sigma1Tot,fmt= 'bo')
pan3.errorbar(time*1e3, num2Tot,yerr=sigma2Tot,fmt= 'go')
pan3.errorbar(time*1e3, num3Tot,yerr=sigma3Tot,fmt= 'ro')
pan3.set_xlabel('Arp length [A]')
pan3.set_ylabel('Total counted atom number')
#dataFile1=np.load('14Apr2016_files_254-298.npz')
#
#time1=dataFile1['tlist']
#sort1=np.argsort(time1)
#time1=time1[sort1]
#
#fractionP1=dataFile1['fractionP'][sort1]
#fraction01=dataFile1['fraction0'][sort1]
#fractionM1=dataFile1['fractionM'][sort1]
#qlist1=dataFile1['qlist'][sort1]
#for i in range(qlist1.size-1):
#    if qlist1[i+1]>qlist1[i]+0.5:
#        qlist1[i+1]=qlist1[i+1]-2.0
#        
#dataFile2=np.load('14Apr2016_files_299-343.npz')
#
#time2=dataFile2['tlist']
#sort2=np.argsort(time2)
#time2=time2[sort2]
#
#fractionP2=dataFile2['fractionP'][sort2]
#fraction02=dataFile2['fraction0'][sort2]
#fractionM2=dataFile2['fractionM'][sort2]
#qlist2=dataFile2['qlist'][sort2]
#for i in range(qlist2.size-1):
#    if qlist2[i+1]>qlist2[i]+0.5:
#        qlist2[i+1]=qlist2[i+1]-2.0
#        
#        
#dataFile3=np.load('14Apr2016_files_344-388.npz')
#
#time3=dataFile3['tlist']
#sort3=np.argsort(time3)
#time3=time3[sort3]
#
#fractionP3=dataFile3['fractionP'][sort3]
#fraction03=dataFile3['fraction0'][sort3]
#fractionM3=dataFile3['fractionM'][sort3]
#qlist3=dataFile3['qlist'][sort3]
#for i in range(qlist3.size-1):
#    if qlist3[i+1]>qlist3[i]+0.5:
#        qlist3[i+1]=qlist3[i+1]-2.0
#
#dataFile4=np.load('15Apr2016_files_129-173.npz')
#
#time4=dataFile4['tlist']
#sort4=np.argsort(time4)
#time4=time4[sort4]
#
#fractionP4=dataFile4['fractionP'][sort4]
#fraction04=dataFile4['fraction0'][sort4]
#fractionM4=dataFile4['fractionM'][sort4]
#qlist4=dataFile4['qlist'][sort4]
#for i in range(qlist4.size-1):
#    if qlist4[i+1]>qlist4[i]+0.5:
#        qlist4[i+1]=qlist4[i+1]-2.0
#        
#time=time1
#fractionP=(fractionP1+fractionP2+fractionP3+fractionP4)/4.0
#fraction0=(fraction01+fraction02+fraction03+fraction04)/4.0
#fractionM=(fractionM1+fractionM2+fractionM3+fractionM4)/4.0
#qlist=(qlist1+qlist2+qlist3+qlist4)/4.0

