# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:55:55 2016

@author: dng5
"""

import numpy as np
import readIgor
import matplotlib.pyplot as plt
from scipy import optimize

def gauss(x,x0,A,sigma):
    return A*np.exp(-(x-x0)**2.0/(2.0*sigma**2.0))
roiFlea=([230,290,185,215])
roiB=np.array([roiFlea[0],roiFlea[1],2*roiFlea[2]-roiFlea[3],roiFlea[2]])
filerootFlea='X:/2016/April/30/Flea3_30Apr2016'  
filelist=np.arange(4,164)
key='oscDelay'


num1=np.zeros(filelist.size)
num2=np.zeros(filelist.size)
imbal=np.zeros(filelist.size)
signalGood=np.zeros(filelist.size,dtype=bool)

for i, filenum in enumerate(filelist):
    print filenum
    filenameFlea=filerootFlea+"_"+ str(filenum).zfill(4) + ".ibw"
    dictFlea =readIgor.processIBW(filenameFlea, angle=-37)
    od1=dictFlea['od1'][roiFlea[0]:roiFlea[1],roiFlea[2]:roiFlea[3]]
    od1B=dictFlea['od1'][roiB[0]:roiB[1],roiB[2]:roiB[3]]
    od2=dictFlea['od2'][roiFlea[0]:roiFlea[1],roiFlea[2]:roiFlea[3]]  
    od2B=dictFlea['od2'][roiB[0]:roiB[1],roiB[2]:roiB[3]]
    infoString=dictFlea["Note"]
    waveDict=readIgor.getIndexedWaves(infoString) 
    num1[i]=np.sum(od1)-np.sum(od1B)
    num2[i]=np.sum(od2)-np.sum(od2B)
    imbal[i]=(num1[i]-num2[i])/(num1[i]+num2[i])
    signalGood[i]=((num1[i]>0) & (num2[i]>0))
date=filerootFlea.split("/")[-1].split("_")[1]
saveName=date+'_files_'+np.str(filelist[0])+'-'+str(filelist[-1])+'uwaveImbalance'
np.savez(saveName,num1=num1,num2=num2,imbal=imbal,signalgood=signalGood)
hist=plt.hist(imbal[signalGood],bins=30,histtype='step')
yVals=hist[0]
xVals=[(hist[1][i]+hist[1][i+1])/2.0 for i in range(yVals.size)]
popt,pcov = optimize.curve_fit(gauss,xVals,yVals, p0=(0.0,6,0.15))
print popt, np.sqrt(np.diag(pcov))
xForFit=np.linspace(np.min(xVals),np.max(xVals),1500)
data_fitted=gauss(xForFit,*popt)
figure=plt.figure()
pan=figure.add_subplot(1,1,1)
pan.plot(xVals,yVals,'bo')
pan.plot(xForFit,data_fitted,'b-')
pan.set_xlabel('Uwave imbalance')
pan.set_ylabel('Occurence')
