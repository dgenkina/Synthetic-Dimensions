# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:20:22 2016

@author: dng5
"""
import numpy as np
import readIgor
import matplotlib.pyplot as plt
from scipy import optimize

def lor(x,x0,Gamma,A,offset):
    L=A*(Gamma/2.0)**2.0/((x-x0)**2.0+(Gamma/2)**2.0) +offset
    return L 
roi=np.array([400, 640, 400, 700])
roiFlea=([230,290,190,220])
roiB=np.array([roiFlea[0],roiFlea[1],2*roiFlea[2]-roiFlea[3],roiFlea[2]])
filerootFlea='X:/2016/April/28/Flea3_28Apr2016'  
filelist=np.arange(4,45)
key='oscDelay'

offset=np.zeros(filelist.size)
num1=np.zeros(filelist.size)
num2=np.zeros(filelist.size)
imbal=np.zeros(filelist.size)


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
    print waveDict
    offset[i]=waveDict['offset']
    num1[i]=np.sum(od1)-np.sum(od1B)
    num2[i]=np.sum(od2)-np.sum(od2B)
    imbal[i]=(num1[i]-num2[i])/(num1[i]+num2[i])


popt1,pcov1 = optimize.curve_fit(lor,offset,num1, p0=(0.491,0.002,1000,0.0))
print popt1, np.sqrt(np.diag(pcov1))
offsForFit=np.linspace(0.471,0.501,1500)
data_fitted1=lor(offsForFit,*popt1)
popt2,pcov2 = optimize.curve_fit(lor,offset,num2, p0=(0.481,0.002,1000,0.0))
print popt2, np.sqrt(np.diag(pcov2))
data_fitted2=lor(offsForFit,*(0.481,0.056,1200,0))

fig=plt.figure()
pan=fig.add_subplot(1,1,1)
#pan.plot(offset,num1, 'bo')
pan.plot(offsForFit,data_fitted1,'b-')
#pan.plot(offset,num2, 'go')
pan.plot(offsForFit,data_fitted2,'g-')
pan.set_xlabel('offset [A]')
pan.set_ylabel('counted number')

fig2=plt.figure()
pan2=fig2.add_subplot(1,1,1)
#pan2.plot(offset,imbal, 'bo')
pan2.plot(offsForFit,(data_fitted1-data_fitted2)/(data_fitted1+data_fitted2),'b-')
pan2.set_xlabel('offset [A]')
pan2.set_ylabel('Uwave imbalance')
pan2.set_ylim(-1.0,1.0)

