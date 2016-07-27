# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:08:06 2016

@author: dng5
"""

import readIgor, Generic
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import scipy.ndimage as snd
from scipy import optimize
from lineFit import lineFit


roi=np.array([390,430,225,260])#np.array([460,495,160,200])#
roiFlea=([100,135,210,245])#([205,260,365,400])

fileroot = 'Y:/Data/2016/July/22/PIXIS_22Jul2016'  
filerootFlea='Y:/Data/2016/July/22/Flea3_22Jul2016'  
filelist=np.append(np.arange(104,117),np.arange(123,154))
key='PicIntensity'


def getRoi(array,xCent,yCent,r=16,eps=5):
    ylen=array.shape[0]
    xlen=array.shape[1]
#    bbox=(xCent-r,yCent-np.int(np.sqrt(r*r-eps*eps)),xCent+r,yCent+np.int(np.sqrt(r*r-eps*eps)))
    
    y,x = np.ogrid[-yCent:ylen-yCent, -xCent:xlen-xCent]
    mask = np.sqrt((x-eps)**2.0 + y*y) +np.sqrt((x+eps)**2.0 + y*y) <= 2.0*r
    maskL = ((mask) & (x<0))
    maskR = ((mask) & (x>0))
    maskT=((mask) & (y<0))
    maskB=((mask) & (y>0))

    counts=np.sum(array[mask])
    countsL=np.sum(array[maskL])
    countsR=np.sum(array[maskR])
    countsT=np.sum(array[maskT])
    countsB=np.sum(array[maskB])

    return (counts, countsL, countsR, countsT, countsB)
    

def findBox(od,xGuess,yGuess):
    allcounts=getRoi(od, xGuess,yGuess)
    i=0
    while ((np.abs(allcounts[4]-allcounts[3])>2.0) & (i<20)):
        if (allcounts[4]-allcounts[3])>0:
            yGuess=yGuess+1
       #     print "new yCent = " +str(yCent)
            allcounts=getRoi(od, xGuess,yGuess)
        else: 
            yGuess=yGuess-1
      #      print "new yCent = " +str(yCent)
            allcounts=getRoi(od, xGuess,yGuess)
        i=i+1
#    print i
    i=0
    while ((np.abs(allcounts[2]-allcounts[1])>2.0) & (i<20)):
        if (allcounts[2]-allcounts[1])>0:
            xGuess=xGuess+1
     #       print "new xCent = " +str(xCent)blochOscFractionsV2(fileroot,filelist,roi,key,plot=True,xlabel='',checkField=True,filerootFlea=filerootFlea,roiFlea=roiFlea
            allcounts=getRoi(od, xGuess,yGuess)
        else: 
            xGuess=xGuess-1
     #       print "new xCent = " +str(xCent)a[6]['Note']
            allcounts=getRoi(od, xGuess,yGuess)
        i=i+1
    box=np.array([yGuess-2,yGuess+3,xGuess-2,xGuess+3])
    return box
    
def makeImage(fileroot,filenum,roi):
    filename=fileroot+"_"+ str(filenum).zfill(4) + ".ibw"
    dict1 =readIgor.processIBW(filename, angle=-41)
    print filename
    od=dict1['OptDepth']
    fig=plt.figure()
    pan=fig.add_subplot(1,3,1)
    pan.imshow(od,vmin=-0.15,vmax=0.5)
    odRoi=od[roi[0]:roi[1],roi[2]:roi[3]]
    pan2=fig.add_subplot(1,3,2)
    pan2.imshow(odRoi,vmin=-0.15,vmax=0.5)
    
    xGuess=(roi[3]-roi[2])/2
    yGuess=(roi[1]-roi[0])/2
    box=findBox(odRoi,xGuess,yGuess)
    pan3=fig.add_subplot(1,3,3)
    pan3.imshow(odRoi[box[0]:box[1],box[2]:box[3]],vmin=-0.15,vmax=0.5)
    return



    
def getIntensities(fileroot,filelist,roi,key):
    PicIntensity=np.zeros(filelist.size)
    Raw1Avg=np.zeros(filelist.size)
    Raw2Avg=np.zeros(filelist.size)
    odAvg=np.zeros(filelist.size)
    for ind,filenum in enumerate(filelist):
        filename=fileroot+"_"+ str(filenum).zfill(4) + ".ibw"
        dict1 =readIgor.processIBW(filename, angle=-41)
        print filename
        Raw1=dict1['Raw1'][roi[0]:roi[1],roi[2]:roi[3]]
        Raw2=dict1['Raw2'][roi[0]:roi[1],roi[2]:roi[3]]
        od=dict1['OptDepth']
        odRoi=od[roi[0]:roi[1],roi[2]:roi[3]]
        xGuess=(roi[3]-roi[2])/2
        yGuess=(roi[1]-roi[0])/2
        
        box=findBox(odRoi,xGuess,yGuess)
        
        
        Raw1Avg[ind]=np.average(Raw1[box[0]:box[1],box[2]:box[3]])
        Raw2Avg[ind]=np.average(Raw2[box[0]:box[1],box[2]:box[3]])
        odAvg[ind]=np.average(odRoi[box[0]:box[1],box[2]:box[3]])
        
        infoString=dict1["Note"]
        waveDictLocal=readIgor.getIndexedWaves(infoString)
        PicIntensity[ind]=waveDictLocal[key]
        
    return Raw1Avg,Raw2Avg,odAvg,PicIntensity

def isatOptThin(I0, nSigma0, Isat):
    I0minusIf=nSigma0*I0/(Isat+I0)
    return I0minusIf


date=fileroot.split("/")[-1].split("_")[1]
saveName=date+'_files_'+np.str(filelist[0])+'-'+str(filelist[-1])   
Raw1Avg,Raw2Avg,odAvg,PicIntensity=getIntensities(fileroot,filelist,roi,key)


(A,B,dA,dB) = lineFit((Raw2Avg-Raw1Avg),odAvg,r'$I_0$-$I_f$ [counts]',r'od=-ln($I_f$/$I_0$)')


fig=plt.figure()
fig.suptitle( r'$I_{sat}\alpha^*$ = %.2f+/-%.2f, $\sigma_0 n/\alpha^*$ = %.3f+/-%.3f'%(-1.0/A,dA/(A**2.0),B,dB),size=20 )
pan=fig.add_subplot(1,2,1)
pan.plot(PicIntensity,Raw1Avg,'bo',label=r'$I_f$')
pan.plot(PicIntensity,Raw2Avg,'go',label=r'$I_0$')
pan.plot(PicIntensity,Raw2Avg-Raw1Avg,'ro',label=r'$I_0-I_f$')
pan.set_xlabel('PicIntensity [V]')
pan.set_ylabel('Intensity [counts]')
legend(loc=2)
pan2=fig.add_subplot(1,2,2)
pan2.plot(PicIntensity,odAvg,'bo',label=r'od')
pan2.set_xlabel('PicIntensity [V]')
pan2.set_ylabel('od')


(nSigma0,Isat),cov=optimize.curve_fit(isatOptThin,Raw2Avg,Raw2Avg-Raw1Avg,p0=(8000,25000.0))
(dnSigma0,dIsat)=np.sqrt(np.diag(cov))
print nSigma0,Isat,cov
I0forFit=np.linspace(np.min(Raw2Avg),np.max(Raw2Avg),100)
data_fitted=isatOptThin(I0forFit,nSigma0,Isat)

fig3=plt.figure()

pan3=fig3.add_subplot(1,1,1)
pan3.plot(Raw2Avg,Raw2Avg-Raw1Avg,'bo')
pan3.plot(I0forFit,data_fitted,'b-')
pan3.set_xlabel(r'$I_0$')
pan3.set_ylabel(r'$I_0-I_f$')
pan3.set_title(saveName +'\n'+r'$\sigma_0 n$=%.3f+/-%.3f, $I_{sat}$=%.0f +/- %.0f'%(nSigma0,dnSigma0,Isat,dIsat))