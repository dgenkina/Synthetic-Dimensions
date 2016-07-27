# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:05:03 2016

@author: dng5
"""
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def line(x,A,B):
    return A*x+B
    
def lineFit(xvars,yvars,xlabel,ylabel,errorbars=False,yerr=0.0,**kwargs):
    if errorbars:
        (A,B), cov=optimize.curve_fit(line,xvars,yvars,sigma=yerr,**kwargs)
        (dA,dB)=np.sqrt(np.diag(cov))
        xrangefit=np.linspace(np.min(xvars),np.max(xvars))
        data_fitted=line(xrangefit,*(A,B))
        
        figure=plt.figure()
        pan=figure.add_subplot(1,1,1)
        pan.errorbar(xvars,yvars,yerr=yerr,fmt='bo')
        pan.plot(xrangefit,data_fitted,'b-')
        pan.set_title('Fit params in Ax+B, A='+str(np.round(A,5))+'+/-'+str(np.round(dA,5))+', B='+str(np.round(B,3))+'+/-'+str(np.round(dB,4)))
        pan.set_xlabel(xlabel)
        pan.set_ylabel(ylabel)
    else:
        (A,B), cov=optimize.curve_fit(line,xvars,yvars,**kwargs)
        (dA,dB)=np.sqrt(np.diag(cov))
        xrangefit=np.linspace(np.min(xvars),np.max(xvars))
        data_fitted=line(xrangefit,*(A,B))
        
        figure=plt.figure()
        pan=figure.add_subplot(1,1,1)
        pan.plot(xvars,yvars,'bo')
        pan.plot(xrangefit,data_fitted,'b-')
        pan.set_title('Fit params in Ax+B, A='+str(np.round(A,3))+'+/-'+str(np.round(dA,4))+', B='+str(np.round(B,3))+'+/-'+str(np.round(dB,4)))
        pan.set_xlabel(xlabel)
        pan.set_ylabel(ylabel)
    return A,B,dA,dB
def parabola(x,a,b,c):
    return a*(x**2.0)+b*x+c
    
def lor(x,x0,A,Gamma):
    out=A/((x-x0)**2.0+(Gamma/2.0)**2.0)
    return out

def parabolicFit(xvars,yvars,xlabel,ylabel,p0=(1,1,0,0)):
    (a,b,c), cov=optimize.curve_fit(parabola,xvars,yvars)
    xrangefit=np.linspace(np.min(xvars),np.max(xvars),600)
    data_fitted=parabola(xrangefit,*(a,b,c))
    print np.sqrt(np.diag(cov))
    figure=plt.figure()
    pan=figure.add_subplot(1,1,1)
    pan.plot(xvars,yvars,'bo')
    pan.plot(xrangefit,data_fitted,'b-')
    pan.set_title('Fit params (a,b,c) in a*x^2+b*x+c='+str(np.round(a,3))+', '+str(np.round(b,3))+', '+str(np.round(c,3)))
    pan.set_xlabel(xlabel)
    pan.set_ylabel(ylabel)
    return a,b,c
    
def sine(x,A,f,phi,offset):
    return offset+A*np.sin(f*x*2.0*np.pi+phi)
    
def sineFit(xvars,yvars,xlabel,ylabel,p0=(1,1,0,0)):
    (A,f,phi,offset), cov=optimize.curve_fit(sine,xvars,yvars,p0=p0)
    xrangefit=np.linspace(np.min(xvars),np.max(xvars),600)
    data_fitted=sine(xrangefit,*(A,f,phi,offset))
    print np.sqrt(np.diag(cov))
    figure=plt.figure()
    pan=figure.add_subplot(1,1,1)
    pan.plot(xvars,yvars,'bo')
    pan.plot(xrangefit,data_fitted,'b-')
    pan.set_title('Fit params (A,f,phi,offset)='+str(np.round(A,3))+', '+str(np.round(f,3))+', '+str(np.round(phi,3))+', '+str(np.round(offset,3)))
    pan.set_xlabel(xlabel)
    pan.set_ylabel(ylabel)
    return A,f,phi,offset
    
def plane((x,y), A,B,C):
    pl=A*x+B*y+C
    return pl.ravel()    

xCent=np.array([143,163,140,165,130,134,128,136,127])
xGuess2=np.array([103,143,106,145,124,121,135,103,90])
yCent=np.array([112,144,117,146,151,139,168,117,99])
(A,B,C), cov = optimize.curve_fit(plane,(xCent,xGuess2),yCent)
#print A,B,C, np.sqrt(np.diag(cov))
    
xrot=np.array([129,139,156,106,138,144])
y=np.array([116,133,146,94,132,145])

Apd=np.array([164,-3.2,236,190,104,128,80,280,316,360,420])
Apow=np.array([2.08,0.06,2.93,2.37,1.32,1.68,1.02,3.41,3.92,4.52,5.21])
Cpd=np.array([168,0,190,76,104,136,244,284,332,380,432])
Cpow=np.array([2.02,0.1,2.27,0.99,1.3,1.65,2.91,3.45,3.96,4.51,5.20])

latPD=np.array([1.06,1.26,0.244,0.192,0.292])
latE=np.array([37.0,43.4,6.82,5.26,8.07])

latPD2=np.array([240,192,440,544])
latE2=np.array([8.78,6.85,16.45,20.34])

dataFile=np.load('18Jun2016_files_208-241.npz')
signalGood=dataFile['signalGood']
imbal=dataFile['imbalArray']

#cutoff=0.05
#fieldGoodArray=((np.abs(imbal)<cutoff) & (signalGood))
#time=dataFile['tlist'][fieldGoodArray] #time=np.linspace(0,0.005,num=fieldGoodArray.size)[fieldGoodArray]#
#fractionP=dataFile['fractionP'][fieldGoodArray]
#sineFit(time,fractionP,'hold time [s]','fraction in mF=+1',(0.1,60,0.0,0.1))
#fractionM=dataFile['fractionM'][fieldGoodArray]
#sineFit(time,fractionM,'hold time [s]','fraction in mF=-1',(0.1,60,-1.3,0.3))
#fraction0=dataFile['fraction0'][fieldGoodArray]
#sineFit(time,fraction0,'hold time [s]','fraction in mF=0',(0.1,60,-1.3,0.3))
#fig3=plt.figure()
#pan3=fig3.add_subplot(1,1,1)
#pan3.plot(time*1.0e3,fractionP,'bo', label='mF=+1')
#pan3.plot(time*1.0e3,fraction0,'go', label='mF=0')
#pan3.plot(time*1.0e3,fractionM,'ro', label='mF=-1')
#pan3.set_xlabel(r'Oscillation time [ms]')
#pan3.set_ylabel('Spin populations')
#
dataFile=np.load('14Jul2016_files_108-159.npz')
imbal=dataFile['imbalArray']
cutoff=0.15
fieldGoodArray=((imbal<cutoff) & (imbal>0.0))
time=dataFile['tlist'][fieldGoodArray]
qlist=dataFile['qlist'][fieldGoodArray]
sort=np.argsort(time)
time=time[sort]
qlist=qlist[sort]
for i in range(qlist.size-1):
    if qlist[i+1]<qlist[i]-0.8:
        qlist[i+1]=qlist[i+1]+2.0
        
lineFit(time*1.0e3,qlist,'time[ms]',r'quasimomentum [$k_L$]')

