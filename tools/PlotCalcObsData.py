import numpy
import re
import copy
from pyproj import Transformer
import matplotlib.pyplot as plt
import os
#===Parameters===================
obsCalcImpedanceFile="ObsCalcImpedance_0.001000_8.txt"
obsImpedanceFile="obsDataImpedance.txt"
obsCalcTipperFile="ObsCalcTipper_0.001000_8.txt"
obsTipperFile="obsDataTipperFile.txt"
outputLocationFile="ObsStationLocation.txt"
nFreqs=16
tipperPlot=True
outputFolder="./plot"
#================================
if not os.path.isdir(outputFolder):
    os.system("mkdir "+outputFolder)
nullNum=-99999
comp=1j
mu=4*numpy.pi*1e-7
class Station:
    def __init__(self, name):
        self.name=name
        self.x=nullNum
        self.y=nullNum
        self.freqs=[]
        self.ZsCalc=[]
        self.ZsObs=[]
        self.ZsVar=[]
        self.TsCalc=[]
        self.TsObs=[]
        self.TsVar=[]
        self.x_calc=nullNum
        self.y_calc=nullNum
        self.RMS_Z=0
        self.RMS_T=0
        self.nFreqs=0
    def CalcRMS_Z(self):
        sum=0
        
        numOfData=0
        for ifreq in range(self.nFreqs):
            for i in range(2):
                for j in range(2):
                    if self.ZsObs[ifreq][i,j]!=0.0:
                        sum+=(abs(self.ZsCalc[ifreq][i,j]-self.ZsObs[ifreq][i,j])/self.ZsVar[ifreq][i,j])**2.0
                        numOfData+=1
        self.RMS_Z=(sum/numOfData)**0.5
    def CalcRMS_T(self):
        sum=0
        
        numOfData=0
        for ifreq in range(self.nFreqs):
            for i in range(2):
                    if self.TsVar[ifreq][i]<1e3:
                        sum+=(abs(self.TsCalc[ifreq][i]-self.TsObs[ifreq][i])/self.TsVar[ifreq][i])**2.0
                        numOfData+=1
        self.RMS_T=(sum/numOfData)**0.5    
def RemoveParenthesis(s): #Convert (a,b) -> a+b*j

    a=float(s.split("(")[1].split(",")[0])

    b=float(s.split("(")[1].split(",")[1].split(")")[0])
    c=a+comp*b

    return c
        
def SplitLine(l,toValue=False):
    s=re.split("[\t  　]",l)
    ret=[]
    for tmp in s:
        if tmp!="":
   
            if tmp[len(tmp)-1]=="\n":
                if tmp[:-1]!="":
                    ret.append(tmp[:-1])
            else:
                ret.append(tmp)
    if toValue:
        ret2=numpy.zeros(len(ret))
        for i in range(len(ret)):
            
            ret2[i]=float(ret[i])
        return ret2
    else:    
        return ret


#========IMPEDANCE======================================= 
plotImpedance=True
if plotImpedance:
    fc=open(obsCalcImpedanceFile)
    linesCalc=fc.readlines()
    freqs=[]
    for i in range(1,nFreqs+1):
        freqs.append(SplitLine(linesCalc[i],True)[0])
        
    fo=open(obsImpedanceFile)
    linesObs=fo.readlines()
    i=0

    stations=[]
    while i<len(linesObs):
        station=Station("NULL")
        loc=SplitLine(linesObs[i])
        station.x=float(loc[0])
        station.y=float(loc[1])
        i+=1
        for j in range(nFreqs):
            z=numpy.zeros((2,2), dtype=numpy.complex128)
            zVar=numpy.zeros((2,2))
            Ztmp=SplitLine(linesObs[i+2*j],True)
            z[0,0]=Ztmp[0]+Ztmp[1]*comp
            z[0,1]=Ztmp[2]+Ztmp[3]*comp
            z[1,0]=Ztmp[4]+Ztmp[5]*comp
            z[1,1]=Ztmp[6]+Ztmp[7]*comp
            ZVarTmp=SplitLine(linesObs[i+2*j+1],True)
            zVar[0,0]=ZVarTmp[0] #Assume that error has same in real and imag
            zVar[0,1]=ZVarTmp[2]
            zVar[1,0]=ZVarTmp[4]
            zVar[1,1]=ZVarTmp[6]
            station.ZsObs.append(copy.deepcopy(z))
            station.ZsVar.append(copy.deepcopy(zVar))
        station.nFreqs=nFreqs
        stations.append(station)
        i+=2*nFreqs

    fl=open(outputLocationFile)
    linesLoc=fl.readlines()
    names={}
    for station in stations:
        dist=1e30
        for i,line in enumerate(linesLoc):
            s=SplitLine(line)
            x=float(s[1])
            y=float(s[2])
            nameTmp=s[0].split("/")[len(s[0].split("/"))-1]
            if ((x-station.x)**2+(y-station.y)**2)**0.5<dist:
                
                name=s[0].split("/")[len(s[0].split("/"))-1]
                dist=((x-station.x)**2+(y-station.y)**2)**0.5
            elif ((x-station.x)**2+(y-station.y)**2)**0.5==dist:
                if name in names:
                    name=s[0].split("/")[len(s[0].split("/"))-1]
        names[name]=1
        station.name=name

            
    i=nFreqs+1

    stations_relatedData=[]
    alreadyRelatedCalc=numpy.zeros(len(stations))
    while i<len(linesCalc):
        loc=SplitLine(linesCalc[i])
        x=float(loc[1])
        y=float(loc[2])
        dist=1e30
        istation=-1
        for j,station in enumerate(stations):    
            if ((x-station.x)**2+(y-station.y)**2)**0.5<dist:   
                
                istation=j
                dist=((x-station.x)**2+(y-station.y)**2)**0.5

        station=stations[istation]
        station.x_calc=x
        station.y_calc=y
        if alreadyRelatedCalc[istation]==0:
            alreadyRelatedCalc[istation]=1     
            stations_relatedData.append(station)
        i+=1
        for j in range(nFreqs):
            z=numpy.zeros((2,2), dtype=numpy.complex128)
            ZcalcTmp=SplitLine(linesCalc[i+2*j])

            z[0,0]=RemoveParenthesis(ZcalcTmp[0])
            z[0,1]=RemoveParenthesis(ZcalcTmp[1])
            z[1,0]=RemoveParenthesis(ZcalcTmp[2])
            z[1,1]=RemoveParenthesis(ZcalcTmp[3])
            station.ZsCalc.append(copy.deepcopy(z))
        stations_relatedData.append(copy.deepcopy(station))
        i+=2*nFreqs
    #同じセルに2つ観測データがある場合の対応
    for i,station1 in enumerate(stations):
            if alreadyRelatedCalc[i]==0:
                x1=station1.x
                y1=station1.y
                dist=1e30
                for j,station2 in enumerate(stations):

                    if alreadyRelatedCalc[j]!=0:
                        x2=station2.x
                        y2=station2.y
                        if dist>((x1-x2)**2+(y1-y2)**2)**0.5:
                                dist=((x1-x2)**2+(y1-y2)**2)**0.5
                                istation=j
                d1=((station1.x-stations[istation].x_calc)**2+(station1.y-stations[istation].y_calc)**2)**0.5
                d2=((stations[istation].x-stations[istation].x_calc)**2+(stations[istation].y-stations[istation].y_calc)**2)**0.5
                print("duplicate:",station1.name,stations[istation].name,d1,d2)            

                station1.ZsCalc=copy.deepcopy(stations[istation].ZsCalc)
                stations_relatedData.append(copy.deepcopy(station1))
    stations=stations_relatedData

    #Plot
    for station in stations:
        station.CalcRMS_Z()
        fig, axes = plt.subplots(2, 4, tight_layout=True,figsize=(14.0, 6.0))
        for i in range(2):
            for j in range(2):
                if i==0 and j==0:
                    component="XX"
                elif i==0 and j==1:
                    component="XY"
                elif i==1 and j==0:
                    component="YX"
                else:
                    component="YY"
                title=station.name+"_Impedance_"+component
                x=numpy.array(freqs)
                apprhoObs=numpy.zeros(nFreqs)
                phaseObs=numpy.zeros(nFreqs)
                apprhoCalc=numpy.zeros(nFreqs)
                phaseCalc=numpy.zeros(nFreqs)
                apprhoErr=numpy.zeros(nFreqs)
                phaseErr=numpy.zeros(nFreqs)
                for ii in range(nFreqs):
                    zobs=station.ZsObs[ii]
                    zcalc=station.ZsCalc[ii]
                    zvar=station.ZsVar[ii]
                    apprhoObs[ii]=(numpy.real(zobs[i,j])**2+numpy.imag(zobs[i,j])**2)/(2*numpy.pi*freqs[ii])/mu
                    apprhoCalc[ii]=(numpy.real(zcalc[i,j])**2+numpy.imag(zcalc[i,j])**2)/(2*numpy.pi*freqs[ii])/mu
                    phaseObs[ii]=numpy.arctan2(numpy.imag(zobs[i,j]),numpy.real(zobs[i,j]))/numpy.pi*180
                    phaseCalc[ii]=numpy.arctan2(numpy.imag(zcalc[i,j]),numpy.real(zcalc[i,j]))/numpy.pi*180
                    if apprhoObs[ii]==0.0:
                        phaseObs[ii]=numpy.nan
                        apprhoObs[ii]=numpy.nan
                    apprhoErr[ii]=2*(numpy.real(zvar[i,j])+numpy.imag(zvar[i,j]))/0.5*(numpy.real(zobs[i,j])**2+numpy.imag(zobs[i,j])**2)**0.5/(2*numpy.pi*freqs[ii])/mu
                    if abs(numpy.real(zvar[i,j])+numpy.imag(zvar[i,j]))/0.5>(numpy.real(zobs[i,j])**2+numpy.imag(zobs[i,j])**2)**0.5:
                        phaseErr[ii]=90
                    else:
                        phaseErr[ii]=180/numpy.pi*numpy.arctan((numpy.real(zvar[i,j])+numpy.imag(zvar[i,j]))/0.5/(numpy.real(zobs[i,j])**2+numpy.imag(zobs[i,j])**2)**0.5)
                print(apprhoErr,phaseErr)
                axes[0,j+2*i].plot(x,apprhoCalc,color="red",label=component)
                axes[0,j+2*i].plot(x,apprhoObs,color="blue",marker="o",linewidth=0,ls=None,label=component)
                axes[0,j+2*i].errorbar(x,apprhoObs,apprhoErr,color="blue",ecolor="blue",fmt='none')
                axes[0,j+2*i].set_yscale("log")
                axes[0,j+2*i].set_ylim(1e-2,1e6)
                axes[0,j+2*i].set_xscale("log")
                axes[0,j+2*i].invert_xaxis()
                axes[0,j+2*i].set_title("AppRho_"+component)
                axes[1,j+2*i].plot(x,phaseCalc,color="red")
                axes[1,j+2*i].plot(x,phaseObs,color="blue",marker="o",linewidth=0,ls=None,label=component)
                axes[1,j+2*i].errorbar(x,phaseObs,phaseErr,color="blue",ecolor="blue",label=component)
                axes[1,j+2*i].set_xscale("log")
                axes[1,j+2*i].set_ylim(-180,180)
                axes[1,j+2*i].invert_xaxis()
                axes[1,j+2*i].set_title("Phase_"+component)
        fig.suptitle(station.name+"_Impedance RMS:"+str(station.RMS_Z))
        #plt.legend()
        plt.savefig(outputFolder+"/"+station.name+"_Impedance.png")   
        plt.close()

#==================================TIPPER=====================================
tipperPlot=True
if tipperPlot:
    fc=open(obsCalcTipperFile)
    linesCalc=fc.readlines()
    freqs=[]
    for i in range(1,nFreqs+1):
        freqs.append(SplitLine(linesCalc[i],True)[0])
        
    fo=open(obsTipperFile)
    linesObs=fo.readlines()
    i=0

    stations=[]
    while i<len(linesObs):
        station=Station("NULL")
        loc=SplitLine(linesObs[i])
        station.x=float(loc[0])
        station.y=float(loc[1])
        i+=1
        for j in range(nFreqs):
            t=numpy.zeros(2, dtype=numpy.complex128)
            tVar=numpy.zeros(2)
            Ttmp=SplitLine(linesObs[i+2*j],True)
            t[0]=Ttmp[0]+Ttmp[1]*comp
            t[1]=Ttmp[2]+Ttmp[3]*comp
            TVarTmp=SplitLine(linesObs[i+2*j+1],True)
            tVar[0]=TVarTmp[0] #Assume that error has same in real and imag
            tVar[1]=TVarTmp[2]

            station.TsObs.append(copy.deepcopy(t))
            station.TsVar.append(copy.deepcopy(tVar))
        station.nFreqs=nFreqs
        stations.append(station)
        
        i+=2*nFreqs
    
    fl=open(outputLocationFile)
    linesLoc=fl.readlines()
    names={}
    for station in stations:
        dist=1e30
        for line in linesLoc:
            s=SplitLine(line)
            x=float(s[1])
            y=float(s[2])
            nameTmp=s[0].split("/")[len(s[0].split("/"))-1]
            if ((x-station.x)**2+(y-station.y)**2)**0.5<dist:
                
                name=s[0].split("/")[len(s[0].split("/"))-1]
                dist=((x-station.x)**2+(y-station.y)**2)**0.5
            elif ((x-station.x)**2+(y-station.y)**2)**0.5==dist:
                if name in names:
                    name=s[0].split("/")[len(s[0].split("/"))-1]

        names[name]=1
        station.name=name


    i=nFreqs+1
    

    stations_relatedData=[]
    alreadyRelatedCalc=numpy.zeros(len(stations))

    while i<len(linesCalc):

        loc=SplitLine(linesCalc[i])
        x=float(loc[1])
        y=float(loc[2])
        dist=1e30
        istation=-1
        for j,station in enumerate(stations):    
            if ((x-station.x)**2+(y-station.y)**2)**0.5<dist:   
                
                istation=j
                dist=((x-station.x)**2+(y-station.y)**2)**0.5

        station=stations[istation]
        if alreadyRelatedCalc[istation]==0:
            alreadyRelatedCalc[istation]=1     
            stations_relatedData.append(station)
        i+=1
        for j in range(nFreqs):
            t=numpy.zeros(2, dtype=numpy.complex128)
            tcalcTmp=SplitLine(linesCalc[i+2*j])

            t[0]=RemoveParenthesis(tcalcTmp[0])
            t[1]=RemoveParenthesis(tcalcTmp[1])
            station.TsCalc.append(copy.deepcopy(t))
        
        i+=2*nFreqs

    #同じセルに2つ観測データがある場合の対応
    for i,station1 in enumerate(stations):
            
            if alreadyRelatedCalc[i]==0:
                x1=station1.x
                y1=station1.y
                dist=1e30
                for j,station2 in enumerate(stations):

                    if alreadyRelatedCalc[j]!=0:
                        x2=station2.x
                        y2=station2.y
                        if dist>((x1-x2)**2+(y1-y2)**2)**0.5:
                                dist=((x1-x2)**2+(y1-y2)**2)**0.5
                                istation=j
                d1=((station1.x-stations[istation].x_calc)**2+(station1.y-stations[istation].y_calc)**2)**0.5
                d2=((stations[istation].x-stations[istation].x_calc)**2+(stations[istation].y-stations[istation].y_calc)**2)**0.5
                print("duplicate:",station1.name,stations[istation].name,d1,d2)            

                station1.TsCalc=copy.deepcopy(stations[istation].TsCalc)
                stations_relatedData.append(copy.deepcopy(station1))
        
    stations=stations_relatedData
    
    #Plot
    for station in stations:
        station.CalcRMS_T()
        fig, axes = plt.subplots(2, 2, tight_layout=True,figsize=(14.0, 6.0))
        for i in range(2):
            if i==0:
                component="X"
            elif i==1:
                component="Y"

            title=station.name+"_Tipper"+component
            x=numpy.array(freqs)
            TRealObs=numpy.zeros(nFreqs)
            TImagObs=numpy.zeros(nFreqs)
            TRealCalc=numpy.zeros(nFreqs)
            TImagCalc=numpy.zeros(nFreqs)
            Tvar=numpy.zeros(nFreqs)
            for ii in range(nFreqs):
                tobs=station.TsObs[ii]
                tcalc=station.TsCalc[ii]
                TRealObs[ii]=numpy.real(tobs[i])
                TRealCalc[ii]=numpy.real(tcalc[i])
                TImagObs[ii]=numpy.imag(tobs[i])
                TImagCalc[ii]=numpy.imag(tcalc[i])
                Tvar[ii]=station.TsVar[ii][i]
                if Tvar[ii]>1e3:
                    Tvar[ii]=numpy.nan
                    TRealObs[ii]=numpy.nan
                    TImagObs[ii]=numpy.nan
            axes[0,i].plot(x,TRealCalc,color="red")
            axes[0,i].plot(x,TRealObs,color="blue",ls=None,marker="o",linewidth=0)
            axes[0,i].errorbar(x,TRealObs,Tvar,color="blue",ecolor="blue",fmt='none')
            axes[0,i].set_xscale("log")
            axes[0,i].invert_xaxis()
            axes[0,i].set_ylim(-1,1)
            axes[0,i].set_title("Tipper_"+component+"_Real")
            axes[1,i].plot(x,TImagCalc,color="red")
            axes[1,i].plot(x,TImagObs,color="blue",ls=None,marker="o",linewidth=0)
            axes[1,i].errorbar(x,TImagObs,Tvar,color="blue",ecolor="blue",fmt='none')
            axes[1,i].set_xscale("log")
            axes[1,i].invert_xaxis()
            axes[1,i].set_ylim(-1,1)
            axes[1,i].set_title("Tipper_"+component+"_Imag")
        fig.suptitle(station.name+"_Tipper RMS:"+str(station.RMS_T))
        plt.savefig(outputFolder+"/"+station.name+"_Tipper.png")   
        plt.close()      