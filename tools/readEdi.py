import glob
import numpy
import re
import copy
from pyproj import Transformer
import matplotlib.pyplot as plt
from scipy import interpolate
#===Parameters===================
ediFileList="edilist.lst"
outputImpedanceFile="obsDataImpedance.txt"
outputTipperFile="obsDataTipperFile.txt"
outputFreqFile="CalcFreq.txt"
outputLocationFile="ObsStationLocation.txt"
wgs84_epsg, rect_epsg = 4326,  3112  #3112  is for australia

minFreq=0.001 #For Calculation
maxFreq=100
numOfFreq=16
efZxx=0.1
efZxy=0.05
efZyx=0.05
efZyy=0.1
efTx=0.1
efTy=0.1
needs_SIconv=1

#=================================
nullNum=-9999
def SplitLine(l,toValue=False):
    s=re.split("[\t ,]",l)
    ret=[]
    for tmp in s:
        if tmp!="":
            if tmp[len(tmp)-1]=="\n":
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

def CalcImpedance(xprInp,stacks):
    try:
        ERx=numpy.zeros((2,1), dtype=numpy.complex128)
        ERy=numpy.zeros((2,1), dtype=numpy.complex128)
        HRx=numpy.zeros((2,1), dtype=numpy.complex128)
        HRy=numpy.zeros((2,1), dtype=numpy.complex128)
        HR=numpy.zeros((2,2), dtype=numpy.complex128)
        ER=numpy.zeros((2,2), dtype=numpy.complex128)
        z=numpy.zeros((2,2), dtype=numpy.complex128)
        varz=numpy.zeros((2,2))
        vartf=numpy.zeros((1,2))
        tf=numpy.zeros((1,2), dtype=numpy.complex128)
        xpr=numpy.zeros((7,7), dtype=numpy.complex128)
        
        hx=0
        hy=1
        hz=2
        ex=3
        ey=4
        rx=5
        ry=6
        xpr=copy.deepcopy(xprInp)
        if needs_SIconv:
            ec=1e-6
            hc=1e-9/4/numpy.pi/1e-7
            chanval=[hc,hc,hc,ec,ec,hc,hc]
            conv=numpy.zeros((7,7))
            for i in range(7):
                for j in range(7):
                    conv[i,j]=chanval[i]*chanval[j]
            xpr=conv*xpr
        
        ERx[0,0]=xpr[ex,rx]
        ERx[1,0]=xpr[ey,rx]
        ERy[0,0]=xpr[ex,ry]
        ERy[1,0]=xpr[ey,ry]

        HRx[0,0]=xpr[hx,rx]
        HRx[1,0]=xpr[hy,rx]
        HRy[0,0]=xpr[hx,ry]
        HRy[1,0]=xpr[hy,ry]
        
        HR[0,0]=HRx[0,0]
        HR[1,0]=HRx[1,0]
        HR[0,1]=HRy[0,0]
        HR[1,1]=HRy[1,0]    
        
        ER[0,0]=ERx[0,0]
        ER[1,0]=ERx[1,0]
        ER[0,1]=ERy[0,0]
        ER[1,1]=ERy[1,0]  
        
        #impedance
        z=numpy.dot(ER,numpy.linalg.inv(HR))
        
        
        tf[0,0]=xpr[hz,rx]
        tf[0,1]=xpr[hz,ry]
        #tipper
        tf=numpy.dot(tf,numpy.linalg.inv(HR))
        
        #variences
        
        d=numpy.linalg.det(HR)
        denom=numpy.abs(d)**2*stacks
        
        
        ax2=  xpr[rx,rx]*abs(xpr[hy,ry])**2 + xpr[ry,ry]*abs(xpr[hy,rx])**2 \
        - 2*numpy.real(xpr[rx,ry]*xpr[hy,ry].conjugate()*xpr[hy,rx])
        
        ay2=  xpr[ry,ry]*abs(xpr[hx,rx])**2 \
        + xpr[rx,rx]*abs(xpr[hx,ry])**2 \
        - 2*numpy.real(xpr[rx,ry]*xpr[hx,ry].conjugate()*xpr[hx,rx])
        
        nx2=xpr[ex,ex] \
        -2*numpy.real( z[0,0]*xpr[hx,ex]+z[0,1]*xpr[hy,ex]-z[0,0]*z[0,1].conjugate()*xpr[hx,hy]) \
        +abs(z[0,0])**2*xpr[hx,hx]\
        +abs(z[0,1])**2*xpr[hy,hy]

        ny2=xpr[ey,ey] \
        -2*numpy.real(z[1,0]*xpr[hx,ey]+z[1,1]*xpr[hy,ey]-z[1,0]*z[1,1].conjugate()*xpr[hx,hy])\
        +abs(z[1,0])**2*xpr[hx,hx]\
        +abs(z[1,1])**2*xpr[hy,hy]

        varz[0,0]=nx2*ax2/denom
        varz[0,1]=nx2*ay2/denom
        varz[1,0]=ny2*ax2/denom
        varz[1,1]=ny2*ay2/denom
        
        nz2=xpr[hz,hz] \
        -2*numpy.real( tf[0,0]*xpr[hx,hz] + tf[0,1]*xpr[hy,hz]- tf[0,0]*tf[0,1].conjugate()*xpr[hx,hy])\
        +abs(tf[0,0])**2*xpr[hx,hx]\
        +abs(tf[0,1])**2*xpr[hy,hy]
        

        vartf[0,0]=nz2*ax2/denom;
        vartf[0,1]=nz2*ay2/denom

    except:
        z=numpy.zeros((2,2), dtype=numpy.complex128)
        varz=numpy.zeros((2,2))
        vartf=numpy.zeros((1,2))
        tf=numpy.zeros((1,2), dtype=numpy.complex128)

    return [z,tf,varz,vartf]


def _read_block_values(lines, start_index):
    # Read a numeric block that follows an EDI section header.
    # The header line itself is at lines[start_index].
    # Returns (values_list, next_index_after_block).
    vals = []
    i = start_index + 1
    while i < len(lines):
        line = lines[i].strip()
        # next section header or empty line ends this block
        if line == "" or line.startswith(">"):
            break
        # split by whitespace / comma and keep things that look like numbers
        for tok in re.split(r"[\s,]+", line):
            if tok == "":
                continue
            # ignore obvious non-numeric tokens
            if not re.match(r"[+-]?\d", tok):
                continue
            try:
                vals.append(float(tok))
            except Exception:
                pass
        i += 1
    return vals, i

def ReadImpedanceFromEDI(lines, station):
    # Read impedance/tipper sections from an EDI file that already contains
    # Zxx, Zxy, Zyx, Zyy and Tzx, Tzy instead of SPECTRA.
    # Filled attributes:
    #   station.freqs, station.Zs, station.Ts, station.varZs, station.varTs
    freqs = None
    # impedance components
    Zxxr = Zxxi = Zxyr = Zxyi = Zyxr = Zyxi = Zyyr = Zyyi = None
    # impedance variances (optional)
    Zxxv = Zxyv = Zyxv = Zyyv = None
    # tipper components (optional)
    Txr = Txi = Tyr = Tyi = None
    # tipper variances (optional)
    Txv = Tyv = None

    i = 0
    nlines = len(lines)
    while i < nlines:
        up = lines[i].strip().upper()
        if up.startswith(">FREQ"):
            freqs, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">ZXXR"):
            Zxxr, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">ZXXI"):
            Zxxi, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">ZXYR"):
            Zxyr, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">ZXYI"):
            Zxyi, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">ZYXR"):
            Zyxr, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">ZYXI"):
            Zyxi, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">ZYYR"):
            Zyyr, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">ZYYI"):
            Zyyi, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">ZXX.VAR"):
            Zxxv, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">ZXY.VAR"):
            Zxyv, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">ZYX.VAR"):
            Zyxv, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">ZYY.VAR"):
            Zyyv, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">TXR"):
            Txr, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">TXI"):
            Txi, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">TYR"):
            Tyr, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">TYI"):
            Tyi, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">TXVAR"):
            Txv, i = _read_block_values(lines, i)
            continue
        elif up.startswith(">TYVAR"):
            Tyv, i = _read_block_values(lines, i)
            continue
        else:
            i += 1

    if freqs is None or Zxxr is None or Zxxi is None \
       or Zxyr is None or Zxyi is None \
       or Zyxr is None or Zyxi is None \
       or Zyyr is None or Zyyi is None:
        # nothing to do – leave station empty
        return

    n = min(len(freqs), len(Zxxr), len(Zxxi), len(Zxyr), len(Zxyi),
            len(Zyxr), len(Zyxi), len(Zyyr), len(Zyyi))

    # helper: safe accessor for variance arrays
    def _get_var(arr, idx):
        if arr is None or idx >= len(arr):
            return 0.0
        return float(arr[idx])

    empty_val = 1e30

    for idx in range(n):
        f = freqs[idx]
        # skip obviously empty entries (Geotools uses 1e32 etc.)
        if abs(Zxxr[idx]) >= empty_val or abs(Zxxi[idx]) >= empty_val:
            continue
        if abs(Zxyr[idx]) >= empty_val or abs(Zxyi[idx]) >= empty_val:
            continue
        if abs(Zyxr[idx]) >= empty_val or abs(Zyxi[idx]) >= empty_val:
            continue
        if abs(Zyyr[idx]) >= empty_val or abs(Zyyi[idx]) >= empty_val:
            continue

        Z = numpy.zeros((2, 2), dtype=numpy.complex128)
        Z[0, 0] = Zxxr[idx] + 1j * Zxxi[idx]
        Z[0, 1] = Zxyr[idx] + 1j * Zxyi[idx]
        Z[1, 0] = Zyxr[idx] + 1j * Zyxi[idx]
        Z[1, 1] = Zyyr[idx] + 1j * Zyyi[idx]

        varZ = numpy.zeros((2, 2))
        varZ[0, 0] = _get_var(Zxxv, idx)
        varZ[0, 1] = _get_var(Zxyv, idx)
        varZ[1, 0] = _get_var(Zyxv, idx)
        varZ[1, 1] = _get_var(Zyyv, idx)

        T = numpy.zeros((1, 2), dtype=numpy.complex128)
        varT = numpy.zeros((1, 2))

        if Txr is not None and Txi is not None and idx < len(Txr) and idx < len(Txi):
            if abs(Txr[idx]) < empty_val and abs(Txi[idx]) < empty_val:
                T[0, 0] = Txr[idx] + 1j * Txi[idx]
        if Tyr is not None and Tyi is not None and idx < len(Tyr) and idx < len(Tyi):
            if abs(Tyr[idx]) < empty_val and abs(Tyi[idx]) < empty_val:
                T[0, 1] = Tyr[idx] + 1j * Tyi[idx]

        varT[0, 0] = _get_var(Txv, idx)
        varT[0, 1] = _get_var(Tyv, idx)

        station.freqs.append(float(f))
        station.Zs.append(copy.deepcopy(Z))
        station.Ts.append(copy.deepcopy(T))
        station.varZs.append(copy.deepcopy(varZ))
        station.varTs.append(copy.deepcopy(varT))


class Station:
    def __init__(self, name):
        self.name=name
        self.lat=nullNum
        self.long=nullNum
        self.elev=nullNum
        self.x=nullNum
        self.y=nullNum
        self.freqs=[]
        self.Zs=[]
        self.Ts=[]
        self.varZs=[]
        self.varTs=[]
        
        self.freqsUsed={}
        self.ZsUsed={}
        self.TsUsed={}
        self.varZsUsed={}
        self.varTsUsed={}
        
    def CalcInterpolate(self):
        #Zs
        self.f_Zs=[[0,0],[0,0]]
        
        for i in range(2):
            for j in range(2):
                vals=[]
                for ii in range(len(self.freqs)):
                    vals.append(self.Zs[ii][i,j])
                
                self.f_Zs[i][j]=interpolate.interp1d(self.freqs, vals)
        #varZs
        self.f_varZs=[[0,0],[0,0]]
        
        for i in range(2):
            for j in range(2):
                vals=[]
                for ii in range(len(self.freqs)):
                    vals.append(self.varZs[ii][i,j])
                
                self.f_varZs[i][j]=interpolate.interp1d(self.freqs, vals)
                
        #Ts
        self.f_Ts=[[0,0]]
        
        for i in range(1):
            for j in range(2):
                vals=[]
                for ii in range(len(self.freqs)):
                    vals.append(self.Ts[ii][i,j])
                
                self.f_Ts[i][j]=interpolate.interp1d(self.freqs, vals)
        
        #varTs
        self.f_varTs=[[0,0]]
        
        for i in range(1):
            for j in range(2):
                vals=[]
                for ii in range(len(self.freqs)):
                    vals.append(self.varTs[ii][i,j])
                
                self.f_varTs[i][j]=interpolate.interp1d(self.freqs, vals)
        
    def GetZs(self,freq):
        Zstmp=numpy.zeros((2,2),dtype=numpy.complex128)
        for i in range(2):
            for j in range(2):
                try:
                    Zstmp[i,j]=self.f_Zs[i][j](freq)
                except:
                    Zstmp[i,j]=0
                    
        return Zstmp     
    def GetVarZs(self,freq):
        Zstmp=numpy.zeros((2,2))
        for i in range(2):
            for j in range(2):
                try:
                    Zstmp[i,j]=self.f_varZs[i][j](freq)
                except:
                    Zstmp[i,j]=1e10
        return Zstmp
    def GetTs(self,freq):
        Zstmp=numpy.zeros((1,2),dtype=numpy.complex128)
        for i in range(1):
            for j in range(2):
                try:
                    Zstmp[i,j]=self.f_Ts[i][j](freq)
                except:
                    Zstmp[i,j]=0
        return Zstmp 
    def GetVarTs(self,freq):
        Zstmp=numpy.zeros((1,2))
        for i in range(1):
            for j in range(2):
                try:
                    Zstmp[i,j]=self.f_varTs[i][j](freq)
                except:
                    Zstmp[i,j]=1e10
        return Zstmp 
        
freqsCalc=numpy.zeros(numOfFreq)
df=(numpy.log10(maxFreq)-numpy.log10(minFreq))/(numOfFreq-1)
for i in range(numOfFreq):
        freqsCalc[i]=10**(numpy.log10(minFreq)+df*i)
print("Frequencies:",freqsCalc)

f=open(ediFileList)
files=f.readlines()
f.close()

lats=[]
longs=[]
elevs=[]

stations=[]
fa=open("latLon.csv","w")
fa.write("ID,lat,lon\n")
for tmpfile in files:
    print(tmpfile)
    file=tmpfile[:-1]
    print(file)
    station=Station(file)
    stations.append(station)
    f=open(file)
    lines=f.readlines()
    f.close()
    allFound=[False,False,False]
    for tmpline in lines:
        line=SplitLine(tmpline)
        if len(line)==0:
            continue
        if line[0][:3].upper()=="LAT":
            vals=tmpline.split("=")[1].split(":")
            val=float(vals[0])+float(vals[1])/60+float(vals[2])/3600
            allFound[0]=True
            station.lat=val
        elif line[0][:4].upper()=="LONG":
            vals=tmpline.split("=")[1].split(":")
            val=float(vals[0])+float(vals[1])/60+float(vals[2])/3600
            allFound[1]=True
            station.long=val
        elif line[0][:4].upper()=="ELEV":
            vals=tmpline.split("=")[1].split(":")
            val=float(vals[0])
            allFound[2]=True
            station.elev=val
        if allFound[0]*allFound[1]*allFound[2]:
            break
    fn=file.split("/")[len(file.split("/"))-1]
    fn=fn.split(".")[0][:6]
    fa.write(fn+","+str(station.lat)+","+str(station.long)+"\n")
    #data read
    print("File:", file)
    # Decide whether this EDI has SPECTRA (cross-power) or pre-computed impedance
    has_spectra = False
    for tmpline in lines:
        parts = SplitLine(tmpline)
        if len(parts) > 0 and parts[0].upper() == ">SPECTRA":
            has_spectra = True
            break

    if has_spectra:
        i = 0
        comp = 1j
        while i < len(lines):
            line = SplitLine(lines[i])
            if len(line) == 0:
                i += 1
                continue

            if line[0].upper() == ">SPECTRA":
                station.freqs.append(float(line[1].split("=")[1]))
                for s in line:
                    if s[:4].upper() == "AVGT":
                        stacks = float(s.split("=")[1])

                array = numpy.zeros((7, 7))
                xpr = numpy.zeros((7, 7), dtype=numpy.complex128)
                for j in range(7):
                    vals = SplitLine(lines[i+1+j], True)
                    for k in range(7):
                        array[j, k] = vals[k]
                for row in range(7):
                    xpr[row, row] = array[row, row]
                    for col in range(row+1, 7):
                        xpr[row, col] = array[col, row]-comp*array[row, col]
                        xpr[col, row] = array[col, row]+comp*array[row, col]

                [Z, T, varZ, varT] = CalcImpedance(xpr, stacks)

                station.Zs.append(copy.deepcopy(Z))
                station.Ts.append(copy.deepcopy(T))
                station.varZs.append(copy.deepcopy(varZ))
                station.varTs.append(copy.deepcopy(varT))
                i += 7

            i += 1
    else:
        # Read impedance/tipper directly from the EDI file
        ReadImpedanceFromEDI(lines, station)
fa.close()
   
#=========以降は解析用に書き込み================================
tr = Transformer.from_proj(wgs84_epsg, rect_epsg)
trInv=Transformer.from_proj(rect_epsg,wgs84_epsg)
#Model Center

aveX=0
aveY=0
for station in stations:
    x,y=tr.transform(station.lat, station.long)
    aveX+=x
    aveY+=y
    
aveX/=len(stations)
aveY/=len(stations)

latc,longc=trInv.transform(aveX,aveY)
f=open("CenterLatLong.txt","w")
f.write(str(latc)+" "+str(longc))
f.close()
xc=aveX
yc=aveY

for station in stations:
    station.x,station.y=tr.transform(station.lat, station.long)
    station.x=station.x-xc
    station.y=station.y-yc
    
    
    station.CalcInterpolate()

    #Freq Data For Calc using interpolation data.
    for freqCalc in freqsCalc:
        
        station.freqsUsed[freqCalc]=freqCalc
        station.ZsUsed[freqCalc]=station.GetZs(freqCalc)
        station.TsUsed[freqCalc]=station.GetTs(freqCalc)
        station.varZsUsed[freqCalc]=station.GetVarZs(freqCalc)
        station.varTsUsed[freqCalc]=station.GetVarTs(freqCalc)
#write to files
fz=open(outputImpedanceFile,"w")
ft=open(outputTipperFile,"w")


for station in stations:
    fz.write(str(station.x)+" "+str(station.y)+" ")
    fz.write(station.name+"\n")
    ft.write(str(station.x)+" "+str(station.y)+" ")
    ft.write(station.name+"\n")
    for freqCalc in freqsCalc:
        wz=""
        wz+=str(numpy.real(station.ZsUsed[freqCalc][0,0]))+" "+str(numpy.imag(station.ZsUsed[freqCalc][0,0]))+" "
        wz+=str(numpy.real(station.ZsUsed[freqCalc][0,1]))+" "+str(numpy.imag(station.ZsUsed[freqCalc][0,1]))+" "
        wz+=str(numpy.real(station.ZsUsed[freqCalc][1,0]))+" "+str(numpy.imag(station.ZsUsed[freqCalc][1,0]))+" "
        wz+=str(numpy.real(station.ZsUsed[freqCalc][1,1]))+" "+str(numpy.imag(station.ZsUsed[freqCalc][1,1]))+"\n"
        
        nonDiag=(((numpy.real(station.ZsUsed[freqCalc][0,1])**2+numpy.imag(station.ZsUsed[freqCalc][0,1])**2)**0.5)*
        ((numpy.real(station.ZsUsed[freqCalc][1,0])**2+numpy.imag(station.ZsUsed[freqCalc][1,0])**2)**0.5))**0.5
        
        e=max(nonDiag*efZxx,station.varZsUsed[freqCalc][0,0]**0.5)
        wz+=str(e)+" "+str(e)+" "
        e=max(nonDiag*efZxy,station.varZsUsed[freqCalc][0,1]**0.5)
        wz+=str(e)+" "+str(e)+" "
        e=max(nonDiag*efZyx,station.varZsUsed[freqCalc][1,0]**0.5)
        wz+=str(e)+" "+str(e)+" "
        e=max(nonDiag*efZyy,station.varZsUsed[freqCalc][1,1]**0.5)
        wz+=str(e)+" "+str(e)+"\n"
        
        wi=""
        wi+=str(numpy.real(station.TsUsed[freqCalc][0,0]))+" "+str(numpy.imag(station.TsUsed[freqCalc][0,0]))+" "
        wi+=str(numpy.real(station.TsUsed[freqCalc][0,1]))+" "+str(numpy.imag(station.TsUsed[freqCalc][0,1]))+"\n"
        
        nonDiag=((((numpy.real(station.TsUsed[freqCalc][0,0]))**2+(numpy.imag(station.TsUsed[freqCalc][0,0]))**2)**0.5)*
        (((numpy.real(station.TsUsed[freqCalc][0,1]))**2+(numpy.imag(station.TsUsed[freqCalc][0,1]))**2)**0.5))**0.5
        e=max(efTx*nonDiag,numpy.real(station.varTsUsed[freqCalc][0,0]**0.5))
        wi+=str(e)+" "+str(e)+" "
        e=max(efTy*nonDiag,numpy.real(station.varTsUsed[freqCalc][0,1]**0.5))
        wi+=str(e)+" "+str(e)+"\n"
        
        fz.write(wz)
        ft.write(wi)
fz.close()
ft.close()

fw=open(outputFreqFile,"w")
for freqCalc in freqsCalc:
    fw.write(str(freqCalc)+"\n")
fw.close()

fl=open(outputLocationFile,"w")
for station in stations:
    fl.write(station.name+" "+str(station.x)+" "+str(station.y)+"\n")
fl.close()
