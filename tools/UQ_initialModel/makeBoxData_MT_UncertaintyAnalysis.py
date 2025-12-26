import numpy
import copy
from scipy import interpolate
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
from pyproj import Transformer
import re
import random
from pykrige.uk3d import UniversalKriging3D
import argparse  
def RemoveParenthesis(s): #Convert (a,b) -> a+b*j

    a=float(s.split("(")[1].split(",")[0])

    b=float(s.split("(")[1].split(",")[1].split(")")[0])
    c=a+comp*b

    return c
        
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
def makeInterpFunc(fname):
    v=numpy.loadtxt(fname)
    f = NearestNDInterpolator(v[:,0:2],v[:,2])
    xmin=v[:,0].min()
    xmax=v[:,0].max()
    ymin=v[:,1].min()
    ymax=v[:,1].max()
    return f,xmin,xmax,ymin,ymax
class Element:
    def __init__(self):
        self.ID=None
        self.dx=0
        self.dy=0
        self.dz=0
        self.layer=0
        self.parent=False
        self.rootCoord=numpy.zeros(3)
        self.centerCoord=numpy.zeros(3)
        self.rhoID=3
        self.ix=-1
        self.iy=-1
        self.iz=-1
        self.nodes=[numpy.zeros(3) for i in range(8)]
        self.childElements=[]
        self.placeInParents=[]
def getBinaryValue(i,j):
    if (i>2 or j>2):
        print("Error")
        exit
    string=str(j)+str(i)
    return string



# xPrecise[1][0]=0.*xMax
# xPrecise[1][1]=1*xMax
# yPrecise[1][0]=0*yMax
# yPrecise[1][1]=1*yMax
# zPrecise[1][0]=0.375*zMax
# zPrecise[1][1]=0.75*zMax

# ===============================
# Configuration Parameters
# (Centralized here for easier tuning / reproducibility)
# NOTE: Values are kept identical to the original script behavior.
# ===============================

CONFIG = {
    # Parameters how to split calculation region
    "numSplitPrecise": 3,

    # Base mesh sizes
    "nx": 34,
    "ny": 34,
    "nz": 66,
    "nAir": 10,

    # Cell-size caps (meters)
    "dxMax": 1000000,
    "dyMax": 1000000,
    "dzMax": 10000000,
    "dzMaxInAir": 1000000,

    # Vertical reference (kept for compatibility)
    "originZ": 100,

    # Core region (number of core cells around the center)
    "corenx": 6,
    "coreny": 6,
    "corenz": 3,

    # Air-side vertical growth handling (effective value in the original script was 1)
    "corenz_air": 1,

    # These were present in the original script (kept for compatibility even if unused)
    "nz10km": 32,
    "nz30km": 10,
    "z_coeff1": 1.3,
    "z_coeff2": 1.3,

    # Minimum cell sizes (meters)
    "minDx": 3200,
    "minDy": 3200,
    "minDz": 30,

    # Geometric growth coefficients
    "x_coeff": 1.3,
    "y_coeff": 1.3,
    "z_coeff": 1.15,
    "z_coeffAir": 1.3,

    # Control points for Kriging (count in x/y/z)
    "ncx": 3,
    "ncy": 3,
    "ncz": 4,

    # Control point placement (indices in the base mesh)
    "nxmin_offset": -3,
    "nxmax_offset": +3,
    "nymin_offset": -3,
    "nymax_offset": +3,
    "nzmin_offset": 0,    # relative to nAir
    "nzmax_offset": 30,   # relative to nAir

    # Resistivity range for control points
    "minResisCp": 1,
    "maxResisCp": 1000,

    # Kriging slope multiplier (kept as-is)
    "factorKr": 10,

    # Frequency settings (for omega list)
    "minFreq": 1/1000,  # For Calculation
    "maxFreq": 100,
    "numOfFreq": 12,

    # "Precise" nested refinement windows (kept identical to original logic)
    # Each item defines a window centered at (nx/2, ny/2) with half-widths (precnx, precny)
    # and a z-range given by offsets relative to nAir (inclusive indices).
    "precise_windows": [
        {"precnx": 6, "precny": 6, "z0_off": -8, "z1_off": 34},
        {"precnx": 4, "precny": 4, "z0_off": -7, "z1_off": 30},
        {"precnx": 3, "precny": 3, "z0_off": -5, "z1_off": 26},
        {"precnx": 2, "precny": 2, "z0_off": -4, "z1_off": 10},
    ],
}

# --- Unpack config (variable names preserved to keep downstream code unchanged) ---
numSplitPrecise = CONFIG["numSplitPrecise"]
nx = CONFIG["nx"]
ny = CONFIG["ny"]
nz = CONFIG["nz"]
nAir = CONFIG["nAir"]

dxMax = CONFIG["dxMax"]
dyMax = CONFIG["dyMax"]
dzMax = CONFIG["dzMax"]
dzMaxInAir = CONFIG["dzMaxInAir"]

originZ = CONFIG["originZ"]

corenx = CONFIG["corenx"]
coreny = CONFIG["coreny"]
corenz = CONFIG["corenz"]
corenz_air = CONFIG["corenz_air"]

nz10km = CONFIG["nz10km"]
nz30km = CONFIG["nz30km"]
z_coeff1 = CONFIG["z_coeff1"]
z_coeff2 = CONFIG["z_coeff2"]

minDx = CONFIG["minDx"]
minDy = CONFIG["minDy"]
minDz = CONFIG["minDz"]

x_coeff = CONFIG["x_coeff"]
y_coeff = CONFIG["y_coeff"]
z_coeff = CONFIG["z_coeff"]
z_coeffAir = CONFIG["z_coeffAir"]

ncx = CONFIG["ncx"]
ncy = CONFIG["ncy"]
ncz = CONFIG["ncz"]

nxmin = nx/2 + CONFIG["nxmin_offset"]
nxmax = nx/2 + CONFIG["nxmax_offset"]
nymin = ny/2 + CONFIG["nymin_offset"]
nymax = ny/2 + CONFIG["nymax_offset"]
nzmin = nAir + CONFIG["nzmin_offset"]
nzmax = nAir + CONFIG["nzmax_offset"]

minResisCp = CONFIG["minResisCp"]
maxResisCp = CONFIG["maxResisCp"]
factorKr = CONFIG["factorKr"]

minFreq = CONFIG["minFreq"]
maxFreq = CONFIG["maxFreq"]
numOfFreq = CONFIG["numOfFreq"]

# Build precise windows (xPrecise/yPrecise/zPrecise) exactly as before
xPrecise=[numpy.zeros(2),numpy.zeros(2),numpy.zeros(2),numpy.zeros(2)]
yPrecise=[numpy.zeros(2),numpy.zeros(2),numpy.zeros(2),numpy.zeros(2)]
zPrecise=[numpy.zeros(2),numpy.zeros(2),numpy.zeros(2),numpy.zeros(2)]

for idx, w in enumerate(CONFIG["precise_windows"]):
    precnx = w["precnx"]
    precny = w["precny"]

    xPrecise[idx][0]=int(nx/2-precnx)
    xPrecise[idx][1]=int(nx/2+precnx)
    yPrecise[idx][0]=int(ny/2-precny)
    yPrecise[idx][1]=int(ny/2+precny)
    zPrecise[idx][0]=int(nAir + w["z0_off"])
    zPrecise[idx][1]=int(nAir + w["z1_off"])

oriNumPre=0
numOfCells=0
for i in range(numSplitPrecise+1):
    if i==0:
        oriNum=nx*ny*nz
        minusNum=0
    else:
        oriNum=(xPrecise[i-1][1]-xPrecise[i-1][0]+1)*(yPrecise[i-1][1]-yPrecise[i-1][0]+1)
        minusNum=oriNum*4**(i-1)
        oriNum=oriNum*4**i
        minusNum=minusNum*(zPrecise[i-1][1]-zPrecise[i-1][0]+1)
        oriNum=oriNum*(zPrecise[i-1][1]-zPrecise[i-1][0]+1)

    numOfCells+=oriNum-minusNum
print("NumOfCells",numOfCells)

xCoords=[0]
for ix in range(nx):
    if ix<nx/2-1-corenx:
        xc=x_coeff**abs(ix-(nx/2-1-corenx))
    elif ix>nx/2+corenx:
        xc=x_coeff**abs(ix-nx/2-corenx)
    else:
        xc=1
    if minDx*xc<dxMax:
        xCoords.append(xCoords[ix]+minDx*xc)
    else:
        xCoords.append(xCoords[ix]+dxMax) 
xCoords=numpy.array(xCoords)
xCoords=xCoords-xCoords[nx]/2
print("Xmax:",xCoords[nx])
xMax=xCoords[nx]

yCoords=[0]
for iy in range(ny):
    if iy<ny/2-1-coreny:
        yc=y_coeff**abs(iy-(ny/2-1-coreny))
    elif iy>ny/2+coreny:
        yc=y_coeff**abs(iy-ny/2-coreny)
    else:
        yc=1
    if minDy*yc<dyMax:
        yCoords.append(yCoords[iy]+minDy*yc)
    else:
        yCoords.append(yCoords[iy]+dyMax)
yCoords=numpy.array(yCoords)
yCoords=yCoords-yCoords[ny]/2

print("Ymax:",yCoords[ny])
yMax=yCoords[ny]

zCoords=[-10000]
for iz in range(nz):
    if iz<nAir-corenz:
        
        zc=z_coeffAir**abs(iz-(nAir-corenz))
    elif iz>nAir+corenz:
        zc=z_coeff**abs(iz-(nAir+corenz))
    else:
        zc=1

    if iz<nAir and minDz*zc>dzMaxInAir:
        zCoords.append(zCoords[iz]+dzMaxInAir)  
    else:
        zCoords.append(zCoords[iz]+min(dzMax, minDz*zc)) 

print("Center Of Z:",zCoords[nz]/2)
print(zCoords)
zMax=zCoords[nz]
fineCenterPoint_Z=zCoords[nAir]
print("Air Layer:",zCoords[0])


#Params for lat long to x y.


latc=38.97327250457843
longc=140.63024004567743
f_large,xmin_large,xmax_large,ymin_large,ymax_large=makeInterpFunc("15ArcSecondsElev.txt")

wgs84_epsg, rect_epsg = 4326, 6678 #6678 is for Yuzawa

tr = Transformer.from_proj(wgs84_epsg, rect_epsg)
trInv = Transformer.from_proj(rect_epsg,wgs84_epsg)
xc,yc=tr.transform(latc, longc)


xa=numpy.zeros((nx+1,ny+1,nz+1))
ya=numpy.zeros((nx+1,ny+1,nz+1))
za=numpy.zeros((nx+1,ny+1,nz+1))
oceanBottom=numpy.ones((nx+1,ny+1))*9999
for ix in range(nx+1):
    for iy in range(ny+1):
        xa[ix,iy,nAir]=xCoords[ix]
        ya[ix,iy,nAir]=yCoords[iy]
        #Calc Elev and set resis
        xtmp=xa[ix,iy,nAir]+xc
        ytmp=ya[ix,iy,nAir]+yc
        
        lattmp,longtmp=trInv.transform(xtmp,ytmp)
        
        elev=f_large(lattmp,longtmp)
        if elev<=0:
            oceanBottom[ix,iy]=elev
            elev=0
        za[ix,iy,nAir]=-elev
        
        
        #-Z方向
        
        coeff=1.0
        
        nmax=1000
        tol=1e-4
        dz=minDz
        for ii in range(nmax):
            zp=0
            z=za[ix,iy,nAir]
            bottomZ=zCoords[0]
            
            for k in range(0,nAir-1):
                if k<=corenz_air:
                    z=z-dz
                else:
                    z=z-dz*coeff**(k-corenz_air)
                    zp=zp-dz*(k-corenz_air)*coeff**((k-corenz_air)-1)
                
            res=(bottomZ-z)**2
            resp=2*(bottomZ-z)*(-zp)

            if res<tol:
                break
            elif ii==nmax-1:
                print("not converged -Z Direction:",z,bottomZ)
            coeff=coeff-0.5*res/resp
        coeff_keep=coeff
        for k in range(0,nAir-1):
            if k<=corenz_air:
                coeff=1
            else:
                coeff=coeff_keep
            za[ix,iy,nAir-(k+1)]=za[ix,iy,nAir-k]-dz*coeff**(k-corenz_air)
            xa[ix,iy,nAir-(k+1)]=xCoords[ix]
            ya[ix,iy,nAir-(k+1)]=yCoords[iy]
        xa[ix,iy,0]=xCoords[ix]
        ya[ix,iy,0]=yCoords[iy]
        za[ix,iy,0]=za[ix,iy,1]-1000     
        #+Z方向
        coeff=1.0
        
        nmax=1000
        tol=1e-4
        
        for ii in range(nmax):
            zp=0
            z=za[ix,iy,nAir]
            bottomZ=zCoords[nz]
            for k in range(1,nz+1-nAir):
                if k<=corenz:
                    z=z+dz
                else:
                    z=z+dz*coeff**(k-1-corenz)
                    zp=zp+dz*(k-1-corenz)*coeff**(k-2-corenz)
            res=(bottomZ-z)**2
            resp=2*(bottomZ-z)*(-zp)
            if res<tol:
                break
            elif ii==nmax-1:
                print("not converged +Z Direction:",z,bottomZ)
            coeff=coeff-0.5*res/resp
        coeff_keep=coeff
        for k in range(1,nz+1-nAir):
            if k<=corenz:
                coeff=1
            else:
                coeff=coeff_keep
            za[ix,iy,nAir+k]=za[ix,iy,nAir+(k-1)]+dz*coeff**(k-1-corenz)
            xa[ix,iy,nAir+k]=xCoords[ix]
            ya[ix,iy,nAir+k]=yCoords[iy]   

               
for layer in range(0,numSplitPrecise):       
    print("± X boundary layer ",layer,xa[int(xPrecise[layer][0]),0,0],xa[int(xPrecise[layer][1]),0,0])
    print("± Y boundary layer ",layer,ya[0,int(yPrecise[layer][0]),0],ya[0,int(yPrecise[layer][1]),0])
    print("Z at bottom layer ",layer,za[0,0,int(zPrecise[layer][1])])


nxchs=[]
nychs=[]
nzchs=[]
xchs=[]
ychs=[]
zchs=[]
oceanBottomchs=[]
nAirchs=[]
corenz_air_keep=corenz_air
for layer in range(0,numSplitPrecise):  
    nxch=int((xPrecise[layer][1]-xPrecise[layer][0]+1)*2**(layer+1)+1)
    nych=int((yPrecise[layer][1]-yPrecise[layer][0]+1)*2**(layer+1)+1)
    nzch=int(zPrecise[layer][1]-zPrecise[layer][0]+1)
    
    nAirch=int(nAir-zPrecise[layer][0])
    nAirchs.append(nAirch)
    nxchs.append(nxch)
    nychs.append(nych)
    nzchs.append(nzch)
    xch=numpy.zeros((nxch+1,nych+1,nzch+1))
    ych=numpy.zeros((nxch+1,nych+1,nzch+1))
    zch=numpy.zeros((nxch+1,nych+1,nzch+1))
    oceanBottomch=numpy.zeros((nxch+1,nych+1))
    
    if layer>0:
        nb=nAirchs[layer-1]-nAirch
        nt=nAirchs[layer-1]+(nzch-nAirch)
        na=nAirchs[layer-1]
        pb=[]
        pt=[]
        pa=[]
        vb=[]
        vt=[]
        va=[]
        for i in range(xchs[layer-1].shape[0]):
            for j in range(ychs[layer-1].shape[1]): 
                
                tmp=numpy.zeros(2)
                tmp[0]=xchs[layer-1][i,j,nb]
                tmp[1]=ychs[layer-1][i,j,nb]
                vb.append(zchs[layer-1][i,j,nb])
                pb.append(copy.deepcopy(tmp))
                tmp=numpy.zeros(2)
                tmp[0]=xchs[layer-1][i,j,nt]
                tmp[1]=ychs[layer-1][i,j,nt]
                vt.append(zchs[layer-1][i,j,nt])
                pt.append(copy.deepcopy(tmp))
                tmp=numpy.zeros(2)
                tmp[0]=xchs[layer-1][i,j,na]
                tmp[1]=ychs[layer-1][i,j,na]
                va.append(zchs[layer-1][i,j,na])
                pa.append(copy.deepcopy(tmp))
        fb=LinearNDInterpolator(pb,vb)
        ft=LinearNDInterpolator(pt,vt)
        fa=LinearNDInterpolator(pa,va)
    for ix in range(nxch+1):
        for iy in range(nych+1):
            
            it=int(ix/(2**(layer+1))+xPrecise[layer][0])
            jt=int(iy/(2**(layer+1))+yPrecise[layer][0])
            kt=int(zPrecise[layer][0])
            dx=(xCoords[it+1]-xCoords[it])/(2**(layer+1))
            xch[ix,iy,nAirch]=xCoords[it]+dx*(ix%(2**(layer+1)))
            dy=(yCoords[jt+1]-yCoords[jt])/(2**(layer+1))
            ych[ix,iy,nAirch]=yCoords[jt]+dy*(iy%(2**(layer+1)))
            #Calc Elev and set resis
            xtmp=xch[ix,iy,nAirch]+xc
            ytmp=ych[ix,iy,nAirch]+yc
            
            #if ix!=0 and ix!=nxch and iy!=0 and iy!=nych:
            lattmp,longtmp=trInv.transform(xtmp,ytmp)
            
            elev=f_large(lattmp,longtmp)
            if elev<=0:
                oceanBottomch[ix,iy]=elev
                elev=0
            zch[ix,iy,nAirch]=-elev
            # #-Z方向

            cs=[]
            zs=[]
            if layer==0:
                for i in range(2):
                    for j in range(2):
                        c=numpy.zeros(2)
                        c[0]=xCoords[it+i%2]
                        c[1]=yCoords[jt+j%2]
                        v=za[it+i,jt+j,kt]
                        cs.append(copy.deepcopy(c))
                        zs.append(v)
                    
            
                fz = LinearNDInterpolator(cs,zs)
                bottomZ=fz(xch[ix,iy,nAirch],ych[ix,iy,nAirch])
            else:
                bottomZ=fb(xch[ix,iy,nAirch],ych[ix,iy,nAirch])
            coeff=1.0
            
            
            #地形が範囲外に突き出してしまう場合 またははしっこ
            if bottomZ>=zch[ix,iy,nAirch]-dz*nAirch or ix==0 or ix==nxch or iy==0 or iy==nych:
                
                if layer==0:
                    cs=[]
                    zs=[]
                    for i in range(2):
                        for j in range(2):
                            c=numpy.zeros(2)
                            c[0]=xCoords[it+i%2]
                            c[1]=yCoords[jt+j%2]
                            v=za[it+i,jt+j,nAir]
                            cs.append(copy.deepcopy(c))
                            zs.append(v)
                    ftmp=LinearNDInterpolator(cs,zs)
                    zch[ix,iy,nAirch]=ftmp(xch[ix,iy,nAirch],ych[ix,iy,nAirch])
                    
                else:
                    zch[ix,iy,nAirch]=fa(xch[ix,iy,nAirch],ych[ix,iy,nAirch])
            
            nmax=1000
            tol=1e-4
            dz=minDz
            for ii in range(nmax):
                zp=0
                z=zch[ix,iy,nAirch]
                
                
                for k in range(0,nAirch):
                    
                    if k<=corenz_air:
                        z=z-dz
                    else:
                        z=z-dz*coeff**(k-corenz_air)
                        zp=zp-dz*(k-corenz_air)*coeff**((k-corenz_air)-1)
                res=(bottomZ-z)**2
                resp=2*(bottomZ-z)*(-zp)

                if res<tol:
                    break
                elif ii==nmax-1:
                    print("not converged -Z Direction:",z,bottomZ,-elev)
                coeff=coeff-0.5*res/resp
            coeff_keep=coeff
            for k in range(0,nAirch):
                if k<=corenz_air:
                    coeff=1
                else:
                    coeff=coeff_keep
                zch[ix,iy,nAirch-(k+1)]=zch[ix,iy,nAirch-k]-dz*coeff**(k-corenz_air)
                xch[ix,iy,nAirch-(k+1)]=xch[ix,iy,nAirch]
                ych[ix,iy,nAirch-(k+1)]=ych[ix,iy,nAirch]
   
            
            #+Z方向
            kt=int(zPrecise[layer][1])
            cs=[]
            zs=[]
            if layer==0:
                for i in range(2):
                    for j in range(2):
                        c=numpy.zeros(2)
                        c[0]=xCoords[it+i%2]
                        c[1]=yCoords[jt+j%2]
                        v=za[it+i,jt+j,kt+1]
                        cs.append(copy.deepcopy(c))
                        zs.append(v)
                    
            
                fz = LinearNDInterpolator(cs,zs)
                bottomZ=fz(xch[ix,iy,nAirch],ych[ix,iy,nAirch])
            else:
                bottomZ=ft(xch[ix,iy,nAirch],ych[ix,iy,nAirch])
            coeff=1.0
            
            nmax=1000
            tol=1e-4
            
            for ii in range(nmax):
                zp=0
                z=zch[ix,iy,nAirch]

                for k in range(1,nzch-nAirch+1):
                    if k<=corenz:
                        z=z+dz
                    else:
                        z=z+dz*coeff**(k-1-corenz)
                        zp=zp+dz*(k-1-corenz)*coeff**(k-2-corenz)
                res=(bottomZ-z)**2
                resp=2*(bottomZ-z)*(-zp)
                if res<tol:
                    break
                elif ii==nmax-1:
                    print("not converged +Z Direction:",z,bottomZ)
                coeff=coeff-0.5*res/resp
            coeff_keep=coeff
            for k in range(1,nzch-nAirch+1):
                if k<=corenz:
                    coeff=1
                else:
                    coeff=coeff_keep
                zch[ix,iy,nAirch+k]=zch[ix,iy,nAirch+(k-1)]+dz*coeff**(k-1-corenz)
                xch[ix,iy,nAirch+k]=xch[ix,iy,nAirch]
                ych[ix,iy,nAirch+k]=ych[ix,iy,nAirch]
    print(zch[0,0,nzch])
    xchs.append(copy.deepcopy(xch))
    ychs.append(copy.deepcopy(ych))
    zchs.append(copy.deepcopy(zch))
    oceanBottomchs.append(copy.deepcopy(oceanBottomch))
#test
for layer in range(0,numSplitPrecise):  
    fn="test"+str(layer+1)+".csv"
    fi=open(fn,"w")
    for i in range(nxchs[layer]+1):
        for j in range(nychs[layer]+1):
            for k in range(nzchs[layer]+1):
                fi.write(str(xchs[layer][i,j,k])+","+str(ychs[layer][i,j,k])+","+str(zchs[layer][i,j,k])+"\n")
    fi.close()


elements={}
tmpElements={}
for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            dx=xa[ix+1,iy,iz]-xa[ix,iy,iz]
            dy=ya[ix,iy+1,iz]-ya[ix,iy,iz]
            dz=za[ix,iy,iz+1]-za[ix,iy,iz]
            element=Element()
            element.ID=str(iz).zfill(3)+str(iy).zfill(3)+str(ix).zfill(3)+"00" #root
            element.rootCoord[0]=xa[ix,iy,iz]
            element.rootCoord[1]=ya[ix,iy,iz]
            element.rootCoord[2]=za[ix,iy,iz]
            element.centerCoord[0]=dx/2+element.rootCoord[0]
            element.centerCoord[1]=dy/2+element.rootCoord[1]
            element.centerCoord[2]=dz/2+element.rootCoord[2]
            element.layer=0
            element.dx=dx
            element.dy=dy
            element.dz=dz
            element.ix=ix
            element.iy=iy
            element.iz=iz
            
            for i in range(8):
                element.nodes[i]=copy.deepcopy(element.rootCoord)
            element.nodes[1][0]=xa[ix+1,iy,iz]
            element.nodes[1][1]=ya[ix+1,iy,iz]
            element.nodes[1][2]=za[ix+1,iy,iz]
            
            element.nodes[2][0]=xa[ix+1,iy+1,iz]
            element.nodes[2][1]=ya[ix+1,iy+1,iz]
            element.nodes[2][2]=za[ix+1,iy+1,iz]
            
            element.nodes[3][0]=xa[ix,iy+1,iz]
            element.nodes[3][1]=ya[ix,iy+1,iz]
            element.nodes[3][2]=za[ix,iy+1,iz]
            
            element.nodes[4][0]=xa[ix,iy,iz+1]
            element.nodes[4][1]=ya[ix,iy,iz+1]
            element.nodes[4][2]=za[ix,iy,iz+1]
            
            element.nodes[5][0]=xa[ix+1,iy,iz+1]
            element.nodes[5][1]=ya[ix+1,iy,iz+1]
            element.nodes[5][2]=za[ix+1,iy,iz+1]
            
            element.nodes[6][0]=xa[ix+1,iy+1,iz+1]
            element.nodes[6][1]=ya[ix+1,iy+1,iz+1]
            element.nodes[6][2]=za[ix+1,iy+1,iz+1]
            
            element.nodes[7][0]=xa[ix,iy+1,iz+1]
            element.nodes[7][1]=ya[ix,iy+1,iz+1]
            element.nodes[7][2]=za[ix,iy+1,iz+1]
            
      
            if iz<nAir:
                element.rhoID=4
            elif (oceanBottom[ix,iy]+oceanBottom[ix+1,iy]+oceanBottom[ix+1,iy+1]+oceanBottom[ix,iy+1])/4.0<0 and -element.centerCoord[2]<0 and -element.centerCoord[2]>=oceanBottom[ix,iy]:
                element.rhoID=7
            else:
                element.rhoID=2
            

                        
            tmpElements[element.ID]=element
elements.update(tmpElements)

allPoints=numpy.zeros((nx+1,ny+1,nz+1,3))   
for ix in range(nx+1):
    for iy in range(ny+1):
        for iz in range(nz+1):
            allPoints[ix,iy,iz,0]=xa[ix,iy,iz]
            allPoints[ix,iy,iz,1]=ya[ix,iy,iz]
            allPoints[ix,iy,iz,2]=za[ix,iy,iz]
            
                

#make control points


xcps=[]
ycps=[]
zcps=[]
vcps=[]
for i in range(ncx):
    for j in range(ncy):
        for k in range(ncz):
            px=int(nxmin+(nxmax-nxmin)*i/(ncx-1))
            py=int(nymin+(nymax-nymin)*j/(ncy-1))
            pz=int(nzmin+(nzmax-nzmin)*k/(ncz-1))
            for element in elements.values():
                if element.ix==px and element.iy==py and element.iz==pz:
                    r=random.random()
                    minl=numpy.log10(minResisCp)
                    maxl=numpy.log10(maxResisCp)
                    v=minl+(maxl-minl)*r
                    vcps.append(v)
                    xcps.append(element.centerCoord[0])
                    ycps.append(element.centerCoord[1])
                    zcps.append(element.centerCoord[2])
                    print(element.ix,element.iy,element.iz,v)

UK = UniversalKriging3D(xcps,ycps,zcps,vcps,"linear",{'slope': factorKr*(minDx+minDy)/2, 'nugget': 0 })

   
for layer in range(1,numSplitPrecise+1):
    ii=layer-1
    tmpElements={}
    for parentElement in elements.values():
        if parentElement.layer==layer-1:
            flag=False
            if parentElement.ix>=xPrecise[ii][0] and parentElement.ix<=xPrecise[ii][1]  \
                    and parentElement.iy>=yPrecise[ii][0] and parentElement.iy<=yPrecise[ii][1] \
                    and parentElement.iz>=zPrecise[ii][0] and parentElement.iz<=zPrecise[ii][1]:
                flag=True
            if flag==False:
                continue
            parentElement.parent=True
            ix=parentElement.ix
            iy=parentElement.iy
            iz=parentElement.iz
            dx=parentElement.dx/2
            dy=parentElement.dy/2
            dz=(za[ix,iy,iz+1]-za[ix,iy,iz])
            for i in range(2):
                for j in range(2):
                    
                    element=Element()
                    element.placeInParents=copy.deepcopy(parentElement.placeInParents)
                    element.placeInParents.append(copy.deepcopy([i,j]))
                    element.ID=parentElement.ID+getBinaryValue(i,j)
                    
                    element.ix=parentElement.ix
                    element.iy=parentElement.iy
                    element.iz=parentElement.iz
                    
                    it=(element.ix-xPrecise[ii][0])*(2**layer)
                    for jj in range(len(element.placeInParents)):
                        it+=element.placeInParents[jj][0]*2**(ii-jj)
                    jt=(element.iy-yPrecise[ii][0])*(2**layer)
                    for jj in range(len(element.placeInParents)):
                        jt+=element.placeInParents[jj][1]*2**(ii-jj)
                    kt=element.iz-zPrecise[ii][0]
                    
                    it=int(it)
                    jt=int(jt)
                    kt=int(kt)

                    element.rootCoord[0]=xchs[ii][it,jt,kt]
                    element.rootCoord[1]=ychs[ii][it,jt,kt]
                    element.rootCoord[2]=zchs[ii][it,jt,kt]
                    
                    element.layer=layer
                    element.dx=dx #今はもう使っていない値なので適当
                    element.dy=dy
                    element.dz=dz
                    
                    
                    element.nodes[0][0]=xchs[ii][it,jt,kt]
                    element.nodes[0][1]=ychs[ii][it,jt,kt]
                    element.nodes[0][2]=zchs[ii][it,jt,kt]
                    
                    element.nodes[1][0]=xchs[ii][it+1,jt,kt]
                    element.nodes[1][1]=ychs[ii][it+1,jt,kt]
                    element.nodes[1][2]=zchs[ii][it+1,jt,kt]
                    
                    element.nodes[2][0]=xchs[ii][it+1,jt+1,kt]
                    element.nodes[2][1]=ychs[ii][it+1,jt+1,kt]
                    element.nodes[2][2]=zchs[ii][it+1,jt+1,kt]
                    
                    element.nodes[3][0]=xchs[ii][it,jt+1,kt]
                    element.nodes[3][1]=ychs[ii][it,jt+1,kt]
                    element.nodes[3][2]=zchs[ii][it,jt+1,kt]
                    
                    element.nodes[4][0]=xchs[ii][it,jt,kt+1]
                    element.nodes[4][1]=ychs[ii][it,jt,kt+1]
                    element.nodes[4][2]=zchs[ii][it,jt,kt+1]
                    
                    element.nodes[5][0]=xchs[ii][it+1,jt,kt+1]
                    element.nodes[5][1]=ychs[ii][it+1,jt,kt+1]
                    element.nodes[5][2]=zchs[ii][it+1,jt,kt+1]
                    
                    element.nodes[6][0]=xchs[ii][it+1,jt+1,kt+1]
                    element.nodes[6][1]=ychs[ii][it+1,jt+1,kt+1]
                    element.nodes[6][2]=zchs[ii][it+1,jt+1,kt+1]
                    
                    element.nodes[7][0]=xchs[ii][it,jt+1,kt+1]
                    element.nodes[7][1]=ychs[ii][it,jt+1,kt+1]
                    element.nodes[7][2]=zchs[ii][it,jt+1,kt+1]
                    
                    element.centerCoord=numpy.zeros(3)
                    oc=(oceanBottomchs[ii][it,jt]+oceanBottomchs[ii][it+1,jt]+oceanBottomchs[ii][it+1,jt+1]+oceanBottomchs[ii][it,jt+1])/4.0
                    for jj in range(8):
                        element.centerCoord+=element.nodes[jj]/8
                    if iz<nAir:
                        element.rhoID=4
                    elif oc<0 and -element.centerCoord[2]<0 and -element.centerCoord[2]>=oc:
                        element.rhoID=7
                    else:
                        element.rhoID=2
                        

                    tmpElements[element.ID]=element


    elements.update(tmpElements)

i=0
pxs=[]
pys=[]
pzs=[]
for element in elements.values():
    if element.rhoID==2:
        pxs.append(element.centerCoord[0])
        pys.append(element.centerCoord[1])
        pzs.append(element.centerCoord[2])
        element.rhoID=100+i
        i+=1
rhos,s=UK.execute("points",pxs,pys,pzs)
parser = argparse.ArgumentParser()   

parser.add_argument('taskID')    
args = parser.parse_args()   


f=open("calcData_"+args.taskID+".txt","w")
numOfCells=0

f.write("ELEMENTS\n")
f.write("NotRectangular\n")
for element in elements.values():
    rootCoord=numpy.zeros(3)
    for i in range(3):
        rootCoord[i]=element.rootCoord[i]
    ID=element.ID
    dx=element.dx
    dy=element.dy
    dz=element.dz
    rhoID=element.rhoID
    parent=element.parent
    
    string=str(ID)+" "+str(rhoID)+" "+str(parent)+"\n"
    f.write(string)
    for i in range(8):
        string=str(element.nodes[i][0])+" "+str(element.nodes[i][1])+" "+str(element.nodes[i][2])+"\n"
        f.write(string)
    
    if parent==False:
        numOfCells+=1
print("Num Of Cells:",numOfCells)        
f.write("END ELEMENTS\n")

f.write("PROPERTIES\n")
f.write("   PROPERTY 1\n")
f.write("       Resistivity 1\n")
f.write("       Type 0\n")
f.write("   END PROPERTY\n")
f.write("   PROPERTY 2\n")
f.write("       Resistivity 10\n")
f.write("       Type 0\n")
f.write("   END PROPERTY\n")
f.write("   PROPERTY 3\n")
f.write("       Resistivity 100\n")
f.write("       Type 0\n")
f.write("   END PROPERTY\n")
f.write("   PROPERTY 4\n")
f.write("       Resistivity 1000000\n")
f.write("       Type 1\n")
f.write("   END PROPERTY\n")
f.write("   PROPERTY 5\n")
f.write("       Resistivity 50\n")
f.write("       Type 0\n")
f.write("   END PROPERTY\n")
f.write("   PROPERTY 6\n")
f.write("       Resistivity 0.1\n")
f.write("       Type 2\n")
f.write("   END PROPERTY\n")
f.write("   PROPERTY 7\n")
f.write("       Resistivity 0.25\n")
f.write("       Type 2\n")
f.write("   END PROPERTY\n")

for i in range(len(rhos)):
    f.write("   PROPERTY "+str(100+i)+"\n")
    r=max(minResisCp, min(10**rhos[i],maxResisCp))
    f.write("       Resistivity "+str(r)+"\n")
    f.write("       Type 0\n")
    f.write("   END PROPERTY\n")
f.write("END PROPERTIES\n")
f.write("   BOUNDARY \n")
f.write("      omega \n")

df=(numpy.log10(maxFreq)-numpy.log10(minFreq))/(numOfFreq-1)
freqs=numpy.zeros(numOfFreq)
for i in range(numOfFreq):
    freqs[i]=10**(numpy.log10(minFreq)+df*i)
    

for ii in range(len(freqs)):
    string=str(2*numpy.pi*freqs[ii])
    f.write(string+"\n")

f.write(" end     omega \n")
f.write("   END BOUNDARY\n")

f.write(" invSettings\n")
f.write(" parameters\n")
f.write(" stepSize 0.01\n")
f.write(" paramN 1\n")
f.write(" lambdaMax 1000\n")
f.write(" lambdaMin 0.1\n")
f.write(" numoflambda 5\n")
f.write(" loosenFactor 1.2\n")
f.write(" decreaseFactor 0.5\n")
f.write(" UseGD true\n")
f.write(" toleranceIterativeSolver 1e-6\n")
f.write(" ToleranceIterativeSolverForAdjointEquation 1e-6\n")
f.write(" maxIterationBiCGSTAB 10000\n")
f.write(" UseIterativeSolver true\n")
f.write(" thresholdObjFunctionChange 0.0\n")
f.write(" UseDistanceInModelConstraint false\n")
f.write(" thresholdResistivityChange 0.01\n")
f.write(" minStep 0.001\n")
f.write(" modifyGradient true\n")
f.write(" safetyFactor 0.1\n")
f.write("outputInterval 50\n")
f.write("minibatches 1\n")
f.write(" ManualSettingFile  ./testMannual.txt\n")
f.write("SettingEachLambda\n")
f.write("data 10 0.01 0.0 101 0.003 0.003\n")
f.write("data 1 0.01 0.0 101 0.003 0.003\n")
f.write("data 0.1 0.01 0.0 101 0.003 0.003\n")
f.write("end SettingEachLambda\n")
f.write(" end Parameters\n")
f.write("ObsFiles\n")
f.write("impedanceFile ./obsDataImpedance.txt\n")
f.write("tipperFile  ./obsDataTipperFile.txt\n")
f.write("end obsfiles\n")
f.write("end invSettings\n")
