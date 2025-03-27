import numpy
import copy
from scipy import interpolate
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
from pyproj import Transformer
import re
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

#====PARAMETERS FOR MESSHER=====================
numSplitPrecise=0
nx=66
ny=104
nz=46
nAir=10

thicknessAirLayer=10000
dxMax=1000000
dyMax=1000000
dzMax=100000
dzMaxInAir=1000000


corenx=20
coreny=40
corenz=5
corenz_air=2


minDx=1000
minDy=1000
minDz=100


x_coeff=1.3
y_coeff=1.3
z_coeff=1.2
z_coeffAir=1.3
topoData="15ArcSecondsElev.txt"

latc=-19.2306829236745
longc= 140.55983307719276

rect_epsg=3112 #for Australia

initResis=10
oceanResis=0.25
airResis=10**6


xPrecise=[numpy.zeros(2,dtype=int) for i in range(numSplitPrecise)]
yPrecise=[numpy.zeros(2,dtype=int) for i in range(numSplitPrecise)]
zPrecise=[numpy.zeros(2,dtype=int) for i in range(numSplitPrecise)]

# precnx=6
# precny=6

# xPrecise[0][0]=int(nx/2-precnx)
# xPrecise[0][1]=int(nx/2+precnx)
# yPrecise[0][0]=int(ny/2-precny)
# yPrecise[0][1]=int(ny/2+precny)
# zPrecise[0][0]=int(nAir-9)
# zPrecise[0][1]=int(nAir+28)


# precnx=4
# precny=4

# xPrecise[1][0]=int(nx/2-precnx)
# xPrecise[1][1]=int(nx/2+precnx)
# yPrecise[1][0]=int(ny/2-precny)
# yPrecise[1][1]=int(ny/2+precny)
# zPrecise[1][0]=int(nAir-7)
# zPrecise[1][1]=int(nAir+26)

# precnx=3
# precny=3

# xPrecise[2][0]=int(nx/2-precnx)
# xPrecise[2][1]=int(nx/2+precnx)
# yPrecise[2][0]=int(ny/2-precny)
# yPrecise[2][1]=int(ny/2+precny)
# zPrecise[2][0]=int(nAir-5)
# zPrecise[2][1]=int(nAir+24)

# precnx=2
# precny=2

# xPrecise[3][0]=int(nx/2-precnx)
# xPrecise[3][1]=int(nx/2+precnx)
# yPrecise[3][0]=int(ny/2-precny)
# yPrecise[3][1]=int(ny/2+precny)
# zPrecise[3][0]=int(nAir-4)
# zPrecise[3][1]=int(nAir+10)

#=====================================================

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

zCoords=[-thicknessAirLayer]
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



f_large,xmin_large,xmax_large,ymin_large,ymax_large=makeInterpFunc(topoData)

wgs84_epsg, rect_epsg = 4326, rect_epsg 

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
        tol=1e-6
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
        tol=1e-6
        
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
            # else:
                # p=[]
                # v=[]
                # if ix==0:
                    # if layer>0:
                        # for j in range(nychs[layer-1]+1):
                            
                            # p.append(ychs[layer-1][0,j,nAirchs[layer-1]])
                            # v.append(zchs[layer-1][0,j,nAirchs[layer-1]])
                    # else:
                        # for j in range(ny):
                            # p.append(ya[xPrecise[layer][0],j,nAir])
                            # v.append(za[xPrecise[layer][0],j,nAir])
                    # ftmp=interpolate.interp1d(p,v)
                    # print(numpy.min(numpy.array(p)),numpy.max(numpy.array(p)))
                    # print(ych[ix,iy,nAirch])
                    # zch[ix,iy,nAirch]=ftmp(ych[ix,iy,nAirch])
                # if ix==nxch:
                    # if layer>0:
                        # for j in range(nychs[layer-1]+1):

                            # p.append(copy.deepcopy(ychs[layer-1][nxchs[layer-1],j,nAirchs[layer-1]]))
                            # v.append(zchs[layer-1][nxchs[layer-1],j,nAirchs[layer-1]])
                    # else:
                        # for j in range(ny):
                            # p.append(copy.deepcopy(ya[xPrecise[layer][1]+1,j,nAir]))
                            # v.append(za[xPrecise[layer][1]+1,j,nAir])
                    # ftmp=interpolate.interp1d(p,v)
                    # zch[ix,iy,nAirch]=ftmp(ych[ix,iy,nAirch])
                # if iy==0:
                    # if layer>0:
                        # for j in range(nxchs[layer-1]+1):

                            # p.append(copy.deepcopy(xchs[layer-1][j,0,nAirchs[layer-1]]))
                            # v.append(zchs[layer-1][j,0,nAirchs[layer-1]])
                    # else:
                        # for j in range(nx):
                            # p.append(copy.deepcopy(xa[j,yPrecise[layer][0],nAir]))
                            # v.append(za[j,yPrecise[layer][0],nAir])
                    # ftmp=interpolate.interp1d(p,v)
                    # zch[ix,iy,nAirch]=ftmp(xch[ix,iy,nAirch])
                # if iy==nych:
                    # if layer>0:
                        # for j in range(nxchs[layer-1]+1):
                            # p.append(copy.deepcopy(xchs[layer-1][j,nychs[layer-1],nAirchs[layer-1]]))
                            # v.append(zchs[layer-1][j,nychs[layer-1],nAirchs[layer-1]])
                    # else:
                        # for j in range(nx):
                            # p.append(copy.deepcopy(xa[j,yPrecise[layer][1]+1,nAir]))
                            # v.append(za[j,yPrecise[layer][1]+1,nAir])

                    # ftmp=interpolate.interp1d(p,v)
                    # zch[ix,iy,nAirch]=ftmp(xch[ix,iy,nAirch])
            # #-Z方向
            
            corenz_air=0
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


                
for layer in range(1,numSplitPrecise+1):
    ii=layer-1
    # dx=xMax/(2**(layer))
    # dy=yMax/(2**(layer))
    # dz=zMax/(2**(layer))
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
            
            
            
            # pl=numpy.zeros((3,3,2,3))
            
            # xi=[]
            # yi=[]
            # zi=[]
            # p=[numpy.zeros(2) for i in range(4)]
            # for i in range(4):
                # xi.append(parentElement.nodes[i][0])
                # yi.append(parentElement.nodes[i][1])
                # zi.append(parentElement.nodes[i][2])
                # p[i][0]=parentElement.nodes[i][0]
                # p[i][1]=parentElement.nodes[i][1]
            # fm = interpolate.LinearNDInterpolator(p, zi)
            
            # xi=[]
            # yi=[]
            # zi=[]
            # p=[numpy.zeros(2) for i in range(4)]
            # for i in range(4):
                # xi.append(parentElement.nodes[i+4][0])
                # yi.append(parentElement.nodes[i+4][1])
                # zi.append(parentElement.nodes[i+4][2])
                # p[i][0]=parentElement.nodes[i][0]
                # p[i][1]=parentElement.nodes[i][1]
            # fp = interpolate.LinearNDInterpolator(p, zi)
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


f=open("calcData.txt","w")
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
f.write("   PROPERTY 2\n")
f.write("       Resistivity "+str(initResis)+"\n")
f.write("       Type 0\n")
f.write("   END PROPERTY\n")
f.write("   PROPERTY 4\n")
f.write("       Resistivity "+str(airResis)+"\n")
f.write("       Type 1\n")
f.write("   END PROPERTY\n")
f.write("   PROPERTY 7\n")
f.write("       Resistivity "+str(oceanResis)+"\n")
f.write("       Type 2\n")
f.write("   END PROPERTY\n")
f.write("END PROPERTIES\n")
f.write("   BOUNDARY \n")
f.write("      omega \n")


#===========PARAMETERS FOR FREQENCIES====================
freqs=[0.001,
0.0021544346900318843,
0.004641588833612777,
0.01,
0.021544346900318832,
0.046415888336127774,
0.1,
0.21544346900318823,
0.46415888336127775,
1.0,
2.154434690031882,
4.6415888336127775,
10.0,
21.54434690031882,
46.41588833612773,
100.0]

#=======================================================
for ii in range(len(freqs)):
    string=str(2*numpy.pi*freqs[ii])
    f.write(string+"\n")

f.write(" end     omega \n")
f.write("   END BOUNDARY\n")

f.write(" invSettings\n")
f.write(" parameters\n")
#=============PARAMETERS FOR INVERSION===================
f.write(" UseGD true\n")
f.write(" toleranceIterativeSolver 1e-6\n")
f.write(" maxIterationBiCGSTAB 2000\n")
f.write(" UseIterativeSolver true\n")
f.write(" UseDistanceInModelConstraint false\n")
f.write(" modifyGradient true\n")
f.write(" decreaseFactor 0.5\n")
f.write(" safetyFactor 0.1\n")
#f.write("outputInterval 10\n")
f.write("SettingEachLambda\n")
f.write("data 100 0.1 0.0 100 0.002 0.00\n")
f.write("data 10 0.1 0.0 100 0.002 0.00\n")
f.write("data 1 0.1 0.0 100 0.002 0.00\n")
f.write("data 0.1 0.1 0.0 100 0.002 0.00\n")
f.write("data 0.01 0.1 0.0 100 0.002 0.00\n")
f.write("data 0.001 0.1 0.0 100 0.002 0.00\n")
f.write("data 0.0001 0.1 0.0 100 0.002 0.00\n")
f.write("end SettingEachLambda\n")
f.write(" end Parameters\n")
f.write("ObsFiles\n")
f.write("impedanceFile ./obsDataImpedance.txt\n")
f.write("tipperFile  ./obsDataTipperFile.txt\n")
f.write("end obsfiles\n")
# f.write("initialSettings\n")
# f.write("initialResistivityFile Rho_0.010000_54.txt\n")
# f.write("initialDistortionFile DistortionForRestart_0.010000_54.txt\n")
# f.write("end initialSettings\n")
f.write("end invSettings\n")

