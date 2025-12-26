import os

ns=200
ID=1
for i in range(23,ns):
    os.system("mkdir case_"+str(ID)+"_"+str(i))
    os.system("copy obsDataImpedance.txt case_"+str(ID)+"_"+str(i))
    os.system("copy obsDataTipperFile.txt case_"+str(ID)+"_"+str(i))
    os.system("copy 15ArcSecondsElev.txt case_"+str(ID)+"_"+str(i))
    os.system("copy makeBoxData_MT_UncertaintyAnalysis.py case_"+str(ID)+"_"+str(i))
    os.system("copy mimalloc* case_"+str(ID)+"_"+str(i))
    os.system("copy FVMTINV.exe case_"+str(ID)+"_"+str(i))
    os.system("copy libiomp5md.dll case_"+str(ID)+"_"+str(i))
    os.chdir("case_"+str(ID)+"_"+str(i))
    os.system("python makeBoxData_MT_UncertaintyAnalysis.py "+str(ID)+"_"+str(i))
    os.system("FVMTINV.exe calcData_"+str(ID)+"_"+str(i)+".txt >info_"+str(ID)+"_"+str(i)+".txt")
    os.system("attrib -A *")
    os.system("attrib +A info*.txt")
    os.system("attrib +A Rho.vtk")
    os.system("attrib +A optimizeInfo.txt")
    os.system("attrib +A InitialRho.vtk")
    os.system("del /A:-A /F /Q .\\*.*")
    os.chdir("..")