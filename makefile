CC = g++
FLAGS = -fopenmp -std=c++17   -O3 -mtune=native -march=native -mfpmath=both
INCB  = -I C:\boost\build\include\boost-1_88
INCE  = -I C:\Users\xxxxx\Desktop\MT_INV\lib\eigen-3.4.0
INCO  = -I C:\Users\xxxxx\Desktop\MT_INV\lib\optimlib\header_only_version
INCK  = -I C:\Users\xxxxx\Desktop\MT_INV\lib\kv-0.4.58
OUT   = MT_INV.exe
MT_INV: MT.o Analysis.o BiCGSafe.o Boundary.o DivergenceCorrection.o Element.o Function.o FineGrainedILU.o InitialDistData.o InitialResisData.o InvSettings.o LocationCalcSettings.o ObsData.o Output.o  Property.o ReadData.o Node.o UncertaintyAnalysis.o UnstructuredElement.o
	$(CC) -o $(OUT)  MT.o Analysis.o BiCGSafe.o Boundary.o DivergenceCorrection.o Element.o Function.o FineGrainedILU.o InitialDistData.o InitialResisData.o InvSettings.o LocationCalcSettings.o ObsData.o Output.o  Property.o ReadData.o Node.o UncertaintyAnalysis.o UnstructuredElement.o $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)

MT.o: MT.cpp
	$(CC) -c MT.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
Analysis.o: Analysis.cpp
	$(CC) -c Analysis.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
BiCGSafe.o: BiCGSafe.cpp
	$(CC) -c BiCGSafe.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
Boundary.o: Boundary.cpp
	$(CC) -c Boundary.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
DivergenceCorrection.o: DivergenceCorrection.cpp
	$(CC) -c DivergenceCorrection.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
Element.o: Element.cpp
	$(CC) -c Element.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
Function.o: Function.cpp
	$(CC) -c Function.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
FineGrainedILU.o: FineGrainedILU.cpp
	$(CC) -c FineGrainedILU.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
InitialDistData.o: InitialDistData.cpp
	$(CC) -c InitialDistData.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
InitialResisData.o: InitialResisData.cpp
	$(CC) -c InitialResisData.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
InvSettings.o: InvSettings.cpp
	$(CC) -c InvSettings.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
LocationCalcSettings.o: LocationCalcSettings.cpp
	$(CC) -c LocationCalcSettings.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
ObsData.o: ObsData.cpp
	$(CC) -c ObsData.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
Output.o: Output.cpp
	$(CC) -c Output.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
Property.o: Property.cpp
	$(CC) -c Property.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
ReadData.o: ReadData.cpp
	$(CC) -c ReadData.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
Node.o: Node.cpp
	$(CC) -c Node.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
UncertaintyAnalysis.o: UncertaintyAnalysis.cpp
	$(CC) -c UncertaintyAnalysis.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
UnstructuredElement.o: UnstructuredElement.cpp
	$(CC) -c UnstructuredElement.cpp $(INCB) $(INCE) $(INCO) $(INCK)  $(FLAGS)
	