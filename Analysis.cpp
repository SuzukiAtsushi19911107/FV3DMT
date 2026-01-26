/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#define OPTIM_ENABLE_EIGEN_WRAPPERS
//#define OPTIM_USE_OPENMP Comment out because openmp is used in each loop in the function Optimize()
#pragma once
#include "mimalloc_config.h"
#include "optim.hpp"
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseLU> 
#include <unsupported/Eigen/IterativeSolvers>
#include <iostream>
#include "Analysis.h"
#include "ConstantValues.h"
#include "Output.h"
#include <omp.h>
#include <time.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <kv/autodif.hpp>
#include <kv/complex.hpp>
#include "Property.h"
#include "InvSettings.h"
#include <iomanip>
#include <string>
#include "BiCGSafe.h"
#include "LocationCalcSettings.h"
#include "UncertaintyAnalysis.h"
#include <random>
#include "Eigen/SVD"
#include <random>
#include <filesystem>
#include <iostream>
#include "Functions.h"
#include "Fista_L1.hpp"

namespace fs = std::filesystem;
//#define MI_MALLOC_OVERRIDE
//#include <mimalloc.h>
//#include <mimalloc-new-delete.h>
//#include <mimalloc-override.h>
//#include <Eigen/PardisoSupport>
#include <unsupported/Eigen/src/IterativeSolvers/Scaling.h>
#define EIGEN_DONT_PARALLELIZE
using namespace std;
using namespace ConstantValues;
Analysis::Analysis::Analysis(ReadData::ReadData* readData) {
	elementsVector = readData->elementsVector;
	elements = readData->elements;
	for (auto itr = readData->properties.begin(); itr != readData->properties.end(); itr++) {
		properties[itr->first] = itr->second;
	}
	propertiesVector = readData->propertiesVector;
	boundary = readData->boundary;
	obsData = readData->obsData;
	invSettings = readData->invSettings;
	uncertaintyAnalysis = readData->uncertaintyAnalysis;
	output =readData->output;
	initialResistivityData = readData->initialResistivityData;
	initialDistortionData = readData->initialDistortionData;

	locationCalcSettings = readData->locationCalcSettings;
	locationData = readData->locationData;

	calcJustDataMisfit = readData->calcJustDataMisfit;

    //変数初期化
	maxResis = invSettings->maxResis;
	minResis = invSettings->minResis;
	paramLogNormalization = invSettings->paramLogNormalization;
	modelConstraintMax = invSettings->modelConstraintMax;
	modelConstraintMin = invSettings->modelConstraintMin;

	isInvertedDistortion = invSettings->isInvertedDistortion;

	useL1Norm = invSettings->useL1Norm;
	rateL1Norm = invSettings->rateL1Norm;

	useLogScaleInElement = invSettings->useLogScaleInterpolation;

	//params for FFT Sensitivity Analysis
	FFTSensitivityMode=readData->isFFTSensitivityMode;
	attenuation=readData->attenuation;
	Nx=readData->Nx;
	Ny=readData->Ny;
	Nz=readData->Nz;
	K=readData->K;
	cells_window=readData->cells_window;
	numEnsemble=readData->numEnsemble;

	minX=readData->minX;
	maxX=readData->maxX;
	minY=readData->minY;
	maxY=readData->maxY;
	minZ=readData->minZ;
	maxZ=readData->maxZ;

	epsR=readData->epsR;
	epsT=readData->epsT;
	eps_window=readData->eps_window;
	confidenceLevel1=readData->confidenceLevel1; //For line search to seek the solution within these values
	confidenceLevel2=readData->confidenceLevel2;
	orthogonalize = readData->orthogonalize;
	lambdaForFFT = readData->lambda;
	usePreviousResult = readData->usePreviousResult;
}
void Analysis::Analysis::CalcForward(bool isCalcInversionValues, bool isCalcJacobiMatrix,bool onePointMode) {
	Eigen::setNbThreads(1);
	if (isDirectSolver) {
		for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
			omega = boundary->omega[iOmega];

			//solver1 = new Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>;
			//solver2 = new Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>;

			for (int i = 0; i < 2; i++) {
				Hpolarization = i;
				cout << "Making Matrix.." << endl;
				time_t start_t = time(NULL);
				if (i == 0) {
					MakeMatrix(true, isCalcInversionValues);
				}
				else if (i == 1) {
					MakeMatrix(false, isCalcInversionValues);
				}
				if (isDirectSolver == false) {

				}
				time_t end_t = time(NULL);
				std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
				cout << "End Making Matrix." << endl;
				solve(iOmega, i);

				CalcE(i, iOmega);
			}
			CalcZ(iOmega);
			CalcT(iOmega);

			//for (int i = 0; i < numOfObsPointElements; i++) {
			//	cout << obsPointElements[i]->boundary << endl;
			//	cout << "Z:" << obsPointElements[i]->Z[iOmega] << endl;
			//}
			if (isCalcInversionValues) {
				ub::vector < kv::autodif<kv::complex<double>>> HTwoItr;
				ub::vector<kv::complex<double>> HVecUb(3 * 2 * numOfCalcElements);
				vector<Eigen::VectorXcd> Hvec{ 2 };
				Hvec[0].resize(3 * numOfCalcElements);
				Hvec[1].resize(3 * numOfCalcElements);
				for (int i = 0; i < 2; i++) {
					for (int j = 0; j < 3 * numOfCalcElements; j++) {
						HVecUb(i * 3 * numOfCalcElements + j) = kv::complex<double>(resultVector.coeff((2 * iOmega + i) * 3 * numOfCalcElements + j).real(), resultVector.coeff((2 * iOmega + i) * 3 * numOfCalcElements + j).imag());
						Hvec[i].coeffRef(j) = resultVector.coeff((2 * iOmega + i) * 3 * numOfCalcElements + j);
					}
				}
				CalcDZDHElements(&HVecUb, iOmega);
				CalcDTDHElements(iOmega);
				if (isCalcJacobiMatrix == false) {
					std::cout << "Calculating Lambda for Gradient Vector." << std::endl;
					CalcLambda(iOmega);
					//delete solver1; //to prevent large memory use 
					//delete solver2;
					std::cout << "End of Calculating Lambda for Gradient Vector." << std::endl;
				}
				ub::vector< kv::complex<double>> rhoVecUb(numOfCalcElements);
				ub::vector< complex<double>> rhoVec(numOfCalcElements);
				for (int i = 0; i < numOfCalcElements; i++) {
					rhoVecUb(i) = calcElementsVector[i]->resistivity;
					rhoVec(i) = calcElementsVector[i]->resistivity;
				}

				CalcDZDRhoElements(&rhoVecUb, &HVecUb, iOmega);
				std::cout << "End of calculating dZdRho." << std::endl;
				if (isCalcJacobiMatrix == false) {
					CalcLambdaDRDRho(&rhoVec, &Hvec,iOmega);
					std::cout << "End of calculating LambdaDRdRho." << std::endl;
				}
				if (isCalcJacobiMatrix == true) {
					CalcJacobian(iOmega);
				}
			}
			if (!isCalcInversionValues) {
				//std::cout << "Output files.." << std::endl;
				//output->VTKFileOputput(omega, &elements);
				//output->VTKFileOputput(omega, &elements,"PHI");
				//output->VTKFileOputput(omega, &elements,"H");
				//output->VTKFileOputput(omega, &elements,"E");

				//output->TxtOutputAppRho(omega, &calcElements);
				output->AppRhoOutputSurface(omega, &elements);
				//output->PhiOutputSurface(omega, &elements);
				output->TipperOutputSurface(iOmega, omega, &elements);
				//output->VTKFileOputput(boundary->omega[iOmega], &elements, "H");
				//output->VTKFileOputput(boundary->omega[iOmega], &elements, "E");
			}
			ClearHAndE();
		}
	}
	else {
		time_t start_t = time(NULL);
		cout << "Update Matrix.." << endl;
		bool isNeededGradient = false;
		omega = boundary->omega[0];
		MakeMatrix(true, isCalcInversionValues);

		

		time_t end_t = time(NULL);
		cout << "End Update Matrix." << endl;
		std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
		Eigen::setNbThreads(1);
		if (invSettings->initialGuessFile != "None") {
			ReadInitialGuess(invSettings->initialGuessFile, resultVector);
		}

#pragma omp parallel for
		for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
			//solver1 = new Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>;
			//solver2 = new Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>;

			Solve_iterative(iOmega, omp_get_thread_num());

				
			
			
		}

		if (invSettings->InitialGuessOutputFile!="None") {
			output->OutputInitialGuessFile(invSettings->InitialGuessOutputFile, resultVector);
		}

		Eigen::setNbThreads(1);
		for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
			omega = boundary->omega[iOmega];

			SetH(iOmega);
			CalcE(0, iOmega);
			CalcE(1, iOmega);
			CalcZ(iOmega);
			CalcT(iOmega);

			if (!isCalcInversionValues && !FFTSensitivityMode) {
				output->TxtOutputAppRho(boundary->omega[iOmega], &calcElements);
				output->AppRhoOutputSurface(boundary->omega[iOmega], &elements);
				output->PhiOutputSurface(boundary->omega[iOmega], &elements);
				output->TipperOutputSurface(iOmega, boundary->omega[iOmega], &elements);
				//output->VTKFileOputput(boundary->omega[iOmega], &elements, "H");
				//output->VTKFileOputput(boundary->omega[iOmega], &elements, "E");
			}
		}
		for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
			if (isCalcInversionValues) {
				ub::vector < kv::autodif<kv::complex<double>>> HTwoItr;
				ub::vector<kv::complex<double>> HVecUb(3 * 2 * numOfCalcElements);
				vector<Eigen::VectorXcd> Hvec{ 2 };
				Hvec[0].resize(3 * numOfCalcElements);
				Hvec[1].resize(3 * numOfCalcElements);
				for (int i = 0; i < 2; i++) {
					for (int j = 0; j < 3 * numOfCalcElements; j++) {
						HVecUb(i * 3 * numOfCalcElements + j) = kv::complex<double>(resultVector.coeff((2 * iOmega + i) * 3 * numOfCalcElements + j).real(), resultVector.coeff((2 * iOmega + i) * 3 * numOfCalcElements + j).imag());
						Hvec[i].coeffRef(j) = resultVector.coeff((2 * iOmega + i) * 3 * numOfCalcElements + j);
					}
				}
				CalcDZDHElements(&HVecUb, iOmega);
				CalcDTDHElements(iOmega);
			}
		}
		if (calcJustDataMisfit) {
			std::cout<<"calcJustDataMisfit is True, so calc will exit after output data!!!" << endl;
			CalcDataMisfit();
			double tmp=weightRoughening * CalcRoughningMatrixPenalty();
			if (isInvertedDistortion) {
				CalcConstraintDistortionTerm();
				tmp += weightRougheningForDistortion * constraintDistortionTerm;
				
			}
			std::cout << "Data Misfit:" << dataMisfit << std::endl;
			std::cout << "Objective Function:" << dataMisfit + tmp << std::endl;
			std::cout << "RMS:"<<std::pow(dataMisfit / numOfObsData, 0.5) << endl;
			exit(0);
		}
		Eigen::setNbThreads(1);

#pragma omp parallel for
		for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
			if (isCalcInversionValues) {
				std::cout << "Calculating Lambda for Gradient Vector." << std::endl;
				CalcLambda(iOmega, omp_get_thread_num(), onePointMode);

				std::cout << "End of Calculating Lambda for Gradient Vector." << std::endl;
			}
		}
		if (isCalcInversionValues) {
			lambdaDRDRho.setZero();
		}

		Eigen::setNbThreads(1);
		
		
		for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
			if (isCalcInversionValues) {
				ub::vector < kv::autodif<kv::complex<double>>> HTwoItr;
				ub::vector<kv::complex<double>> HVecUb(3 * 2 * numOfCalcElements);
				vector<Eigen::VectorXcd> Hvec{ 2 };
				Hvec[0].resize(3 * numOfCalcElements);
				Hvec[1].resize(3 * numOfCalcElements);
				for (int i = 0; i < 2; i++) {
					for (int j = 0; j < 3 * numOfCalcElements; j++) {
						HVecUb(i * 3 * numOfCalcElements + j) = kv::complex<double>(resultVector.coeff((2 * iOmega + i) * 3 * numOfCalcElements + j).real(), resultVector.coeff((2 * iOmega + i) * 3 * numOfCalcElements + j).imag());
						Hvec[i].coeffRef(j) = resultVector.coeff((2 * iOmega + i) * 3 * numOfCalcElements + j);
					}
				}

				ub::vector< kv::complex<double>> rhoVecUb(numOfCalcElements);
				ub::vector< complex<double>> rhoVec(numOfCalcElements);
				for (int i = 0; i < numOfCalcElements; i++) {
					rhoVecUb(i) = calcElementsVector[i]->resistivity;
					rhoVec(i) = calcElementsVector[i]->resistivity;
				}

				CalcDZDRhoElements(&rhoVecUb, &HVecUb, iOmega);
				std::cout << "End of calculating dZdRho." << std::endl;
				if (isCalcJacobiMatrix == false) {
					CalcLambdaDRDRho(&rhoVec, &Hvec,iOmega);
					std::cout << "End of calculating LambdaDRdRho." << std::endl;
				}

			}

			//std::cout << ("Output files..") << std::endl;
			//output->VTKFileOputput(omega, &elements);
			//output->VTKFileOputput(omega, &elements,"PHI");
			//output->VTKFileOputput(omega, &elements,"H");
			//output->VTKFileOputput(omega, &elements,"E");


			
		}
		ClearHAndE();
	}

	output->ImpedanceOutputSurface(boundary->omega, &elements);
	output->TipperOutputSurface(boundary->omega, &elements);

	Eigen::setNbThreads(1);
}
void Analysis::Analysis::SetH(int iOmega) {
	for (int i = 0; i < numOfCalcElements; i++) {
		calcElementsVector[i]->SetH(resultVector, iOmega, numOfCalcElements);
	}
}
void Analysis::Analysis::RunAnalysis() {

	std::cout << ("Initialize Data") << std::endl;
	Initialize();
	std::cout << ("Initialization End") << std::endl;
	
		
	//SetSameResistivityToBoundaryCell();

	CalcForward(false);
		

	ClearZ();


	
}
void Analysis::Analysis::Initialize() {
	SetNumOfCalcElementsAndCalcElementsAndElementsVector();
	cout << "Number of calculated elements: " << numOfCalcElements << endl;
	AssociationPropertiesToElements();

	SetLayerOfElements();

	SetNeighborElements();

	SetInitialResistivityFromFile();

	SetTransitionZoneElements();
	SetNotBoundaryElements();

	CheckElements();

	SetSameResistivityToBoundaryCell(); //set master/slave

	if (locationCalcSettings->isCalc) {
		SetLocationDataToElement();
	}

	output->RhoOutput(&elements, "InitialRho.vtk", true);
	output->TxtOutputResistivity(&elements, "InitialResis.txt");

	SetObsDataToElement();

	CountObsData();

	CalcNumOfRelatedCalcVariablesElements();

	SetCalcDivGradCells();

	SetLogScale();
	std::cout << "Calc Surface Resistivity.." << std::endl;;
	CalcSurfaceResistivityElements();
	std::cout << "End Calc Surface Resistivity.." << std::endl;

	std::cout << "Calc SumNCrossRhoRotHdSElements.." << std::endl;
	CalcSumNCrossRhoRotHdSElements();
	std::cout << "End Calc SumNCrossRhoRotHdSElements.." << std::endl;
	CalcNumOfDirichletConditionCells();
	std::cout << "Calc DivergenceCorrection.." << std::endl;
	time_t start_t = time(NULL);

	divergenceCorrection = new DivergenceCorrection::DivergenceCorrection(numOfCalcElements, reservedVector_rough);
	std::cout << "     Calc Matrix.." << std::endl;
	divergenceCorrection->initialize(&elements, &calcElementsVector);
	time_t end_t = time(NULL);
	std::cout << "     Calculation Time:" << end_t - start_t << " Seconds." << endl;
	start_t = time(NULL);
	std::cout << "     Calc Laplacian.." << std::endl;
	CalcDivergenceElements();
	end_t = time(NULL);
	std::cout << "     Calculation Time:" << end_t - start_t << " Seconds." << endl;
	std::cout << "End Calc DivergenceCorrection.." << std::endl;

	CalcNumOfReserveNeededInRow();
	//Initiialze globalMatrix

	globalMatrixNoOmegaTerm = new Eigen::SparseMatrix< double, Eigen::RowMajor>;
	globalMatrixNoOmegaTerm->resize(3 * numOfCalcElements, 3 * numOfCalcElements);
	globalMatrixNoOmegaTerm->reserve(reservedVector);

	/*resisMatDotSumDivHdSMat = new Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>;
	resisMatDotSumDivHdSMat->resize(3 * numOfCalcElements, 3 * numOfCalcElements);
	resisMatDotSumDivHdSMat->reserve(Eigen::VectorXi::Constant(3 * numOfCalcElements, 100));*/
	globalMatrixNoOmegaTermAdjoint = new Eigen::SparseMatrix< double , Eigen::RowMajor>;
	globalMatrixNoOmegaTermAdjoint->resize(3 * numOfCalcElements, 3 * numOfCalcElements);





	iterativeSolverVector.resize(2* boundary->omega.size());
	//globalVectorEachThread.resize(omp_get_max_threads());
	//globalVectorAdjointEachThread.resize(omp_get_max_threads());


	CountNumOfAirCells();
	SetSolverOrder();


	for (int i = 0; i < 2*boundary->omega.size(); i++) {
		bool flg;
		flg = false;
		/*if (i < boundary->omega.size()) {
			flg = false;
			iterativeSolverVector[i] = new BiCGSafe::BiCGSafe(2, 3 * numOfCalcElements, flg);
		}
		else {
			flg = false;
			iterativeSolverVector[i] = new BiCGSafe::BiCGSafe(2, 3 * numOfCalcElements, flg);
		}*/
		iterativeSolverVector[i] = new BiCGSafe::BiCGSafe(2, 3 * numOfCalcElements);

		iterativeSolverVector[i]->m_maxIteration = invSettings->maxIterationBiCGSafe;
		iterativeSolverVector[i]->m_relSolTol = invSettings->toleranceIterativeSolver;
		if (invSettings->toleranceIterativeSolverAdjoint < 0.0) {
			iterativeSolverVector[i]->m_relSolTolForAdjoint = iterativeSolverVector[i]->m_relSolTol;
		}
		iterativeSolverVector[i]->m_relSolTolForAdjoint = invSettings->toleranceIterativeSolverAdjoint;

		iterativeSolverVector[i]->calcElementsVector = &calcElementsVector;
		iterativeSolverVector[i]->solverToOriginal = &solverOrderToOriginalOrder;

		
	}

	//if (useMultGrid) {
	//	SetMultiGridPreconditioner();
	//}

	reservedVectorReordering.resize( 3 * numOfCalcElements );
	reservedVector_roughReordering.resize(3 * numOfCalcElements);
	for (int j = 0; j < 3 * numOfCalcElements; j++) {
		reservedVectorReordering[(j % 3) * numOfCalcElements + int(j / 3)] = reservedVector[j];
		reservedVector_roughReordering[(j % 3) * numOfCalcElements + int(j / 3)] = reservedVector_rough[j];
	}

	int parallelSolveSize = std::min(omp_get_max_threads(), int(boundary->omega.size()));
	//globalVectorEachThread.resize(parallelSolveSize);
	//globalVectorAdjointEachThread.resize(parallelSolveSize); 

	//for (int i = 0; i < parallelSolveSize; i++) {
	//	
	//	
	//	

	//	globalVectorEachThread[i].resize(2);
	//	globalVectorEachThread[i][0].resize( 3 * numOfCalcElements );
	//	globalVectorEachThread[i][1].resize(3 * numOfCalcElements);
	//	globalVectorEachThread[i][0].setZero();
	//	globalVectorEachThread[i][1].setZero();
	//	//globalVectorEachThread[i][0] = Eigen::VectorXcd{3 * numOfCalcElements};
	//	//globalVectorEachThread[i][1] = Eigen::VectorXcd{ 3 * numOfCalcElements };


	//	globalVectorAdjointEachThread[i].resize(2);
	//	//globalVectorAdjointEachThread[i][0] = Eigen::VectorXcd{ 3 * numOfCalcElements };
	//	//globalVectorAdjointEachThread[i][1] = Eigen::VectorXcd{ 3 * numOfCalcElements };
	//	globalVectorAdjointEachThread[i][0].resize(3 * numOfCalcElements);
	//	globalVectorAdjointEachThread[i][1].resize(3 * numOfCalcElements);
	//	globalVectorAdjointEachThread[i][0].setZero();
	//	globalVectorAdjointEachThread[i][1].setZero();
	//}

	globalVector = new Eigen::SparseMatrix<std::complex< double >, Eigen::ColMajor>;
	globalVector->resize( 3 * numOfCalcElements, 1 );
	globalVector->setZero();
	globalVector->makeCompressed();
	globalVector->reserve(numOfDirichletConditionCells);
	//Set resultVector;
	resultVector.resize(2 * boundary->omega.size() * 3 * numOfCalcElements);
	resultVector.setZero();
	resultAdjointVector.resize(2 * boundary->omega.size() * 3 * numOfCalcElements);
	resultAdjointVector.setZero();
	//Set result_pre;
	//resultVector_pre.resize(2*boundary->omega.size()*3*numOfCalcElements);
	//resultVector_pre.setZero();
	//resultAdjointVector_pre.resize(2 * boundary->omega.size() * 3 * numOfCalcElements);
	//resultAdjointVector_pre.setZero();
	

	//for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
	//	omega = boundary->omega[iOmega];
	//	vector<vector<vector<complex<double>>>> tmpResult = Calc1D(); //Initial Guess From 1D Calc
	//	for (int i = 0; i < numOfCalcElements; i++) {
	//		Element::Element* element = calcElementsVector[i];
	//		resultVector_pre.coeffRef((2*iOmega)*3*numOfCalcElements + 3 * element->calcID)
	//				= tmpResult[element->IDX][element->IDY][element->IDZ];

	//		resultVector_pre.coeffRef((2 * iOmega + 1) * 3 * numOfCalcElements + 3 * element->calcID + 1)
	//			= tmpResult[element->IDX][element->IDY][element->IDZ];

	//	}
	//}
	//resultVector_pre.setZero();
	

	//globalMatrixForParallel.resize(numOfCalcElements);
	//globalVectorForParallel.resize(numOfCalcElements);
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	globalMatrixForParallel[i] = Eigen::SparseMatrix < std::complex<double >, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
	//	//globalMatrixTmp[i].resize(3, 3 * numOfCalcElements);
	//	globalMatrixForParallel[i].reserve(Eigen::VectorXi::Constant(3, reservedVector[3*i]));
	//	//globalMatrixTmp[i].makeCompressed();
	//	//globalMatrixTmp[i].reserve(243);
	//	globalVectorForParallel[i].resize(3);
	//	globalVectorForParallel[i].setZero();
	//}

	//Set H and E Vector Size
	for (int i = 0; i < numOfCalcElements; i++) {
		calcElementsVector[i]->InitializeHAndEAndZ(boundary->omega.size());
	}
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	calcElementsVector[i]->H.resize(2);
	//	calcElementsVector[i]->E.resize(2);
	//	for (int ii = 0; ii < 2; ii++) {
	//		calcElementsVector[i]->H[ii].resize(2);
	//		calcElementsVector[i]->E[ii].resize(2);
	//		calcElementsVector[i]->H[ii].setZero();
	//		calcElementsVector[i]->E[ii].setZero();
	//	}
	//}
	////Set Z Vector Size
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	calcElementsVector[i]->Z.resize(boundary->omega.size());
	//	calcElementsVector[i]->dZdH.resize(boundary->omega.size());
	//	calcElementsVector[i]->dZdRho.resize(boundary->omega.size());
	//	for (int ii = 0; ii < boundary->omega.size(); ii++) {
	//		calcElementsVector[i]->dZdH[ii].resize(2, 2);
	//		calcElementsVector[i]->dZdRho[ii].resize(2, 2);
	//	}
	//}

	//lambda.resize(boundary->omega.size() * 2 * 2* 3 * numOfCalcElements);
	lambdaEachOmega.resize(boundary->omega.size());
	for (int i = 0; i < boundary->omega.size(); i++) {
		lambdaEachOmega[i] = new Eigen::VectorXcd(2 * 3 * numOfCalcElements);
		lambdaEachOmega[i]->setZero();
	}
	SearchRelatedCalcElements();
	SetInvertedElements();
	rougheningMatrix = new Eigen::SparseMatrix<double, Eigen::RowMajor>;
	rougheningMatrix->resize(6*numOfInvertedResistivityElements, numOfInvertedResistivityElements);
	//rougheningMatrix->resize(27*numOfInvertedResistivityElements, numOfInvertedResistivityElements);
	//rougheningMatrix->setZero();
	CalcRougheningMatrix();
	//SetDKDRhoElements();
	//CalcDKDRhoElements();
	//lambdaDRDRho =  Eigen::VectorXcd{ numOfInvertedResistivityElements };
	lambdaDRDRho.resize(numOfInvertedResistivityElements);
	lambdaDRDRho.setZero();
	//dRdRho = new Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>;
	//dRdRho->resize( boundary->omega.size() * 2 * 3 * numOfCalcElements,numOfInvertedResistivityElements );
	//dRdRho->reserve(Eigen::VectorXi::Constant(boundary->omega.size() * 2 * 3 * numOfCalcElements,13));//this valuable is used as uncompressed mode for parallelization
	//dRdRho->reserve(243);
	dDataMisfitDRho.resize(numOfInvertedResistivityElements);
	dDataMisfitDRho.setZero();

	SetOutput();

	coeffsForModifiedGradient.resize(numOfInvertedResistivityElements);
	elementsForModifiedGradient.resize(numOfInvertedResistivityElements);
	if (isInvertedDistortion) {
		dJdRho.resize(numOfInvertedResistivityElements + 4* numOfObsImpedanceElements);
		
	}
	else {
		dJdRho.resize(numOfInvertedResistivityElements);
	}
	dUdRho_output.resize(numOfInvertedResistivityElements,2);
	dUdRho_output.setZero();
	dJdRho.setZero();

	//jacobian = new Eigen::MatrixXd;
	//jacobian->resize(numOfObsData, numOfInvertedResistivityElements);
	//jacobian->setZero();


	dRhoDParam.resize(numOfInvertedResistivityElements);
	dRhoDParam.setZero();


	

	useImpedanceDataArray.resize(numOfImpedanceDataset);
	useImpedanceDataArray.setOnes();
	useTipperDataArray.resize(numOfTipperDataset);
	useTipperDataArray.setOnes();

	for (int i = 0; i < numOfCalcElements; i++) {
		calcElementsVector[i]->InitializeDistortionMatrix();
	}
	SetInitialDistortionFromFile();
	dDataMisfitDDistortionParam.resize(4 * numOfObsImpedanceElements);
	invSettings->eps_distortionConstraint = 4 * numOfObsImpedanceElements * std::pow(invSettings->epsRatio , 2.0);
	//CalcResistivityAtTransitionElements();
	if (isDirectSolver) {
		m_res.resize(1);
		m_res[0].resize(2);
		dJdH.resize(1);
		lambdaDRDRhoEachThread.resize(1);
		dZdHCalc.resize(1);
	}
	else {
		m_res.resize(omp_get_max_threads());
		for (int i = 0; i < omp_get_max_threads(); i++) {
			m_res[i].resize(2);
		}
		dJdH.resize(omp_get_max_threads());
		lambdaDRDRhoEachThread.resize(omp_get_max_threads());
		dZdHCalc.resize(omp_get_max_threads());
		
	}
	for (int i = 0; i < dZdHCalc.size(); i++) {
		dZdHCalc[i].resize(2, 2);
		for (int ii = 0; ii < 2; ii++) {
			for (int jj = 0; jj < 2; jj++) {
				dZdHCalc[i](ii, jj).resize(0, 2 * 3 * numOfCalcElements);
				dZdHCalc[i](ii, jj).reserve(200);

			}
		}
	}
	
	dZdRhoCalc.resize(2, 2);
	for (int ii = 0; ii < 2; ii++) {
		for (int jj = 0; jj < 2; jj++) {
			dZdRhoCalc(ii, jj).resize(0, numOfInvertedResistivityElements);
			dZdRhoCalc(ii, jj).reserve(100);

		}
	}

}
void Analysis::Analysis::CalcSurfaceResistivityElements() {
	
	SetSameLayerElements();

	for (int iLayer = 0; iLayer <= maxLayer; iLayer++) { //this is needed because larger layer elements need the result of lower one.
		for (int i = 0; i < sameLayerElementsVector[iLayer].size(); i++) {
			Element::Element* element = sameLayerElementsVector[iLayer][i];
			element->CalcSurfaceResistivity(&elements, &calcElementsVector, numOfCalcElements);
		}
	}

//#pragma omp parallel for
//	for (int i = 0; i < numOfCalcElements; i++) {
//		Element::Element* element = calcElementsVector[i];
//		element->CalcSurfaceResistivity(&elements,&calcElementsVector, numOfCalcElements, &modelNormalizationCoeff);
//		
//	}
}
void Analysis::Analysis::solve(int iOmega,int itr) {
	cout<<"Direct Solver in Non Intel MKL Version is not Implemented."<<endl;
		exit(1);		
}
void Analysis::Analysis::Solve_iterative(int iOmega,int threadID) {


	//Eigen::SparseMatrix<double, Eigen::RowMajor> prunedMat{ 3 * numOfCalcElements,3 * numOfCalcElements };
	//prunedMat = *globalMatrixNoOmegaTerm;
	//prunedMat.prune(0.0);
	//prunedMat.makeCompressed();

	//divergence correction in subsurface, now which is included in makeMatrix();


	//Eigen::SparseMatrix<double, Eigen::RowMajor> factorMat;
	//factorMat.resize(numOfCalcElements, numOfCalcElements);
	//factorMat.reserve(Eigen::VectorXi::Constant(numOfCalcElements, 1));
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	if (!calcElementsVector[i]->calcGradDivOperationElement && calcElementsVector[i]->boundary == "NOT_BOUNDARY") {
	//		//double ave = 0;
	//		double resis;
	//		if (calcElementsVector[i]->property->type == Property::Property::AIR) {
	//			/*resis = 0.0;
	//			for (int j = 0; j < 6; j++) {

	//				resis += (*calcElementsVector[i]->resistivitySurface)[j] / 6.0;
	//			}*/
	//			resis = 0.0;
	//			for (int j = 0; j < 6; j++) {
	//				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*(calcElementsVector[i]->resistivitySurfaceCoeff[j]), 0); it; ++it) {
	//					if (resis < calcElementsVector[it.col()]->resistivity) {
	//						resis = calcElementsVector[it.col()]->resistivity;
	//					}
	//				}
	//				//ave += (*calcElementsVector[i]->resistivitySurface)[j];
	//			}
	//			calcElementsVector[i]->factorForDivCorr = resis;
	//		}
	//		else {
	//			resis = 1e30;
	//			for (int j = 0; j < 6; j++) {
	//				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*(calcElementsVector[i]->resistivitySurfaceCoeff[j]), 0); it; ++it) {
	//					if (resis > calcElementsVector[it.col()]->resistivity) {
	//						resis = calcElementsVector[it.col()]->resistivity;
	//					}
	//				}
	//				//ave += (*calcElementsVector[i]->resistivitySurface)[j];
	//			}
	//			calcElementsVector[i]->factorForDivCorr = invSettings->safetyFactor * resis;// *min(1.0, sqrt(boundary->omega[iOmega]));
	//		}

	//		/*resis = 0.0;
	//		for (int j = 0; j < 6; j++) {
	//			resis += (*calcElementsVector[i]->resistivitySurface)[j]/6.0;
	//		}*/
	//		//double mid = exp((log(calcElementsVector[i]->resistivity) + log(boundary->omega[iOmega] * mu)) / 2.0);




	//		factorMat.coeffRef(i, i) = -calcElementsVector[i]->factorForDivCorr;
	//		/*factorMat.coeffRef(3*i, 3*i) = -calcElementsVector[i]->factorForDivCorr;
	//		factorMat.coeffRef(3 * i + 1, 3 * i + 1) = -calcElementsVector[i]->factorForDivCorr;
	//		factorMat.coeffRef(3 * i +2, 3 * i + 2) = -calcElementsVector[i]->factorForDivCorr;*/
	//		//*minResis;// calcElementsVector[i]->resistivity;
	//	}
	//}

	//precond

	//Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> precondMat{ 3 * numOfCalcElements,3 * numOfCalcElements };
	//
	//vector < Eigen::Triplet<complex<double>>> precondMatTriplet;
	//precondMatTriplet.reserve(3 * numOfCalcElements * 90);

	//for (int i = 0; i < 3 * numOfCalcElements; i++) {
	//	Element::Element* element = calcElementsVector[int(solverOrderToOriginalOrder[i] / 3)];
	//	double dv = element->dv;
	//	for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(*globalMatrixNoOmegaTerm, i); it; ++it) {


	//		int solRow = i;
	//		int solCol = it.col();

	//		int oriRow = solverOrderToOriginalOrder[solRow];
	//		int oriCol = solverOrderToOriginalOrder[solCol];


	//		/*if (solRow < numOfCalcElements && solCol >= numOfCalcElements) {
	//			continue;
	//		}
	//		else if ((solRow >= numOfCalcElements && solRow < 2*numOfCalcElements) && (solCol < numOfCalcElements || solCol >= 2*numOfCalcElements)) {
	//			continue;
	//		}
	//		else if (solRow >= 2*numOfCalcElements && solCol < 2*numOfCalcElements ) {
	//			continue;
	//		}*/

	//		Eigen::Triplet<complex<double>> val(solRow, solCol,
	//			globalMatrixNoOmegaTerm->coeff(solRow, solCol) +
	//			double(int(element->boundary == "NOT_BOUNDARY" && !element->calcGradDivOperationElement && oriRow == oriCol)) *
	//			complex<double>(0, boundary->omega[iOmega] * mu * dv) +

	//			double(int(element->boundary == "NOT_BOUNDARY" && element->calcGradDivOperationElement && oriRow == oriCol)) *
	//			complex<double>(0, 1.0 / element->resistivity * boundary->omega[iOmega] * mu * dv));

	//		precondMatTriplet.push_back(val);
	//	}

	//}



	//precondMat.setFromTriplets(precondMatTriplet.begin(), precondMatTriplet.end());
	//precondMat.pruned();

	


	//precondMat = diagMat * precondMat;
	//iterativeSolverVector[iOmega]->precond.compute(precondMat);

	// end precond


	
	//Eigen::SparseMatrix<double, Eigen::RowMajor> globalMatrixToBeSolvedReal;
	//globalMatrixToBeSolvedReal.resize(3 * numOfCalcElements, 3 * numOfCalcElements);
	Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> globalMatrixToBeSolvedImag;
	globalMatrixToBeSolvedImag.resize(3 * numOfCalcElements, 3 * numOfCalcElements);
	//globalMatrixToBeSolved.reserve(Eigen::VectorXi::Constant(4*numOfCalcElements,200));

	//vector < Eigen::Triplet<double>> globalMatrixToBeSolvedRealTriplet;
	//globalMatrixToBeSolvedRealTriplet.reserve(globalMatrixNoOmegaTerm->nonZeros());
	vector < Eigen::Triplet<complex<double>>> globalMatrixToBeSolvedImagTriplet;
	globalMatrixToBeSolvedImagTriplet.reserve(3 * numOfCalcElements);
	for (int i = 0; i < 3 * numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[int(i / 3)];
		double dv = element->dv;

		int oriRow = i;
		int oriCol = i;

		int solRow = originalOrderToSolverOrder[oriRow];
		int solCol = originalOrderToSolverOrder[oriCol];


		/*Eigen::Triplet<double> valR(solRow, solCol,
			globalMatrixNoOmegaTerm->coeff(oriRow, oriCol));*/

		Eigen::Triplet<complex<double>> valI(solRow, solCol,
			(double(int(element->boundary == "NOT_BOUNDARY" && !element->calcGradDivOperationElement )) *
			complex<double>(0, boundary->omega[iOmega] * mu * dv) +

			double(int(element->boundary == "NOT_BOUNDARY" && element->calcGradDivOperationElement )) *
			complex<double>(0, -1.0 / element->resistivity * boundary->omega[iOmega] * mu * dv)));

		//globalMatrixToBeSolvedRealTriplet.push_back(valR);
		if (valI.value() != 0.0) {
			globalMatrixToBeSolvedImagTriplet.push_back(valI);
		}


	}

	//divergence correction in subsurface, now which is included in MakeMatrix();



	//for (int i = 0; i < 3 * numOfCalcElements; i++) {
	//	Element::Element* element = calcElementsVector[int(i / 3)];
	//	for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(divergenceCorrection->sumDivHdSMatrix, i); it; ++it) {
	//		int oriRow = i;
	//		int oriCol = it.col();
	//		Eigen::Triplet<double> val(originalOrderToSolverOrder[i],originalOrderToSolverOrder[oriCol],
	//			 factorMat.coeff(oriRow / 3, oriRow / 3) * divergenceCorrection->sumDivHdSMatrix.coeff(oriRow, oriCol));

	//		globalMatrixToBeSolvedRealTriplet.push_back(val);
	//	}
	//}
	


	//lobalMatrixToBeSolvedReal.setFromTriplets(globalMatrixToBeSolvedRealTriplet.begin(), globalMatrixToBeSolvedRealTriplet.end());
	//globalMatrixToBeSolvedReal.pruned();
	globalMatrixToBeSolvedImag.setFromTriplets(globalMatrixToBeSolvedImagTriplet.begin(), globalMatrixToBeSolvedImagTriplet.end());
	globalMatrixToBeSolvedImag.pruned();

	//memo:
	//Original: Ax=b
	//Modified Eq: LA(xR^(-1))R=Lb
	//A=A.real()+A.imag()
	//Eigen::IterScaling<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> > scal;
	// Compute the left and right scaling vectors. The matrix is equilibrated at output
	//scal.computeRef(globalMatrixToBeSolved);


	vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> rhs(2);
	rhs[0].resize(3 * numOfCalcElements);
	rhs[1].resize(3 * numOfCalcElements);
	rhs[0].setZero();
	rhs[1].setZero();
	for (int i = 0; i <  numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];

		if (element->boundary == "-Z_BOUNDARY") {
			for (int itr = 0; itr < 2; itr++) {
				if (itr == 0) {
					int solRow = originalOrderToSolverOrder[3 * i];
					rhs[0].coeffRef(solRow) = 1.0;
				}
				else {
					int solRow = originalOrderToSolverOrder[3 * i + 1];
					rhs[1].coeffRef(solRow) = 1.0;
				}
			}
		}
	}

	//globalVectorEachThread[threadID][0] = scal.LeftScaling().cwiseProduct(globalVectorEachThread[threadID][0]);
	//globalVectorEachThread[threadID][1] = scal.LeftScaling().cwiseProduct(globalVectorEachThread[threadID][1]);



	std::cout << "Solve H.. #Omega: " << iOmega  << endl;
	time_t start_t = time(NULL);


	if (isDirectSolver == true) {
		std::cout << "Solve_iterative Function cannot be used in Direct Solver!!!" << std::endl;
		exit(1);

	}
	else {

		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> sol(2);
		sol[0].resize(3 * numOfCalcElements);
		sol[1].resize(3 * numOfCalcElements);
		sol[0].setZero();
		sol[1].setZero();
		


		bool infFlag = false;
		for (int itr = 0; itr < 2; itr++) {
			for (int i = 0; i < 3 * numOfCalcElements; i++) {
				int solCol = originalOrderToSolverOrder[i];
				sol[itr].coeffRef(solCol) = resultVector.coeff((2 * iOmega + itr) * 3 * numOfCalcElements + i); //phi is zero as initial guess
				if (!std::isfinite(sol[itr].coeffRef(solCol).real()) || !std::isfinite(sol[itr].coeffRef(solCol).imag())) {
					infFlag = true;
				}
			}
		}
		if (infFlag) {
			sol[0].setZero();
			sol[1].setZero();
		}
		//Eigen::SparseMatrix < complex<double>, Eigen::RowMajor> mat;
		//mat.resize(3 * numOfCalcElements, 3 * numOfCalcElements );
		//mat.reserve(Eigen::VectorXi::Constant(3 * numOfCalcElements, 100));
		//for (int i = 0; i < 3 * numOfCalcElements; i++) {
		//	Element::Element* element = calcElementsVector[int(i / 3)];
		//	for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(*globalMatrixNoOmegaTerm, i); it; ++it) {
		//		mat.coeffRef(i, it.col()) = globalMatrixNoOmegaTerm->coeff(i, it.col()) + globalMatrixToBeSolvedImag.coeff(i, it.col());
		//	}
		//}
		//mat.makeCompressed();
		//iterativeSolverVector[iOmega]->solve(&mat, &sol, &rhs, iterativeSolverVector[iOmega]->m_relSolTol, false);
		//
		iterativeSolverVector[iOmega]->omega = boundary->omega[iOmega];
		iterativeSolverVector[iOmega]->precond.compute(*globalMatrixNoOmegaTerm, globalMatrixToBeSolvedImag);
		//if (useMultGrid) {
		//	iterativeSolverVector[iOmega]->precond_multi.compute(*globalMatrixNoOmegaTerm, globalMatrixToBeSolvedImag);
		//}
		//else {
		//	iterativeSolverVector[iOmega]->precond.compute(*globalMatrixNoOmegaTerm, globalMatrixToBeSolvedImag);
		//}
		iterativeSolverVector[iOmega]->solve(globalMatrixNoOmegaTerm, &globalMatrixToBeSolvedImag,&sol, &rhs, iterativeSolverVector[iOmega]->m_relSolTol, false);

		//iterativeSolverVector[iOmega]->solve(&precondMat, &sol, &rhs, iterativeSolverVector[iOmega]->m_relSolTol, false);

		if (isFirstLambdaAndLoop == false && iterativeSolverVector[iOmega]->m_maxIteration == iterativeSolverVector[iOmega]->m_iters) {
			std::cout << "In BiCGSafe Solution NOT Converged!!! Restart Without  Solution Guess." << std::endl;
			sol[0].setZero();
			sol[1].setZero();
			iterativeSolverVector[iOmega]->solve(globalMatrixNoOmegaTerm, &globalMatrixToBeSolvedImag, &sol, &rhs, iterativeSolverVector[iOmega]->m_relSolTol, false);
		}
		if (iterativeSolverVector[iOmega]->m_maxIteration == iterativeSolverVector[iOmega]->m_iters) {
			std::cout << "WARNING!!!!!!!!!!!!!!!! In BiCGSafe Solution NOT Converged!!!" << std::endl;
		}

		// Scale back the computed solution
		//for (int itr = 0; itr < 2; itr++) {
		//	m_res[threadID][itr] = scal.RightScaling().cwiseProduct(m_res[threadID][itr]);
		//}
		
		//vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> gradPhi(2);
		//gradPhi[0] = divergenceCorrection->gradOperatorMatrix * sol[0].block(3 * numOfCalcElements, 0, numOfCalcElements, 1);
		//gradPhi[1] = divergenceCorrection->gradOperatorMatrix * sol[1].block(3 * numOfCalcElements, 0, numOfCalcElements, 1);
		for (int itr = 0; itr < 2; itr++) {
			for (int i = 0; i < 3 * numOfCalcElements; i++) {
				double dv = calcElementsVector[i / 3]->dv;
				int solRow = originalOrderToSolverOrder[i];
				resultVector.coeffRef((2 * iOmega + itr) * 3 * numOfCalcElements + i) = sol[itr].coeff(solRow);// -gradPhi[itr].coeff(i) / dv;
			}
		}


		std::cout << "In BiCGSafe Period:     " << 2 * pi / boundary->omega[iOmega] << std::endl;
		std::cout << "In BiCGSafe #iterations:     " << iterativeSolverVector[iOmega]->m_iters << std::endl;
		std::cout << "In BiCGSafe estimated error: " << iterativeSolverVector[iOmega]->m_error << std::endl;
		std::cout << "In BiCGSafe Last Iteration, Relative Change of Solution:" << iterativeSolverVector[iOmega]->m_lastRelativeSolChange << std::endl;
		//debug


		/*int numOfCalcElements = itetativeSolverVector[iOmega]->rReturn[0].size() / 3;
		for (int i = 0; i < 3; i++) {
			for (int k = 0; k < 2; k++) {
				Eigen::VectorXd outputRes{ numOfCalcElements };
				for (int j = 0; j < numOfCalcElements; j++) {
					outputRes.coeffRef(j) = abs(itetativeSolverVector[iOmega]->rReturn[k].coeff(i * numOfCalcElements + j));
				}
				string filename = "residual_iter_Direction_" + to_string(k) + "_" + to_string(i) + ".vtk";
				output->VTKFileOputput(&calcElementsVector, &outputRes, filename);
			}

		}*/


	}

	time_t end_t = time(NULL);
	std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
}
void Analysis::Analysis::SetNotBoundaryElements() {
	
	int numElem = 0;
	for (int i = 0; i < numOfCalcElements; i++) {
		if (calcElementsVector[i]->boundary == "NOT_BOUNDARY") {
			numElem++;
		}
	}
	notBoundaryElements.resize(numElem);
	for (int i = 0; i < numOfCalcElements; i++) {
		if (calcElementsVector[i]->boundary == "NOT_BOUNDARY") {
			notBoundaryElements.push_back(calcElementsVector[i]);
		}
	}
}
void Analysis::Analysis::ClearHAndE() {
	for (int i = 0; i < calcElementsVector.size(); i++) {
		Element::Element* element = calcElementsVector[i];
		element->ClearHAndE(); 
	}
}
void Analysis::Analysis::ClearZ() {
	for (int i = 0; i < calcElementsVector.size(); i++) {
		Element::Element* element = calcElementsVector[i];
		element->ClearZ();
	}
}
void Analysis::Analysis::CalcE(int itr,int iOmega) {
	Eigen::VectorXcd Hresult{ 3 * numOfCalcElements };
	for (int i = 0; i < 3*calcElementsVector.size(); i++) {
		Hresult.coeffRef(i) = resultVector.coeff((2*iOmega + itr)*3*numOfCalcElements + i);
	}
	#pragma omp parallel for
	for (int i = 0; i < calcElementsVector.size();i++) {
	//for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
		Element::Element* element = calcElementsVector[i];
		element->CalcE(&Hresult,&elements,numOfCalcElements, itr); //本当はHの計算と整合性が取れるよう深いレイヤーから計算していき、浅いレイヤはその和にしないといけない？→rotHdSがそのようにして算出しているので、なってるはず
																  //if (Hpolarization == 0) {
		//	cout << element->E.back() <<" H:" << element->H.back() << " coord:" << element->rootCoord << endl;
		//}
	}

}
void Analysis::Analysis::CalcZ(int iOmega) {
	#pragma omp parallel for
	for (int i = 0; i < calcElementsVector.size(); i++) {
	//for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
		Element::Element* element = calcElementsVector[i];
		element->CalcZ(&elements, numOfCalcElements,iOmega);
		element->rhoXY = pow(std::sqrt(std::pow(element->Z[iOmega].coeff(0, 1).real(), 2.0) + std::pow(element->Z[iOmega].coeff(0, 1).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
		element->rhoYX = pow(std::sqrt(std::pow(element->Z[iOmega].coeff(1, 0).real(), 2.0) + std::pow(element->Z[iOmega].coeff(1, 0).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
		double tmp = 0;
		if (element->Z[iOmega].coeff(0, 1).real() != 0) {
			tmp = element->Z[iOmega].coeff(0, 1).imag() / element->Z[iOmega].coeff(0, 1).real();
			element->phiXY = std::atan(tmp) / ConstantValues::pi * 180;
		}
		else {
			element->phiXY = 0;
		}
		if (element->Z[iOmega].coeff(1, 0).real() != 0) {
			tmp = element->Z[iOmega].coeff(1, 0).imag() / element->Z[iOmega].coeff(1, 0).real();
			element->phiYX = std::atan(tmp) / ConstantValues::pi * 180;
		}
		else {
			element->phiYX = 0;
		}

		

	}
	
}
void Analysis::Analysis::CalcT(int iOmega) {
#pragma omp parallel for
	for (int i = 0; i < calcElementsVector.size(); i++) {
		//for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
		Element::Element* element = calcElementsVector[i];
		element->CalcT(iOmega);
	}

}
void Analysis::Analysis::SetNeighborElements() {
	#pragma omp parallel for
	//for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
	for (int i = 0; i < elementsVector.size(); i++) {
		Element::Element* element = elementsVector[i];
		element->SetNeighborElements(&elements);
	}
}
void Analysis::Analysis::MakeMatrix(bool isRebuildMatrix,bool isCalcInversionValues) {
	double resisAir;
	for (int i = 0; i < propertiesVector.size(); i++) {
		if (propertiesVector[i]->type == Property::Property::AIR) {
			resisAir = propertiesVector[i]->resistivity;
		}
	}

	globalVector->setZero();
	globalMatrixNoOmegaTerm->setZero();

	vector<Eigen::Triplet<double>> globalMatrixNoOmegaTermTriplet;
	globalMatrixNoOmegaTermTriplet.resize(int(3*numOfCalcElements * 100)); //100 per row as average as enough value, this is for preventing from program running is crushed.
	int count = 0;
	for (int i = 0; i < numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];

		double dv = element->dv;

		if (element->boundary == "NOT_BOUNDARY") {
			if (!element->calcGradDivOperationElement) {
				for (int ii = 0; ii < 3; ii++) {
					for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*(element->sumNCrossRhoRotHdS), ii); it; ++it) {
						int j = it.col();
						int solRow = originalOrderToSolverOrder[3 * i + ii];
						int solCol = originalOrderToSolverOrder[j];
						Eigen::Triplet<double> val(solRow, solCol, element->sumNCrossRhoRotHdS->coeff(ii, j).real());
						globalMatrixNoOmegaTermTriplet[count] = val;
						count++;
					}
				}
				element->calcGradDivOperationElement = false;
				//std::complex<double> term;
				//term.imag(+omega * mu * dv);
				//Eigen::Triplet<complex<double>> val1(3 * i, 3 * i, term);// SetFromTriplet should sum up all same row col elems
				//globalMatrixTriplet.push_back(val1);
				//Eigen::Triplet<complex<double>> val2(3 * i + 1, 3 * i + 1, term);// SetFromTriplet should sum up all same row col elems
				//globalMatrixTriplet.push_back(val2);
				//Eigen::Triplet<complex<double>> val3(3 * i + 2, 3 * i + 2, term);// SetFromTriplet should sum up all same row col elems
				//globalMatrixTriplet.push_back(val3);
			}
			else { //J=0 => rotH=0 => rot rot H=0 => div grad H =0 in air layer 
				int calcID = element->calcID;
				element->calcGradDivOperationElement = true;
				for (int ii = 0; ii < 3; ii++) {
					for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(divergenceCorrection->divGradMatrix, calcID); it; ++it) {
						int row = 3 * calcID + ii;
						int col = 3 * it.col() + ii;
						int solRow = originalOrderToSolverOrder[row];
						int solCol = originalOrderToSolverOrder[col];
						double v = divergenceCorrection->divGradMatrix.coeff(calcID, it.col());
						Eigen::Triplet<double> val(solRow , solCol, v);
						globalMatrixNoOmegaTermTriplet[count]=val;
						count++;
					}
				}
			}


		}

		

		else if (element->boundary == "-Z_BOUNDARY") {
			//double factor = 0.0;
			//for (int isurf = 0; isurf < 6; isurf++) {
			//	factor += resisAir * element->dv / element->dSvector[isurf];
			//}
			double factor = 1.0;
			int solRow = originalOrderToSolverOrder[3 * i];
			int solCol = originalOrderToSolverOrder[3 * i];
			Eigen::Triplet<double> val1(solRow, solCol , factor);// SetFromTriplet should sum up all same row col elems
			globalMatrixNoOmegaTermTriplet[count]=val1;
			count++;
			solRow = originalOrderToSolverOrder[3 * i + 1];
			solCol = originalOrderToSolverOrder[3 * i + 1];
			Eigen::Triplet<double> val2(solRow, solCol, factor);// SetFromTriplet should sum up all same row col elems
			globalMatrixNoOmegaTermTriplet[count]=val2;
			count++;
			solRow = originalOrderToSolverOrder[3 * i + 2];
			solCol = originalOrderToSolverOrder[3 * i + 2];
			Eigen::Triplet<double> val3(solRow, solCol, factor);// SetFromTriplet should sum up all same row col elems
			globalMatrixNoOmegaTermTriplet[count]=val3;
			count++;

			if (Hpolarization == 0) {
				globalVector->coeffRef(3 * i, 0) = factor;
			}
			else {
				globalVector->coeffRef(3 * i + 1, 0) = factor;
			}

		}
		else if(element->boundary == "+Z_BOUNDARY") {
			//double factor = 0.0;
			//for (int isurf = 0; isurf < 6; isurf++) {
			//	factor += resisAir * element->dv / element->dSvector[isurf];
			//}
			double factor = 1.0;
			int solRow = originalOrderToSolverOrder[3 * i];
			int solCol = originalOrderToSolverOrder[3 * i];
			Eigen::Triplet<double> val1(solRow, solCol, factor);// SetFromTriplet should sum up all same row col elems
			globalMatrixNoOmegaTermTriplet[count] = val1;
			count++;
			solRow = originalOrderToSolverOrder[3 * i + 1];
			solCol = originalOrderToSolverOrder[3 * i + 1];
			Eigen::Triplet<double> val2(solRow, solCol, factor);// SetFromTriplet should sum up all same row col elems
			globalMatrixNoOmegaTermTriplet[count] = val2;
			count++;
			solRow = originalOrderToSolverOrder[3 * i + 2];
			solCol = originalOrderToSolverOrder[3 * i + 2];
			Eigen::Triplet<double> val3(solRow, solCol, factor);// SetFromTriplet should sum up all same row col elems
			globalMatrixNoOmegaTermTriplet[count] = val3;
			count++;
		}
		else {
			//double factor = 0.0;
			//for (int isurf = 0; isurf < 6; isurf++) {
			//	factor += resisAir * element->dv / element->dSvector[isurf];
			//}
			double factor = 1.0;
			Eigen::Vector3i pos;
			pos.setZero();
			if (element->boundary == "-X_BOUNDARY") {
				pos[0] = 1;
			}
			else if (element->boundary == "+X_BOUNDARY") {
				pos[0] = -1;
			}
			else if (element->boundary == "-Y_BOUNDARY") {
				pos[1] = 1;
			}
			else if (element->boundary == "+Y_BOUNDARY") {
				pos[1] = -1;
			}
			else if (element->boundary == "+Z_BOUNDARY") {
				pos[2] = -1;
			}

			int solRow = originalOrderToSolverOrder[3 * i];
			int solCol = originalOrderToSolverOrder[3 * i];
			Eigen::Triplet<double> val1(solRow, solCol, factor);// SetFromTriplet should sum up all same row col elems
			globalMatrixNoOmegaTermTriplet[count] = val1;
			count++;
			solRow = originalOrderToSolverOrder[3 * i + 1];
			solCol = originalOrderToSolverOrder[3 * i + 1];
			Eigen::Triplet<double> val2(solRow, solCol, factor);// SetFromTriplet should sum up all same row col elems
			globalMatrixNoOmegaTermTriplet[count] = val2;
			count++;
			solRow = originalOrderToSolverOrder[3 * i + 2];
			solCol = originalOrderToSolverOrder[3 * i + 2];
			Eigen::Triplet<double> val3(solRow, solCol, factor);// SetFromTriplet should sum up all same row col elems
			globalMatrixNoOmegaTermTriplet[count] = val3;
			count++;

			string neighborID = Functions::GetNeighborElement(&elements, element, pos, nx, ny, nz);
			int j = elements[neighborID]->calcID;
			solRow = originalOrderToSolverOrder[3 * i];
			solCol = originalOrderToSolverOrder[3 * j];
			Eigen::Triplet<double> val4(solRow, solCol, -factor);// SetFromTriplet should sum up all same row col elems
			globalMatrixNoOmegaTermTriplet[count]=val4;
			count++;
			solRow = originalOrderToSolverOrder[3 * i + 1];
			solCol = originalOrderToSolverOrder[3 * j + 1];
			Eigen::Triplet<double> val5(solRow, solCol, -factor);// SetFromTriplet should sum up all same row col elems
			globalMatrixNoOmegaTermTriplet[count]=val5;
			count++;
			solRow = originalOrderToSolverOrder[3 * i + 2];
			solCol = originalOrderToSolverOrder[3 * j + 2];
			Eigen::Triplet<double> val6(solRow, solCol, -factor);// SetFromTriplet should sum up all same row col elems
			globalMatrixNoOmegaTermTriplet[count]=val6;
			count++;
		}

	}

	for (int i = 0; i<int(3 * numOfCalcElements * 100) - count; i++)
	{
		globalMatrixNoOmegaTermTriplet.erase(globalMatrixNoOmegaTermTriplet.end() - 1);
	}


	//divergence correction in subsurface


	Eigen::SparseMatrix<double, Eigen::RowMajor> factorMat;
	factorMat.resize(numOfCalcElements, numOfCalcElements);
	factorMat.reserve(Eigen::VectorXi::Constant(numOfCalcElements, 1));
	for (int i = 0; i < numOfCalcElements; i++) {
		if (!calcElementsVector[i]->calcGradDivOperationElement && calcElementsVector[i]->boundary == "NOT_BOUNDARY") {
			//double ave = 0;
			double resis;
			if (calcElementsVector[i]->property->type == Property::Property::AIR) {
				/*resis = 0.0;
				for (int j = 0; j < 6; j++) {

					resis += (*calcElementsVector[i]->resistivitySurface)[j] / 6.0;
				}*/
				resis = 0.0;
				for (int j = 0; j < 6; j++) {
					for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*(calcElementsVector[i]->resistivitySurfaceCoeff[j]), 0); it; ++it) {
						if (resis < calcElementsVector[it.col()]->resistivity) {
							resis = calcElementsVector[it.col()]->resistivity;
						}
					}
					//ave += (*calcElementsVector[i]->resistivitySurface)[j];
				}
				calcElementsVector[i]->factorForDivCorr = resis;
			}
			else {
				resis = 1e30;
				for (int j = 0; j < 6; j++) {
					for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*(calcElementsVector[i]->resistivitySurfaceCoeff[j]), 0); it; ++it) {
						if (resis > calcElementsVector[it.col()]->resistivity) {
							resis = calcElementsVector[it.col()]->resistivity;
						}
					}
					//ave += (*calcElementsVector[i]->resistivitySurface)[j];
				}
				calcElementsVector[i]->factorForDivCorr = invSettings->safetyFactor * resis;// *min(1.0, sqrt(boundary->omega[iOmega]));
			}

			/*resis = 0.0;
			for (int j = 0; j < 6; j++) {
				resis += (*calcElementsVector[i]->resistivitySurface)[j]/6.0;
			}*/
			//double mid = exp((log(calcElementsVector[i]->resistivity) + log(boundary->omega[iOmega] * mu)) / 2.0);




			factorMat.coeffRef(i, i) = -calcElementsVector[i]->factorForDivCorr;
			/*factorMat.coeffRef(3*i, 3*i) = -calcElementsVector[i]->factorForDivCorr;
			factorMat.coeffRef(3 * i + 1, 3 * i + 1) = -calcElementsVector[i]->factorForDivCorr;
			factorMat.coeffRef(3 * i +2, 3 * i + 2) = -calcElementsVector[i]->factorForDivCorr;*/
			//*minResis;// calcElementsVector[i]->resistivity;
		}
	}


	for (int i = 0; i < 3 * numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[int(i / 3)];
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(divergenceCorrection->sumDivHdSMatrix, i); it; ++it) {
			int oriRow = i;
			int oriCol = it.col();
			int solRow = originalOrderToSolverOrder[oriRow];
			int solCol = originalOrderToSolverOrder[oriCol];
			Eigen::Triplet<double> val(solRow, solCol,
				factorMat.coeff(oriRow / 3, oriRow / 3) * divergenceCorrection->sumDivHdSMatrix.coeff(oriRow, oriCol));

			globalMatrixNoOmegaTermTriplet.push_back(val);
		}
	}



	globalMatrixNoOmegaTerm->setFromTriplets(globalMatrixNoOmegaTermTriplet.begin(), globalMatrixNoOmegaTermTriplet.end());
	globalMatrixNoOmegaTerm->pruned(0.0);
	globalMatrixNoOmegaTerm->makeCompressed();

	if (isCalcInversionValues) {
		*globalMatrixNoOmegaTermAdjoint = globalMatrixNoOmegaTerm->adjoint();
	}




	*globalMatrixNoOmegaTerm = (*globalMatrixNoOmegaTerm);

	globalVector->makeCompressed();	
		

}

void Analysis::Analysis::CalcSumNCrossRhoRotHdSElements() {
	
	SetSameLayerElements();
	
	cout <<"MaxLayer:"<< maxLayer << endl;
	for (int iLayer = maxLayer; iLayer >= 0; iLayer--) {
		//this would be faster by compiling layer by layer and parallelize
#pragma omp parallel for
		for (int i = 0; i < sameLayerElementsVector[iLayer].size(); i++) {
		//for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
			Element::Element* element = sameLayerElementsVector[iLayer][i];
			if (element->boundary == "NOT_BOUNDARY") {
				element->CalcSumNCrossRhoRotHdS(&elements, numOfCalcElements);
			}
		}
	}
	//Eigen::setNbThreads(0);
	//Eigen::initParallel();
}
void Analysis::Analysis::SetTransitionZoneElements() {
	for (int i = 0; i < numOfCalcElements;i++) {
		Element::Element* element = calcElementsVector[i];
		element->SetTransitionZone(&elements,numOfCalcElements);
	}
}
void Analysis::Analysis::SetNumOfCalcElementsAndCalcElementsAndElementsVector() {
	numOfCalcElements = 0;
	 minDx = 10000000000;
	 minDy = 10000000000;
	 minDz = 10000000000;
	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
		
		Element::Element* element = *itr;
		if (element->isParent == false) {
			calcElementsVector.push_back(element);
			calcElements[element->ID] = element;
			element->calcID = numOfCalcElements;
			numOfCalcElements++;
			if (element->dx < minDx) {
				minDx = element->dx;
			}
			if (element->dy < minDy) {
				minDy = element->dy;
			}
			if (element->dz < minDz) {
				minDz = element->dz;
			}
		}
	}


	cout << "minimum dx:" << minDx << " minimum dy:" << minDy << " minimum dz:" << minDz << endl;
}

//
void Analysis::Analysis::AssociationPropertiesToElements() {
	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
		Element::Element* element = *itr;
		bool findProp = false;
		if (!properties.contains(element->propID)) {
			std::cout << "No Property ID which set to Element in Data" << std::endl;
			exit(1);
		}
		Property::Property* property = properties[element->propID];
		int propID = property->ID;

		element->property = property;
		element->resistivity = property->resistivity;
		element->initialResistivity = property->resistivity;



	}
}
//
//
//
//void Analysis::Analysis::CalcResistivityAtTransitionElements() {
//	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
//		Element::Element* element=*itr;
//		bool isAirCellAtBoundaryAirGround = false;
//		if (element->property->type == Property::Property::AIR) {
//			Element::Element* groundElement;
//			for (auto itr2 = element->facesVector.begin(); itr2 != element->facesVector.end(); itr2++) {
//				Face::Face* face = *itr2;
//				for (auto itr3 = face->elementsVector.begin(); itr3 != face->elementsVector.end(); itr3++) {
//					Element::Element* tmpElement=*itr3;
//					if (tmpElement->property->type != Property::Property::AIR) {
//						isAirCellAtBoundaryAirGround = true;
//						groundElement = tmpElement;
//						break;
//					}
//				}
//				if (isAirCellAtBoundaryAirGround == true) {
//					break;
//				}
//
//			}
//
//			if (isAirCellAtBoundaryAirGround == true) {
//				//element->resistivity = (element->resistivity + groundElement->resistivity) / 2;
//				element->resistivity = ((0.5*element->resistivity.inverse()) +
//					(0.5*groundElement->resistivity.inverse())).inverse();
//				//element->resistivity = groundElement->resistivity;
//			}
//
//		}
//	}
//}
void Analysis::Analysis::SetLayerOfElements() {

	int maxNx = 0;
	int maxNy = 0;
	int maxNz = 0;
	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
		Element::Element* element = *itr;
		element->layer = (int)(element->ID.length()-9) / 2 - 1; //9 is nx,ny,nz place

		int nztmp = std::stoi(element->ID.substr(0, 3));
		int nytmp = std::stoi(element->ID.substr(3, 3));
		int nxtmp = std::stoi(element->ID.substr(6, 3));
		element->IDX = nxtmp;
		element->IDY = nytmp;
		element->IDZ = nztmp;
		if (maxNx < nxtmp) {
			maxNx = nxtmp;
		}
		if (maxNy < nytmp) {
			maxNy = nytmp;
		}
		if (maxNz < nztmp) {
			maxNz = nztmp;
		}
	}
	nx = maxNx + 1;
	ny = maxNy + 1;
	nz = maxNz + 1;

	sameIDZElements.resize(nz);
	for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
		Element::Element* element = *itr;
		sameIDZElements[element->IDZ].push_back(element);
		sameIDXYZElements[element->IDX][element->IDY][element->IDZ].push_back(element);
	}

	basedElementsSortByNxNyNz.resize(nx);
	for (int i = 0; i < nx; i++) {
		basedElementsSortByNxNyNz[i].resize(ny);
		for (int j = 0; j < ny; j++) {
			basedElementsSortByNxNyNz[i][j].resize(nz);
		}
	}
	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
		Element::Element* element = *itr;
		if (element->layer!=0) {
			continue;
		}
		basedElementsSortByNxNyNz[element->IDX][element->IDY][element->IDZ] = element;
	}
	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
		Element::Element* element = *itr;
		element->nx = nx;
		element->ny = ny;
		element->nz = nz;
	}


	int maxLayer = 0;
	for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
		Element::Element* element = *itr;
		if (element->layer > maxLayer) {
			maxLayer = element->layer;
		}
	}

	//set Roughen Matrix Normalization Unit Value

	for (int iLayer = 0; iLayer <= maxLayer; iLayer++) {
		double minDx = 10000000000;
		double minDy = 10000000000;
		double minDz = 10000000000;
		for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
			Element::Element* element= *itr;
			if (element->layer == iLayer) {
				if (element->dx < minDx) {
					minDx = element->dx;
				}
				if (element->dy < minDy) {
					minDy = element->dy;
				}
				if (element->dz < minDz) {
					minDz = element->dz;
				}
			}
		}
		for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
			Element::Element* element = *itr;
			if (element->layer == iLayer) {
				element->roughenMatrixUnit = std::min(std::min(minDx, minDy), minDz);
			}
		}
	}

}
void Analysis::Analysis::CalcNumOfDirichletConditionCells() {
	numOfDirichletConditionCells = 0;
	for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
		Element::Element* element = *itr;
		if (element->boundary == "-Z_BOUNDARY") {
			numOfDirichletConditionCells++;
		}
	}
}


void Analysis::Analysis::SetObsDataToElement() {
	numOfObsPointElements = 0;
	obsPointElements.resize(0);
	int impedanceID = 0;
	int tipperID = 0;
	for (auto itr = elements.begin(); itr != elements.end(); itr++) {
		Element::Element* element = itr->second;
		bool isObsElement = false;
		if (element->property->type == Property::Property::AIR) {
			continue;
		}
		Eigen::Vector3i pos;
		//for (int i = 0; i < 6; i++) {
		pos.setZero();
		pos.coeffRef(2) = -1;
		int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
		if (!(element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false && elements[element->alreadyFoundNeighborID[ipos]]->isAirGroundBoundaryCell == true && element->property->type != Property::Property::AIR)) {
			continue;
		}

		Element::Element* tmpElement = element;
		Element::Element* obsElement = element;

		bool upsideIsSea = true;
		while (true) {
			if (tmpElement->property->type == Property::Property::SEA) {
				tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1つ深いセルへ

			}
			else if (upsideIsSea) {
				upsideIsSea = false;
				tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1つ深いセルへ
			}
			else {
				obsElement = tmpElement;
				break;
				//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1]; 
			}
		}

		//Comment out Below Because We set Air Cell Around Ground To AirGroundBoundaryCell 
		//while (true) {
		//	bool isFoundElement = true;
		//	for (int i = 0; i < 3; i++) {
		//		for (int j = 0; j < 3; j++) {
		//			for (int k = 0; k < 3; k++) {
		//				ipos = i + 3 * j + 9 * k;
		//				if (tmpElement->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && tmpElement->neighborElements[ipos]->property->type == Property::Property::AIR) {
		//					isFoundElement = false;
		//				}
		//			}
		//		}
		//	}
		//	if (isFoundElement) {
		//		obsElement = tmpElement;
		//		break;
		//	}
		//	else {
		//		tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1つ深いセルへ
		//		//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1]; 
		//	}
		//}
		obsPointElements.reserve(obsData.size());// max size we should consider is  obsData.size().
		double eps = 1e-6;
		for (int i = 0; i < obsData.size(); i++) {
			if (obsData[i]->isImpedanceData == true) {
				Eigen::Vector3d X;
				X.coeffRef(0) = obsData[i]->coord.coeff(0);
				X.coeffRef(1) = obsData[i]->coord.coeff(1);
				X.coeffRef(2) = obsElement->centerCoord.coeff(2); //not related


				//cout << obsElement->rootCoord.coeff(0) << " " << x << " " << obsElement->rootCoord.coeff(1) << " " << y << endl;
				if (obsElement->CheckThePointInside2D(X,4)) {

					if (obsData[i]->isAlreadyFoundElementImpedance == false) {
						obsData[i]->isAlreadyFoundElementImpedance = true;
						
						if (obsElement->impedanceObsID == -1) {
							obsElement->impedanceObsID = impedanceID;
							impedanceID++;
							obsElement->impedanceObsData = obsData[i];
						}
						else {
							double d1 = (obsElement->centerCoord - X).norm();
							Eigen::Vector3d X2;
							X2.coeffRef(0) = obsElement->impedanceObsData->coord.coeff(0);
							X2.coeffRef(1) = obsElement->impedanceObsData->coord.coeff(1);
							X2.coeffRef(2) = obsElement->centerCoord.coeff(2); //not related
							double d2 = (obsElement->centerCoord - X2).norm();
							if (d1 < d2) {
								cout << "Warning!!!! In Impedance, Duplicate Stations:" << obsData[i]->name<<" "<< obsElement->impedanceObsData->name<< endl;
								cout << "Use Station Name:" << obsData[i]->name << endl;
								obsElement->impedanceObsData = obsData[i];
								
							}
							else {
								cout << "Warning!!!! In Impedance, Duplicate Stations:" << obsData[i]->name << " " << obsElement->impedanceObsData->name << endl;
								cout << "Use Station Name:" << obsElement->impedanceObsData->name << endl;
							}
							
						}
						if (obsElement->isObservationElement == false) {
							obsElement->isObservationElement = true;
							
							obsPointElements.push_back(obsElement);
							obsImpedanceElements.push_back(obsElement);
							numOfObsPointElements++;
						}
						obsElement->isInversionImpedance = true;
						if (obsElement->boundary != "NOT_BOUNDARY") {
							std::cout << "ERROR:Obs Data Location is in Boundary Cell." << std::endl;
							exit(1);
						}
					}
				}
			}

			double eps = 1e-6;
			if (obsData[i]->isTipperData == true) {
				Eigen::Vector3d X;
				X.coeffRef(0) = obsData[i]->coord.coeff(0);
				X.coeffRef(1) = obsData[i]->coord.coeff(1);
				X.coeffRef(2) = obsElement->centerCoord.coeff(2); //not related
				//cout << obsElement->rootCoord.coeff(0) << " " << x << " " << obsElement->rootCoord.coeff(1) << " " << y << endl;
				if (obsElement->CheckThePointInside2D(X,4)) {
					if (obsData[i]->isAlreadyFoundElementTipper == false) {
						obsData[i]->isAlreadyFoundElementTipper = true;
						
						if (obsElement->tipperObsID == -1) {
							obsElement->tipperObsID = tipperID;
							tipperID++;
							obsElement->tipperObsData = obsData[i];
						}
						else {
							double d1 = (obsElement->centerCoord - X).norm();
							Eigen::Vector3d X2;
							X2.coeffRef(0) = obsElement->tipperObsData->coord.coeff(0);
							X2.coeffRef(1) = obsElement->tipperObsData->coord.coeff(1);
							X2.coeffRef(2) = obsElement->centerCoord.coeff(2); //not related
							double d2 = (obsElement->centerCoord - X2).norm();
							
							if (d1 < d2) {
								cout << "Warning!!!! In Tipper, Duplicate Stations:" << obsData[i]->name << " " << obsElement->tipperObsData->name << endl;
								cout << "Use Station Name:" << obsData[i]->name << endl;
								obsElement->tipperObsData = obsData[i];

							}
							else {
								cout << "Warning!!!! In Tipper, Duplicate Stations:" << obsData[i]->name << " " << obsElement->tipperObsData->name << endl;
								cout << "Use Station Name:" << obsElement->tipperObsData->name << endl;
							}
						}
						if (obsElement->isObservationElement == false) {
							obsElement->isObservationElement = true;
							
							obsPointElements.push_back(obsElement);
							numOfObsPointElements++;
						}
						obsElement->isInversionTipper = true;
						if (obsElement->boundary != "NOT_BOUNDARY") {
							std::cout << "ERROR:Obs Data Location is in Boundary Cell." << std::endl;
							exit(1);
						}
					}
				}
			}

		}	
	}
	numOfObsImpedanceElements = impedanceID;
	for (int i = 0; i < obsData.size(); i++) {
		if (obsData[i]->isImpedanceData==true && obsData[i]->isAlreadyFoundElementImpedance == false) {
			std::cout << "Zobs has data out of range." << std::endl;
			std::cout << std::fixed;
			std::cout << "ID:" << i << endl;
			std::cout << "X:"<<std::setprecision(5) << obsData[i]->coord.coeff(0) << endl;
			std::cout << "Y:"<< std::setprecision(5) << obsData[i]->coord.coeff(1) << endl;
			exit(1);
		}
		if (obsData[i]->isTipperData==true && obsData[i]->isAlreadyFoundElementTipper == false) {
			std::cout << "Tobs has data out of range." << std::endl;
			std::cout << std::fixed;
			std::cout << "X:" << std::setprecision(5) << obsData[i]->coord.coeff(0) << endl;
			std::cout << "Y:" << std::setprecision(5) << obsData[i]->coord.coeff(1) << endl;
			exit(1);
		}
	}

}

void Analysis::Analysis::CalcLambda(int iOmega, int threadID,bool onePointMode) {
	lambdaEachOmega[iOmega]->setZero();
	//====Calc ∂J/∂H==========
	if (dJdH[threadID].size() == 0) {
		dJdH[threadID].resize(2 * 3 * numOfCalcElements); //iteration*(real and imag part)*(Hx,Hy and Nz)*numOfCalcElements
	}
	dJdH[threadID].setZero();

	//===Impedance=====
	//====Calc ∂/∂H_i　Σ_j (Zcalc_j-ZObs_j)**2

	//std::ofstream f2;
	//f2.open("debugZtmpdZdHtmp.txt", std::ios::trunc);
	
	Eigen::Matrix2cd Zcalc;
	if (onePointMode == false) {
		for (int j = 0; j < numOfObsPointElements; j++) {
			Element::Element* element = obsPointElements[j];
			if (element->isInversionImpedance == true) {
				int ID = element->impedanceObsID * boundary->omega.size() + iOmega;
				if (useImpedanceDataArray.coeff(ID) == 0.0) {
					continue;
				}

				if (isInvertedDistortion) {
					Zcalc = element->distortionMatrix * element->Z[iOmega];
					dZdHCalc[threadID](0, 0) = element->distortionMatrix.coeff(0, 0) * element->dZdH[iOmega](0, 0) +
						element->distortionMatrix.coeff(0, 1) * element->dZdH[iOmega](1, 0);
					dZdHCalc[threadID](0, 1) = element->distortionMatrix.coeff(0, 0) * element->dZdH[iOmega](0, 1) +
						element->distortionMatrix.coeff(0, 1) * element->dZdH[iOmega](1, 1);
					dZdHCalc[threadID](1, 0) = element->distortionMatrix.coeff(1, 0) * element->dZdH[iOmega](0, 0) +
						element->distortionMatrix.coeff(1, 1) * element->dZdH[iOmega](1, 0);
					dZdHCalc[threadID](1, 1) = element->distortionMatrix.coeff(1, 0) * element->dZdH[iOmega](0, 1) +
						element->distortionMatrix.coeff(1, 1) * element->dZdH[iOmega](1, 1);

				}
				else {
					Zcalc = element->Z[iOmega];
					for (int ii = 0; ii < 2; ii++) {
						for (int jj = 0; jj < 2; jj++) {
							dZdHCalc[threadID](ii, jj) = element->dZdH[iOmega](ii, jj);
						}
					}
				}
				for (int ii = 0; ii < 2; ii++) {
					for (int jj = 0; jj < 2; jj++) {
						for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dZdH[iOmega](ii, jj), 0); it; ++it)
						{
							int iCol = it.col();
							/*cout << "in calcLambda" << ii << " " << jj << " " << element->dZdH[iOmega](ii, jj).coeff(0, 13981) << endl;*/
							std::complex<double> dZtmp = Zcalc.coeff(ii, jj) - element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj);
							std::complex<double> dZdHtmp = dZdHCalc[threadID](ii, jj).coeff(0, iCol);
							//f2 << iCol << " " << ii << " " << jj << " " << dZtmp << " " << dZdHtmp << "before" << endl;
							double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
							double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));


							//Real Part
							if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0 ) {
								dZtmp.real(dZtmp.real() / epsReal);
								dZdHtmp.real(dZdHtmp.real() / epsReal);
								//cout << "dZtmpReal:" << dZtmp.real() << endl;
								//cout << "dZdHtmpReal:" << dZdHtmp.real() << endl;
							}
							else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
								dZtmp.real(0.0);
							}
							else {
								//そのまま
							}
							if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0 ) {
								dZtmp.imag(dZtmp.imag() / epsImag);
								dZdHtmp.imag(dZdHtmp.imag() / epsImag);
								//cout << "dZtmpImag:" << dZtmp.imag() << endl;
								//cout << "dZdHtmpImag:" << dZdHtmp.imag() << endl;
							}
							else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
								dZtmp.imag(0.0);
							}
							else {
								//そのまま
							}
							dJdH[threadID].coeffRef(iCol).real(dJdH[threadID].coeffRef(iCol).real() + 2.0 * (conj(dZtmp) * dZdHtmp).real());
							//f2 << iCol << " " << dZtmp << " " << dZdHtmp << "middlle" << endl;

							//Imag Part
							dZdHtmp = dZdHCalc[threadID](ii, jj).coeff(0, iCol);
							if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0) {
								dZdHtmp.real(dZdHtmp.real() / epsImag); //ひっくり返る
								//cout << "dZtmpReal:" << dZtmp.real() << endl;
								//cout << "dZdHtmpReal:" << dZdHtmp.real() << endl;
							}
							else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
								dZdHtmp.real(0.0);
							}
							else {
								//そのまま
							}
							if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0 ) {
								dZdHtmp.imag(dZdHtmp.imag() / epsReal); //ひっくり返る
									//cout << "dZtmpImag:" << dZtmp.imag() << endl;
									//cout << "dZdHtmpImag:" << dZdHtmp.imag() << endl;
							}
							else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
								dZdHtmp.imag(0.0);
							}
							else {
								//そのまま
							}


							dJdH[threadID].coeffRef(iCol).imag(dJdH[threadID].coeffRef(iCol).imag() + 2.0 * (conj(dZtmp) * dZdHtmp).imag());
						}

					}
				}

			}
		}

		//===Tipper=====
		//====Calc ∂/∂H_i　Σ_j (Tcalc_j-TObs_j)**2

		for (int j = 0; j < numOfObsPointElements; j++) {
			Element::Element* element = obsPointElements[j];
			if (element->isInversionTipper == true) {
				int ID = element->tipperObsID * boundary->omega.size() + iOmega;
				if (useTipperDataArray.coeff(ID) == 0.0) {
					continue;
				}
				for (int ii = 0; ii < 2; ii++) {
					for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dTdH[iOmega](ii), 0); it; ++it)
					{
						int iCol = it.col();
						std::complex<double> dTtmp = element->T[iOmega].coeff(ii) - element->tipperObsData->TobsVector[iOmega].coeff(ii);
						std::complex<double> dTdHtmp = element->dTdH[iOmega](ii).coeff(0, iCol);
						double epsReal = std::abs(element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii));
						double epsImag = std::abs(element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii));


						//Real Part
						if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) > 0 ) {
							dTtmp.real(dTtmp.real() / epsReal);
							dTdHtmp.real(dTdHtmp.real() / epsReal);
						}
						else if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) <= 0) {
							dTtmp.real(0.0);
						}
						else {
							//そのまま
						}
						if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) > 0 ) {
							dTtmp.imag(dTtmp.imag() / epsImag);
							dTdHtmp.imag(dTdHtmp.imag() / epsImag);

						}
						else if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) <= 0) {
							dTtmp.imag(0.0);
						}
						else {
							//そのまま
						}
						dJdH[threadID].coeffRef(iCol).real(dJdH[threadID].coeffRef(iCol).real() + 2.0 * (conj(dTtmp) * dTdHtmp).real());

						//Imag Part
						dTdHtmp = element->dTdH[iOmega](ii).coeff(0, iCol);
						if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) > 0 ) {
							dTdHtmp.real(dTdHtmp.real() / epsImag); //ひっくり返る
						}
						else if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) <= 0) {
							dTdHtmp.real(0.0);
						}
						else {
							//そのまま
						}
						if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) > 0 ) {
							dTdHtmp.imag(dTdHtmp.imag() / epsReal); //ひっくり返る
						}
						else if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) <= 0) {
							dTdHtmp.imag(0.0);
						}
						else {
							//そのまま
						}
						dJdH[threadID].coeffRef(iCol).imag(dJdH[threadID].coeffRef(iCol).imag() + 2.0 * (conj(dTtmp) * dTdHtmp).imag());

						//debug
						//double preJ = CalcDataMisfit();
						//int itr = iCol / (3 * numOfCalcElements);
						//int elemNum = iCol % (3 * numOfCalcElements)/3;
						//int direcH = iCol % 3;
						//cout << "Tx or Ty:" << ii << endl;
						//cout << "eleNum:" << elemNum << " "
						//	<< "itr:" << itr << " "
						//	<< "direcH:" << direcH << endl;
						//cout <<"H1:"<< calcElementsVector[elemNum]->H[0] << endl;
						//cout << "H2:" << calcElementsVector[elemNum]->H[1] << endl;
						//cout << "dTxdHz1:" << calcElementsVector[elemNum]->dTdH[iOmega](0).coeffRef(0,3*elemNum+2) << endl;
						//cout << "dTxdHz2:" << calcElementsVector[elemNum]->dTdH[iOmega](0).coeffRef(0,3*numOfCalcElements + 3 * elemNum + 2) << endl;
						//cout << "dTydHz1:" << calcElementsVector[elemNum]->dTdH[iOmega](1).coeffRef(0, 3 * elemNum + 2) << endl;
						//cout << "dTydHz2:" << calcElementsVector[elemNum]->dTdH[iOmega](1).coeffRef(0, 3 * numOfCalcElements + 3 * elemNum + 2) << endl;

						//complex<double> dH = 0.001;
						//calcElementsVector[elemNum]->H[itr].coeffRef(direcH) += dH;
						//calcElementsVector[elemNum]->CalcT(iOmega);
						//double postJ = CalcDataMisfit();
						//cout<< "numerical Real:" << (postJ - preJ) / dH << endl;
						//calcElementsVector[elemNum]->H[itr].coeffRef(direcH) -= dH;
						//dH = complex < double>( 0 , 0.001);
						//calcElementsVector[elemNum]->H[itr].coeffRef(direcH) += dH;
						//calcElementsVector[elemNum]->CalcT(iOmega);
						//postJ = CalcDataMisfit();
						//cout <<"numerical Imag:" << (postJ - preJ) / dH << endl;
						//calcElementsVector[elemNum]->H[itr].coeffRef(direcH) -= dH;
						//calcElementsVector[elemNum]->CalcT(iOmega);
						//cout << "analysis:" << 2.0 * (conj(dTtmp)*dTdHtmp) << endl;

					}
				}
			}
		}
	}
	// Todo::他のテンソル量を逆解析する場合はここに足す
	else {
		Element::Element* element = locElement;

		if (isInvertedDistortion) {
			Zcalc = element->distortionMatrix * element->Z[iOmega];
			dZdHCalc[threadID](0, 0) = element->distortionMatrix.coeff(0, 0) * element->dZdH[iOmega](0, 0) +
				element->distortionMatrix.coeff(0, 1) * element->dZdH[iOmega](1, 0);
			dZdHCalc[threadID](0, 1) = element->distortionMatrix.coeff(0, 0) * element->dZdH[iOmega](0, 1) +
				element->distortionMatrix.coeff(0, 1) * element->dZdH[iOmega](1, 1);
			dZdHCalc[threadID](1, 0) = element->distortionMatrix.coeff(1, 0) * element->dZdH[iOmega](0, 0) +
				element->distortionMatrix.coeff(1, 1) * element->dZdH[iOmega](1, 0);
			dZdHCalc[threadID](1, 1) = element->distortionMatrix.coeff(1, 0) * element->dZdH[iOmega](0, 1) +
				element->distortionMatrix.coeff(1, 1) * element->dZdH[iOmega](1, 1);

		}
		else {
			Zcalc = element->Z[iOmega];
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					dZdHCalc[threadID](ii, jj) = element->dZdH[iOmega](ii, jj);
				}
			}
		}
		for (int ii = 0; ii < 2; ii++) {
			for (int jj = 0; jj < 2; jj++) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dZdH[iOmega](ii, jj), 0); it; ++it)
				{
					int iCol = it.col();
					std::complex<double> dZtmp = Zcalc.coeff(ii, jj);
					std::complex<double> dZdHtmp = dZdHCalc[threadID](ii, jj).coeff(0, iCol);

					dJdH[threadID].coeffRef(iCol).real(dJdH[threadID].coeffRef(iCol).real() + 2.0 * (conj(dZtmp) * dZdHtmp / boundary->omega[iOmega] / ConstantValues::mu).real()); //App Resis
					//f2 << iCol << " " << dZtmp << " " << dZdHtmp << "middlle" << endl;

					//Imag Part

					dJdH[threadID].coeffRef(iCol).imag(dJdH[threadID].coeffRef(iCol).imag() + 2.0 * (conj(dZtmp) * dZdHtmp / boundary->omega[iOmega] / ConstantValues::mu).imag()); //App Resis
				}

			}
		}

	}
	
	if (isDirectSolver) {
		
		cout<<"Direct Solver in Non Intel MKL Version is not Implemented."<<endl;
		exit(1);
		

	}
	else {
		//====Calc adjoint(∂R/∂H) = adjoint(A of Ax=b) =======================
			//====Hiterごとに分けて計算できる(RがH1とH2で独立）ので分けて計算
			//====H1========
		time_t start_t = time(NULL);

		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> sol(2);
		sol[0].resize(3 * numOfCalcElements);
		sol[1].resize(3 * numOfCalcElements);
		sol[0].setZero();
		sol[1].setZero();

		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> rhs(2);
		rhs[0].resize(3 * numOfCalcElements);
		rhs[1].resize(3 * numOfCalcElements);
		rhs[0].setZero();
		rhs[1].setZero();

		//Reordering from Hx0,Hy0,Hz0,Hx1,Hy1,Hz1,...Hxn,Hyn,Hzn to Hx0,Hx1,...Hxn,Hy0,Hy1,...Hyn,Hz0,Hz1,...Hzn
		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			int row = originalOrderToSolverOrder[i];
			rhs[0].coeffRef(row) = conj(dJdH[threadID].coeff(i));
		}

		//Eigen::SparseMatrix<double, Eigen::RowMajor> prunedMat{ 3 * numOfCalcElements,3 * numOfCalcElements };
		//prunedMat = *globalMatrixNoOmegaTerm;
		//prunedMat.prune(0.0);
		//prunedMat.makeCompressed();

		//divergence correction in subsurface,now which is included in MakeMatrix();


		//Eigen::SparseMatrix<double, Eigen::RowMajor> factorMat;
		//factorMat.resize(numOfCalcElements, numOfCalcElements);
		//factorMat.reserve(Eigen::VectorXi::Constant(numOfCalcElements, 1));
		//for (int i = 0; i < numOfCalcElements; i++) {
		//	if (!calcElementsVector[i]->calcGradDivOperationElement && calcElementsVector[i]->boundary == "NOT_BOUNDARY") {
		//		//double ave = 0;
		//		double resis;
		//		if (calcElementsVector[i]->property->type == Property::Property::AIR) {
		//			/*resis = 0.0;
		//			for (int j = 0; j < 6; j++) {

		//				resis += (*calcElementsVector[i]->resistivitySurface)[j] / 6.0;
		//			}*/
		//			resis = 0.0;
		//			for (int j = 0; j < 6; j++) {
		//				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*(calcElementsVector[i]->resistivitySurfaceCoeff[j]), 0); it; ++it) {
		//					if (resis < calcElementsVector[it.col()]->resistivity) {
		//						resis = calcElementsVector[it.col()]->resistivity;
		//					}
		//				}
		//				//ave += (*calcElementsVector[i]->resistivitySurface)[j];
		//			}
		//			calcElementsVector[i]->factorForDivCorr = resis;
		//		}
		//		else {
		//			resis = 1e30;
		//			for (int j = 0; j < 6; j++) {
		//				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*(calcElementsVector[i]->resistivitySurfaceCoeff[j]), 0); it; ++it) {
		//					if (resis > calcElementsVector[it.col()]->resistivity) {
		//						resis = calcElementsVector[it.col()]->resistivity;
		//					}
		//				}
		//				//ave += (*calcElementsVector[i]->resistivitySurface)[j];
		//			}
		//			calcElementsVector[i]->factorForDivCorr = invSettings->safetyFactor * resis;// *min(1.0, sqrt(boundary->omega[iOmega]));
		//		}

		//		/*resis = 0.0;
		//		for (int j = 0; j < 6; j++) {
		//			resis += (*calcElementsVector[i]->resistivitySurface)[j]/6.0;
		//		}*/
		//		//double mid = exp((log(calcElementsVector[i]->resistivity) + log(boundary->omega[iOmega] * mu)) / 2.0);




		//		factorMat.coeffRef(i, i) = -calcElementsVector[i]->factorForDivCorr;
		//		/*factorMat.coeffRef(3*i, 3*i) = -calcElementsVector[i]->factorForDivCorr;
		//		factorMat.coeffRef(3 * i + 1, 3 * i + 1) = -calcElementsVector[i]->factorForDivCorr;
		//		factorMat.coeffRef(3 * i +2, 3 * i + 2) = -calcElementsVector[i]->factorForDivCorr;*/
		//		//*minResis;// calcElementsVector[i]->resistivity;
		//	}
		//}

		//precond

		/*Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> precondMat{ 3 * numOfCalcElements,3 * numOfCalcElements };


		vector < Eigen::Triplet<complex<double>>> precondMatTriplet;
		precondMatTriplet.reserve(3 * numOfCalcElements * 30);
		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			Element::Element* element = calcElementsVector[int(solverOrderToOriginalOrder[i] / 3)];
			double dv = element->dv;
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(*globalMatrixNoOmegaTerm, i); it; ++it) {


				int solRow = i;
				int solCol = it.col();

				int oriRow = solverOrderToOriginalOrder[solRow];
				int oriCol = solverOrderToOriginalOrder[solCol];

				if (solRow < numOfCalcElements && solCol >= numOfCalcElements) {
					continue;
				}
				else if ((solRow >= numOfCalcElements && solRow < 2 * numOfCalcElements) && (solCol < numOfCalcElements || solCol >= 2 * numOfCalcElements)) {
					continue;
				}
				else if (solRow >= 2 * numOfCalcElements && solCol < 2 * numOfCalcElements) {
					continue;
				}

				Eigen::Triplet<complex<double>> val(solCol,solRow,
					globalMatrixNoOmegaTerm->coeff(solRow, solCol) +
					double(int(element->boundary == "NOT_BOUNDARY" && !element->calcGradDivOperationElement && oriRow == oriCol)) *
					complex<double>(0, -boundary->omega[iOmega] * mu * dv) +

					double(int(element->boundary == "NOT_BOUNDARY" && element->calcGradDivOperationElement && oriRow == oriCol)) *
					complex<double>(0, -1.0 / element->resistivity * boundary->omega[iOmega] * mu * dv));

				precondMatTriplet.push_back(val);
			}

		}


		precondMat.setFromTriplets(precondMatTriplet.begin(), precondMatTriplet.end());
		precondMat.pruned();
		iterativeSolverVector[boundary->omega.size() +iOmega]->precond.compute(precondMat,false);*/

		// end precond



		//Eigen::SparseMatrix<double, Eigen::RowMajor> globalMatrixToBeSolvedReal;
		//globalMatrixToBeSolvedReal.resize(3 * numOfCalcElements, 3 * numOfCalcElements);
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> globalMatrixToBeSolvedImag;
		globalMatrixToBeSolvedImag.resize(3 * numOfCalcElements, 3 * numOfCalcElements);
		//globalMatrixToBeSolved.reserve(Eigen::VectorXi::Constant(4*numOfCalcElements,200));

		//vector < Eigen::Triplet<double>> globalMatrixToBeSolvedRealTriplet;
		//globalMatrixToBeSolvedRealTriplet.reserve(globalMatrixNoOmegaTerm->nonZeros());
		vector < Eigen::Triplet<complex<double>>> globalMatrixToBeSolvedImagTriplet;
		globalMatrixToBeSolvedImagTriplet.reserve(globalMatrixNoOmegaTerm->nonZeros());
		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			Element::Element* element = calcElementsVector[int(i / 3)];
			double dv = element->dv;

			int oriRow = i;
			int oriCol = i;

			int solRow = originalOrderToSolverOrder[oriRow];
			int solCol = originalOrderToSolverOrder[oriCol];


			//Eigen::Triplet<double> valR(solCol,solRow,
			//	globalMatrixNoOmegaTerm->coeff(oriRow, oriCol));

			Eigen::Triplet<complex<double>> valI(solCol, solRow,
				(double(int(element->boundary == "NOT_BOUNDARY" && !element->calcGradDivOperationElement)) *
				complex<double>(0, -boundary->omega[iOmega] * mu * dv) +

				double(int(element->boundary == "NOT_BOUNDARY" && element->calcGradDivOperationElement )) *
				complex<double>(0, 1.0 / element->resistivity * boundary->omega[iOmega] * mu * dv)));

			//globalMatrixToBeSolvedRealTriplet.push_back(valR);
			if (valI.value() != 0.0) {
				globalMatrixToBeSolvedImagTriplet.push_back(valI);
			}


		}

		//divergence correction in subsurface, which is now included in MakeMatrix();



		/*for (int i = 0; i < 3 * numOfCalcElements; i++) {
			Element::Element* element = calcElementsVector[int(i / 3)];
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(divergenceCorrection->sumDivHdSMatrix, i); it; ++it) {
				int oriRow = i;
				int oriCol = it.col();
				Eigen::Triplet<double> val(originalOrderToSolverOrder[oriCol],originalOrderToSolverOrder[i],
					factorMat.coeff(oriRow / 3, oriRow / 3) * divergenceCorrection->sumDivHdSMatrix.coeff(oriRow, oriCol));

				globalMatrixToBeSolvedRealTriplet.push_back(val);
			}
		}*/



		//globalMatrixToBeSolvedReal.setFromTriplets(globalMatrixToBeSolvedRealTriplet.begin(), globalMatrixToBeSolvedRealTriplet.end());
		//globalMatrixToBeSolvedReal.pruned();
		globalMatrixToBeSolvedImag.setFromTriplets(globalMatrixToBeSolvedImagTriplet.begin(), globalMatrixToBeSolvedImagTriplet.end());
		globalMatrixToBeSolvedImag.pruned();


		bool infFlag = false;
		
		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			int solRow = originalOrderToSolverOrder[i];
			sol[0].coeffRef(solRow) = resultAdjointVector.coeff((2 * iOmega) * 3 * numOfCalcElements + i);
			if (!std::isfinite(sol[0].coeffRef(solRow).real()) || !std::isfinite(sol[0].coeffRef(solRow).imag())) {
				infFlag = true;
			}
		}
		if (infFlag) {
			sol[0].setZero();
		}
		//====H2========

		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			int solRow = originalOrderToSolverOrder[i];
			rhs[1].coeffRef(solRow) = conj(dJdH[threadID].coeff(3 * numOfCalcElements + i));
		}

		infFlag = false;
		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			int solRow = originalOrderToSolverOrder[i];
			sol[1].coeffRef(solRow) = resultAdjointVector.coeff((2 * iOmega + 1) * 3 * numOfCalcElements + i);
			if (!std::isfinite(sol[1].coeffRef(solRow).real()) || !std::isfinite(sol[1].coeffRef(solRow).imag())) {
				infFlag = true;
			}
		}	

		if (infFlag) {
			sol[1].setZero();
		}

		iterativeSolverVector[boundary->omega.size() + iOmega]->omega = boundary->omega[iOmega];
		iterativeSolverVector[boundary->omega.size() + iOmega]->m_maxIteration = invSettings->maxIterationBiCGSafe;
		iterativeSolverVector[boundary->omega.size() + iOmega]->m_relSolTol = invSettings->toleranceIterativeSolver;
		iterativeSolverVector[boundary->omega.size() + iOmega]->m_relSolTolForAdjoint = invSettings->toleranceIterativeSolverAdjoint;
		if (invSettings->toleranceIterativeSolverAdjoint < 0.0) {
			iterativeSolverVector[boundary->omega.size() + iOmega]->m_relSolTolForAdjoint = iterativeSolverVector[boundary->omega.size() + iOmega]->m_relSolTol;
		}
		//==============Solve H1,H2 together===============================================
		iterativeSolverVector[boundary->omega.size() + iOmega]->precond.compute(*globalMatrixNoOmegaTermAdjoint, globalMatrixToBeSolvedImag);
		/*if (useMultGrid) {
			iterativeSolverVector[boundary->omega.size() + iOmega]->precond_multi.compute(*globalMatrixNoOmegaTermAdjoint, globalMatrixToBeSolvedImag);

		}
		else {
			iterativeSolverVector[boundary->omega.size() + iOmega]->precond.compute(*globalMatrixNoOmegaTermAdjoint, globalMatrixToBeSolvedImag);

		}*/
		iterativeSolverVector[boundary->omega.size() + iOmega]->solve(globalMatrixNoOmegaTermAdjoint, &globalMatrixToBeSolvedImag, &sol, &rhs, iterativeSolverVector[boundary->omega.size() + iOmega]->m_relSolTolForAdjoint, false);
		
		if (iterativeSolverVector[boundary->omega.size() + iOmega]->m_maxIteration == iterativeSolverVector[boundary->omega.size() + iOmega]->m_iters) {
			std::cout << "WARNING!!!!!!!!!!!!!!!! In BiCGSafe Solution NOT Converged!!!" << std::endl;
		}
		std::cout << "In BiCGSafe Period:     " << 2*pi/boundary->omega[iOmega] << std::endl;
		std::cout << "In BiCGSafe #iterations:     " << iterativeSolverVector[boundary->omega.size() + iOmega]->m_iters << std::endl;
		std::cout << "In BiCGSafe estimated error: " << iterativeSolverVector[boundary->omega.size() + iOmega]->m_error << std::endl;
		std::cout << "In BiCGSafe Last Iteration, Relative Change of Solution:" << iterativeSolverVector[boundary->omega.size() + iOmega]->m_lastRelativeSolChange << std::endl;
				

		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			int solRow = originalOrderToSolverOrder[i];
			resultAdjointVector.coeffRef((2 * iOmega) * 3 * numOfCalcElements + i) = sol[0].coeffRef(solRow);
			resultAdjointVector.coeffRef((2 * iOmega + 1) * 3 * numOfCalcElements + i) = sol[1].coeffRef(solRow);
		}


		
		time_t end_t = time(NULL);
		cout << "Calculate Lambda of H2. #Omega:" << iOmega << endl;
		std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;

		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			int solRow = originalOrderToSolverOrder[i];
			lambdaEachOmega[iOmega]->coeffRef(i) = sol[0].coeff(solRow);
			lambdaEachOmega[iOmega]->coeffRef(3 * numOfCalcElements + i) = sol[1].coeff(solRow);
		}
		
	}
}

void Analysis::Analysis::SearchRelatedCalcElements() {
	for (int i = 0; i < calcElementsVector.size(); i++) {
		calcElementsVector[i]->SearchRelatedCalcElements(&elements);
		for (auto itr = calcElementsVector[i]->relatedNeighborCalcElementsMap.begin(); itr != calcElementsVector[i]->relatedNeighborCalcElementsMap.end(); itr++) {
			Element::Element* tmpElement = itr->second;
			calcElementsVector[i]->relatedNeighborCalcElementsVector.push_back(tmpElement);
		}
	}
}

void Analysis::Analysis::CalcLambdaDRDRho(const ub::vector<complex<double>>* rhoVec, const vector<Eigen::VectorXcd>* HresultItr,int iOmega) {
	
	int numThreads = omp_get_max_threads();

	//Analysis::Analysis::CalcLambdaDRDRhoParameters valForLambdaDRDRho;
	////if (valForLambdaDRDRho.isInitialized == false) {
	//	int maxLayer = 0;
	//	valForLambdaDRDRho.threadIDGroup.resize(numThreads);
	//	int numOfNotBoundaryElements = 0;
	//	for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
	//		Element::Element* element = *itr;
	//		if (element->layer > maxLayer) {

	//			maxLayer = element->layer;
	//		}
	//	}
	//	valForLambdaDRDRho.maxLayer = maxLayer;

	//	for (int iLayer = 0; iLayer <= maxLayer; iLayer++) {
	//		vector < Element::Element* > layerElementsVector;
	//		for (int i = 0; i < calcElementsVector.size(); i++) {
	//			Element::Element* element = calcElementsVector[i];
	//			if (element->layer == iLayer && element->boundary == "NOT_BOUNDARY") {
	//				layerElementsVector.push_back(element);
	//				numOfNotBoundaryElements++;
	//			}
	//		}
	//		valForLambdaDRDRho.sameLayerElementsVector.push_back(layerElementsVector);
	//	}

	//	for (int iLayer = maxLayer; iLayer >= 0; iLayer--) {
	//		vector < Element::Element* >tmpVector = valForLambdaDRDRho.sameLayerElementsVector[iLayer];
	//		for (int i = 0; i < tmpVector.size(); i++) {
	//			Element::Element* element = valForLambdaDRDRho.sameLayerElementsVector[iLayer][i];
	//			valForLambdaDRDRho.threadIDGroup[i%numThreads].push_back(element->calcID);
	//		}
	//	}
	//	valForLambdaDRDRho.isInitialized = true;
	////}


	//Multi Thread
	//vector<Eigen::VectorXcd> lambdaDRDRhoEachThread(numThreads);
	//for (int i = 0; i < numThreads; i++) {
	//	lambdaDRDRhoEachThread[i].resize(numOfInvertedResistivityElements);
	//	lambdaDRDRhoEachThread[i].setZero();
	//}

		
	//}


	time_t start_t = time(NULL);

	//vector<Eigen::VectorXcd> lambdaDRDRhoEachThread(numThreads);
	//for (int i = 0; i < numThreads; i++) {
	//	lambdaDRDRhoEachThread[i] = Eigen::VectorXcd(numOfInvertedResistivityElements);
	//	lambdaDRDRhoEachThread[i].setZero();
	//}

	
//	for (int iLayer = valForLambdaDRDRho.maxLayer; iLayer >= 0; iLayer--) {
//#pragma omp parallel for
//		for (int i = 0; i < numThreads; i++) {
//			for (int j = 0; j < valForLambdaDRDRho.threadIDGroup[i].size(); j++) {
//				Element::Element* element = calcElementsVector[valForLambdaDRDRho.threadIDGroup[i][j]];
//				if (element->layer == iLayer) {
//					element->CalcLambdaDSumNCrossRhoRotHdSDRho(&elements, rhoVecUb, HresultItr, calcElementsVector, numOfCalcElements, numOfInvertedResistivityElements, lambdaEachOmega, &lambdaDRDRhoEachThread[i]);
//				}
//			}
//		}
//	}

	for (int i = 0; i < numThreads; i++) {
		if (lambdaDRDRhoEachThread[i].size() == 0) {
			lambdaDRDRhoEachThread[i].resize(numOfInvertedResistivityElements);
		}
		lambdaDRDRhoEachThread[i].setZero();
	}

	//Eigen::setNbThreads(1);
#pragma omp parallel for
	for (int i = 0; i < numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];
		element->CalcLambdaDSumNCrossRhoRotHdSDRho(&elements, rhoVec, HresultItr, &calcElementsVector, numOfCalcElements, numOfInvertedResistivityElements, lambdaEachOmega[iOmega], &lambdaDRDRhoEachThread[omp_get_thread_num()]);
		//element->CalcLambdaDSumNCrossRhoRotHdSDRho(&elements, rhoVec, HresultItr, calcElementsVector, numOfCalcElements, numOfInvertedResistivityElements, lambdaEachOmega, &lambdaDRDRho);

	}

	time_t end_t = time(NULL);
	std::cout << "Parallel Part Calculation Time:" << end_t - start_t << " Seconds." << endl;

	for (int i = 0; i < numThreads; i++) {
		lambdaDRDRho += lambdaDRDRhoEachThread[i];
	}


	//Single Thread
	//time_t start_t = time(NULL);
	//for (int iLayer = valForLambdaDRDRho.maxLayer; iLayer >= 0; iLayer--) {
	//	for (int j = 0; j < numOfCalcElements; j++) {
	//		Element::Element* element = calcElementsVector[j];
	//		if (element->layer == iLayer) {
	//			element->CalcLambdaDSumNCrossRhoRotHdSDRho(&elements, rhoVecUb, HresultItr, calcElementsVector, numOfCalcElements, numOfInvertedResistivityElements, lambdaEachOmega, lambdaDRDRho);
	//		}
	//	}
	//}

	end_t = time(NULL);
	std::cout << "Total Calc Lambda DRDRho Time:" << end_t - start_t << " Seconds." << endl;
}

void Analysis::Analysis::SetInvertedElements() {
	//テスト
	//for (int i = 0; i < numOfObsPointElements; i++) {
	//	obsPointElements[i]->property = propertiesVector[5];
	//	obsPointElements[i]->neighborElements[0 + 3 * 1 + 9 * 1]->property = propertiesVector[5];
	//	obsPointElements[i]->neighborElements[2 + 3 * 1 + 9 * 1]->property = propertiesVector[5];
	//	obsPointElements[i]->neighborElements[1 + 3 * 0 + 9 * 1]->property = propertiesVector[5];
	//	obsPointElements[i]->neighborElements[1 + 3 * 2 + 9 * 1]->property = propertiesVector[5];
	//	obsPointElements[i]->neighborElements[1 + 3 * 1 + 9 * 0]->property = propertiesVector[5];
	//	obsPointElements[i]->neighborElements[1 + 3 * 1 + 9 * 2]->property = propertiesVector[5];
	//	cout << propertiesVector[5]->type << endl;
	//}
	//テスト終わり
	//テスト
	/*for (int i = 0; i < numOfCalcElements; i++) {
		calcElementsVector[i]->property = propertiesVector[5];
	}
	for (int i = 0; i < numOfObsPointElements; i++) {
		obsPointElements[i]->property = propertiesVector[1];
		obsPointElements[i]->neighborElements[0 + 3 * 1 + 9 * 1]->property = propertiesVector[1];
		obsPointElements[i]->neighborElements[2 + 3 * 1 + 9 * 1]->property = propertiesVector[1];
		obsPointElements[i]->neighborElements[1 + 3 * 0 + 9 * 1]->property = propertiesVector[1];
		obsPointElements[i]->neighborElements[1 + 3 * 2 + 9 * 1]->property = propertiesVector[1];
		obsPointElements[i]->neighborElements[1 + 3 * 1 + 9 * 0]->property = propertiesVector[1];
		obsPointElements[i]->neighborElements[1 + 3 * 1 + 9 * 2]->property = propertiesVector[1];
	}*/
	//テスト終わり

	//numOfInvertedResistivityElements = 0;
	//invertedRhoIDToElementVector.clear();
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	Element::Element* element = calcElementsVector[i];
	//	if (element->boundary == "NOT_BOUNDARY" && (element->property->type == Property::Property::NORMAL || element->isAirGroundBoundaryCell)){// && element->boundary=="NOT_BOUNDARY" && element->isAirGroundBoundaryCell == false) {
	//		//elements of isAirGroundBoundaryCell and Boundary are inverted, but not independent.so here, they are included in inverted elements group.
	//		element->invertedRhoElementsID = numOfInvertedResistivityElements;
	//		invertedRhoIDToElementMap[numOfInvertedResistivityElements] = element;
	//		invertedRhoIDToElementVector.push_back(element);
	//		element->isInvertedElement = true;
	//		numOfInvertedResistivityElements++;
	//	}
	//	else {
	//		element->invertedRhoElementsID = -1;
	//	}
	//}
	numOfInvertedResistivityElements = 0;
	invertedRhoIDToElementVector.clear();
	for (int i = 0; i < numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];
		if (element->boundary == "NOT_BOUNDARY" && (element->property->type == Property::Property::NORMAL)) {// && element->boundary=="NOT_BOUNDARY" && element->isAirGroundBoundaryCell == false) {
			//elements of isAirGroundBoundaryCell and Boundary are inverted, but not independent.so here, they are included in inverted elements group.
			element->invertedRhoElementsID = numOfInvertedResistivityElements;
			invertedRhoIDToElementMap[numOfInvertedResistivityElements] = element;
			invertedRhoIDToElementVector.push_back(element);
			element->isInvertedElement = true;
			numOfInvertedResistivityElements++;
		}
		else {
			element->invertedRhoElementsID = -1;
		}
	}
	for (int i = 0; i < numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];
		if (element->isAirGroundBoundaryCell && element->neighborElements[1 + 3 * 1 + 9 * 2]->isInvertedElement) {// && element->boundary=="NOT_BOUNDARY" && element->isAirGroundBoundaryCell == false) {
			element->invertedRhoElementsID = numOfInvertedResistivityElements;
			invertedRhoIDToElementMap[numOfInvertedResistivityElements] = element;
			invertedRhoIDToElementVector.push_back(element);
			element->isInvertedElement = true;
			numOfInvertedResistivityElements++;
		}

	}
}


void Analysis::Analysis::SetDKDRhoElements() {
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		Element::Element* element = invertedRhoIDToElementVector[i];
		Eigen::SparseMatrix<double, Eigen::RowMajor> tmpDKdRho{ 3 * numOfCalcElements, 3 * numOfCalcElements };
		element->dKDRho=tmpDKdRho;
		//Eigen::SparseMatrix<double, Eigen::RowMajor> tmp{ 1, 3 * numOfCalcElements };
		//tmp.setZero();
		//tmp.makeCompressed();
		//tmp.prune(1e-9);
		//tmp.data().squeeze();
		//tmp.reserve(0);
		//tmpDKdRho.row(j) = tmp;

		//calcElementsVector[i]->dKDRho.makeCompressed();
		//calcElementsVector[i]->dKDRho.prune(1e-9);
		//calcElementsVector[i]->dKDRho.data().squeeze();
		//calcElementsVector[i]->dKDRho.reserve(81);
		//calcElementsVector[i]->dKDRho.resize(3 * numOfCalcElements, 3 * numOfCalcElements);

	}
}
//void Analysis::Analysis::CalcDKDRhoElements() {
//	for (int i = 0; i < numOfInvertedRhoElements; i++) {
//		invertedRhoIDToElementVector[i]->CalcDKDRho(&elements, &invertedRhoIDToElementMap, numOfCalcElements, numOfInvertedRhoElements);
//	}
//}

double Analysis::Analysis::CalcDataMisfit(bool onePointMode) {
	dataMisfit = 0.0;
	if (onePointMode == false) {
		//Impedance Tensor
		for (int i = 0; i < numOfObsPointElements; i++) {
			Element::Element* element = obsPointElements[i];
			if (element->isInversionImpedance == true) {
				for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
					Eigen::Matrix2cd Zcalc;


					if (isInvertedDistortion) {
						Zcalc = element->distortionMatrix * element->Z[iOmega];

					}
					else {
						Zcalc = element->Z[iOmega];

					}
					for (int ii = 0; ii < 2; ii++) {
						for (int jj = 0; jj < 2; jj++) {
							std::complex<double> dZtmp = Zcalc.coeff(ii, jj) - element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj);
							double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
							double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));

							if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0 ) {
								dZtmp.real(dZtmp.real() / epsReal);
							}
							else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
								dZtmp.real(0.0);
							}
							else {
								//そのまま
							}
							if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0 ) {
								dZtmp.imag(dZtmp.imag() / epsImag);
							}
							else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
								dZtmp.imag(0.0);
							}
							else {
								//そのまま
							}
							

							dataMisfit += (dZtmp * conj(dZtmp)).real();

						}
					}
				}
			}
		}

		//Tipper 
		for (int i = 0; i < numOfObsPointElements; i++) {
			Element::Element* element = obsPointElements[i];
			if (element->isInversionTipper == true) {
				for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
					for (int ii = 0; ii < 2; ii++) {
						std::complex<double> dTtmp = element->T[iOmega].coeff(ii) - element->tipperObsData->TobsVector[iOmega].coeff(ii);
						double epsReal = std::abs(element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii));
						double epsImag = std::abs(element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii));

						if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) > 0 ) {
							dTtmp.real(dTtmp.real() / epsReal);
						}
						else if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) <= 0) {
							dTtmp.real(0.0);
						}
						else {
							//そのまま
						}
						if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) > 0 ) {
							dTtmp.imag(dTtmp.imag() / epsImag);
						}
						else if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) <= 0) {
							dTtmp.imag(0.0);
						}
						else {
							//そのまま
						}

						dataMisfit += (dTtmp * conj(dTtmp)).real();


						//cout <<"Tcalc Tobs:"<< element->T[iOmega].coeff(ii) << " " << element->tipperObsData->TobsVector[iOmega].coeff(ii) << endl;
						//cout << "epsReal:" << epsReal << endl;
						//cout << "epsImag:" << epsImag << endl;
						//cout << "misfit" << (dTtmp*conj(dTtmp)).real()<< endl;


					}

				}
			}
		}
		// Todo::他のテンソル量を逆解析する場合はここに足す
		//dataMisfit /= numOfObsData;
	}
	else {
		//Impedance Tensor

		Element::Element* element = locElement;
		for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
			Eigen::Matrix2cd Zcalc;


			if (isInvertedDistortion) {
				Zcalc = element->distortionMatrix * element->Z[iOmega];

			}
			else {
				Zcalc = element->Z[iOmega];

			}
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					std::complex<double> dZtmp = Zcalc.coeff(ii, jj);

					dataMisfit += (dZtmp * conj(dZtmp)).real() / boundary->omega[iOmega] / ConstantValues::mu; //App Resis;
				}
			}
				
		}
	}
	return dataMisfit;
}
double Analysis::Analysis::CalcRoughningMatrixPenalty() {
	Eigen::VectorXd rhoVec{ numOfInvertedResistivityElements };

	
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		//rhoVec.coeffRef(i) = log10(invertedRhoIDToElementVector[i]->resistivity);
		rhoVec.coeffRef(i) = log(invertedRhoIDToElementVector[i]->resistivity);
		//rhoVec.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity;
		//rhoVec.coeffRef(i) = log10(invertedRhoIDToElementVector[i]->resistivity) - log10(invertedRhoIDToElementVector[i]->initialResistivity);
	}


	if (useL1Norm) {
		
		double sum = 0.0;
		for (int i = 0; i < rougheningMatrix->outerSize(); ++i) {
			double val = std::abs(rougheningMatrix->row(i) * rhoVec);
			if (val< epsForL1Norm){
				val = 0.0;
			}
			sum += val;
		}
		sum = rateL1Norm * sum;
		mWTWm = 0.0;
		mWTWm = rhoVec.transpose() * rougheningMatrix->transpose() * (*rougheningMatrix) * rhoVec;
		mWTWm = (1 - rateL1Norm) * mWTWm;
		sum += mWTWm;
		return sum;
	}
	else {
		mWTWm = 0.0;
		mWTWm = rhoVec.transpose() * rougheningMatrix->transpose() * (*rougheningMatrix) * rhoVec;
		return mWTWm;
	}

	
}

void Analysis::Analysis::CalcDZDHElements(const ub::vector<kv::complex<double>>* HVecUb, int iOmega) {
	for (int i = 0; i < numOfObsPointElements; i++) {
		Element::Element* element = obsPointElements[i];
		if (element->isInversionImpedance) {
			element->CalcDZDH(HVecUb, &elements, numOfCalcElements, iOmega);
		}
		
	}
	for (int i = 0; i < numOfLocationCalcElements; i++) {
		Element::Element* element = locPointElements[i];
		element->CalcDZDH(HVecUb, &elements, numOfCalcElements, iOmega);
	}

}
void Analysis::Analysis::CalcDTDHElements(int iOmega) {
	for (int i = 0; i < numOfObsPointElements; i++) {
		Element::Element* element = obsPointElements[i];
		if (element->isInversionTipper) {
			element->CalcDTDH(numOfCalcElements, iOmega);
			//test
			//Eigen::Vector2cd preT = element->T[iOmega];
			//std::complex<double>dH = 0.001;
			//element->H[0].coeffRef(0) += dH;
			//element->CalcT(iOmega);
			//Eigen::Vector2cd postT = element->T[iOmega];
			//cout << "dT/dHx1=" << (postT - preT) / dH <<" "<<element->dTdH[iOmega](0).coeff(0,3*element->calcID)
			//	<< " " << element->dTdH[iOmega](1).coeff(0,3 * element->calcID) << endl;
			//element->H[0].coeffRef(0) -= dH;
			//element->CalcT(iOmega);

			//element->H[0].coeffRef(1) += dH;
			//element->CalcT(iOmega);
			//postT = element->T[iOmega];
			//cout << "dT/dHy1=" << (postT - preT) / dH << " " << element->dTdH[iOmega](0).coeff(0,3 * element->calcID+1) 
			//	<< " " << element->dTdH[iOmega](1).coeff(0, 3 * element->calcID + 1) << endl;
			//element->H[0].coeffRef(1) -= dH;
			//element->CalcT(iOmega);

			//element->H[0].coeffRef(2) += dH;
			//element->CalcT(iOmega);
			//postT = element->T[iOmega];
			//cout << "dT/dHz1=" << (postT - preT) / dH << " " << element->dTdH[iOmega](0).coeff(0,3 * element->calcID+2) 
			//	<< " " << element->dTdH[iOmega](1).coeff(0, 3 * element->calcID + 2) << endl;
			//element->H[0].coeffRef(2) -= dH;
			//element->CalcT(iOmega);

			//element->H[1].coeffRef(0) += dH;
			//element->CalcT(iOmega);
			//postT = element->T[iOmega];
			//cout << "dT/dHx2=" << (postT - preT) / dH << " " << element->dTdH[iOmega](0).coeff(0,3*numOfCalcElements+ 3 * element->calcID) 
			//	<< " " << element->dTdH[iOmega](1).coeff(0, 3 * numOfCalcElements + 3 * element->calcID) << endl;
			//element->H[1].coeffRef(0) -= dH;
			//element->CalcT(iOmega);

			//element->H[1].coeffRef(1) += dH;
			//element->CalcT(iOmega);
			//postT = element->T[iOmega];
			//cout << "dT/dHy2=" << (postT - preT) / dH << " " << element->dTdH[iOmega](0).coeff(0,3 * numOfCalcElements+3 * element->calcID + 1)
			//	<< " " << element->dTdH[iOmega](1).coeff(0, 3 * numOfCalcElements + 3 * element->calcID + 1) << endl;
			//element->H[1].coeffRef(1) -= dH;
			//element->CalcT(iOmega);

			//element->H[1].coeffRef(2) += dH;
			//element->CalcT(iOmega);
			//postT = element->T[iOmega];
			//cout << "dT/dHz2=" << (postT - preT) / dH << " " << element->dTdH[iOmega](0).coeff(0,3 * numOfCalcElements+3 * element->calcID + 2) 
			//	<< " " << element->dTdH[iOmega](1).coeff(0, 3 * numOfCalcElements + 3 * element->calcID + 2) << endl;
			//element->H[1].coeffRef(2) -= dH;
			//element->CalcT(iOmega);
		}

	}

}

void Analysis::Analysis::CalcDDataMisfitDRho(bool onePointMode) { 
	
	//dDataMisfitDRho.resize(numOfInvertedResistivityElements);
	//dDataMisfitDRho.setZero();


	//Impedance Tensor
	if (onePointMode == false) {
		for (int j = 0; j < numOfObsPointElements; j++) {
			Element::Element* element = obsPointElements[j];
			if (element->isInversionImpedance == true) {
				for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
					int ID = element->impedanceObsID * boundary->omega.size() + iOmega;
					if (useImpedanceDataArray.coeff(ID) == 0.0) {
						continue;
					}
					Eigen::Matrix2cd Zcalc;

					if (isInvertedDistortion) {
						Zcalc = element->distortionMatrix * element->Z[iOmega];
						dZdRhoCalc(0, 0) = element->distortionMatrix.coeff(0, 0) * element->dZdRho[iOmega](0, 0) +
							element->distortionMatrix.coeff(0, 1) * element->dZdRho[iOmega](1, 0);
						dZdRhoCalc(0, 1) = element->distortionMatrix.coeff(0, 0) * element->dZdRho[iOmega](0, 1) +
							element->distortionMatrix.coeff(0, 1) * element->dZdRho[iOmega](1, 1);
						dZdRhoCalc(1, 0) = element->distortionMatrix.coeff(1, 0) * element->dZdRho[iOmega](0, 0) +
							element->distortionMatrix.coeff(1, 1) * element->dZdRho[iOmega](1, 0);
						dZdRhoCalc(1, 1) = element->distortionMatrix.coeff(1, 0) * element->dZdRho[iOmega](0, 1) +
							element->distortionMatrix.coeff(1, 1) * element->dZdRho[iOmega](1, 1);
					}
					else {
						Zcalc = element->Z[iOmega];
						for (int ii = 0; ii < 2; ii++) {
							for (int jj = 0; jj < 2; jj++) {
								dZdRhoCalc(ii, jj) = element->dZdRho[iOmega](ii, jj);
							}
						}
					}

					for (int ii = 0; ii < 2; ii++) {
						for (int jj = 0; jj < 2; jj++) {
							for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dZdRho[iOmega](ii, jj), 0); it; ++it)
							{
								int iCol = it.col();
								int invertedID = calcElementsVector[it.col()]->invertedRhoElementsID; //element->dZdRhoはnumOfCalcID分微分値を保存している。
																										//一方dDataMisfitDRhoはnumOfInvertedRhoID分の微分値を保存する
								std::complex<double> dZtmp = Zcalc.coeff(ii, jj) - element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj);;
								std::complex<double> dZdRhotmp = dZdRhoCalc(ii, jj).coeff(0, iCol);
								//cout << dZtmp << " " << dZdRhotmp <<" "<<j<< " " << iOmega <<" "<<ii<<" "<<jj<<" "<<" "<< iCol << endl;
								double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
								double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));


								if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0 ) {
									dZtmp.real(dZtmp.real() / epsReal);
									dZdRhotmp.real(dZdRhotmp.real() / epsReal);
								}
								else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
									dZtmp.real(0.0);
								}
								else {
									//そのまま
								}
								if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0 ) {
									dZtmp.imag(dZtmp.imag() / epsImag);
									dZdRhotmp.imag(dZdRhotmp.imag() / epsImag);
								}
								else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
									dZtmp.imag(0.0);
								}
								else {
									//そのまま
								}
								/*if (invertedRhoIDToElementVector[invertedID]->masterResistivityElement != nullptr) {
									dDataMisfitDRho.coeffRef(invertedRhoIDToElementVector[invertedID]->masterResistivityElement->invertedRhoElementsID)
										+= 2.0*(conj(dZtmp)*dZdRhotmp).real();
								}
								else {*/
								if (invertedID >= 0) {
									dDataMisfitDRho.coeffRef(invertedID) += 2.0 * (conj(dZtmp) * dZdRhotmp).real();
								}
								//}


							}
						}
					}
					//for (int i = 0; i < numOfInvertedResistivityElements; i++) {
					//	if (dDataMisfitDRho.coeff(i) != 0.0 && iOmega==0) {
					//		cout << "ID" << i << "iOmega " << iOmega << " dDataMisfitDRho" << dDataMisfitDRho.coeff(i) << endl;
					//	}
					//}
				}
			}
		}

		//Tipper Tensor, this is zero because tipper does not explicitly depend on Resistivity.
		// Todo::他のテンソル量を逆解析する場合はここに足す
		//dDataMisfitDRho /=  numOfObsData;
	}
	else {
		Element::Element* element = locElement;
		for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
			Eigen::Matrix2cd Zcalc;

			if (isInvertedDistortion) {
				Zcalc = element->distortionMatrix * element->Z[iOmega];
				dZdRhoCalc(0, 0) = element->distortionMatrix.coeff(0, 0) * element->dZdRho[iOmega](0, 0) +
					element->distortionMatrix.coeff(0, 1) * element->dZdRho[iOmega](1, 0);
				dZdRhoCalc(0, 1) = element->distortionMatrix.coeff(0, 0) * element->dZdRho[iOmega](0, 1) +
					element->distortionMatrix.coeff(0, 1) * element->dZdRho[iOmega](1, 1);
				dZdRhoCalc(1, 0) = element->distortionMatrix.coeff(1, 0) * element->dZdRho[iOmega](0, 0) +
					element->distortionMatrix.coeff(1, 1) * element->dZdRho[iOmega](1, 0);
				dZdRhoCalc(1, 1) = element->distortionMatrix.coeff(1, 0) * element->dZdRho[iOmega](0, 1) +
					element->distortionMatrix.coeff(1, 1) * element->dZdRho[iOmega](1, 1);
			}
			else {
				Zcalc = element->Z[iOmega];
				for (int ii = 0; ii < 2; ii++) {
					for (int jj = 0; jj < 2; jj++) {
						dZdRhoCalc(ii, jj) = element->dZdRho[iOmega](ii, jj);
					}
				}
			}
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dZdRho[iOmega](ii, jj), 0); it; ++it)
					{
						int iCol = it.col();
						int invertedID = calcElementsVector[it.col()]->invertedRhoElementsID; //element->dZdRhoはnumOfCalcID分微分値を保存している。
																								//一方dDataMisfitDRhoはnumOfInvertedRhoID分の微分値を保存する
						std::complex<double> dZtmp = Zcalc.coeff(ii, jj) ;
						std::complex<double> dZdRhotmp = dZdRhoCalc(ii, jj).coeff(0, iCol);
								
						if (invertedID >= 0) {
							dDataMisfitDRho.coeffRef(invertedID) += 2.0 * (conj(dZtmp) * dZdRhotmp).real()/ boundary->omega[iOmega] / ConstantValues::mu; //App Resis
						}


					}
				}
			}

		}

	}
}

void Analysis::Analysis::CalcDZDRhoElements(const ub::vector<kv::complex<double>>* rhoVecUb,const ub::vector<kv::complex<double>>* HresultTwoItr,const int iOmega) {
	for (int i = 0; i < numOfObsPointElements; i++) {
		Element::Element* element = obsPointElements[i];
		if (element->isInversionImpedance == true) {
			element->CalcDZDRho(rhoVecUb, HresultTwoItr,&calcElementsVector, numOfCalcElements, iOmega);
		}
	}
	for (int i = 0; i < numOfLocationCalcElements; i++) {
		Element::Element* element = locPointElements[i];
		element->CalcDZDRho(rhoVecUb, HresultTwoItr, &calcElementsVector, numOfCalcElements, iOmega);
	}
}
void Analysis::Analysis::CalcDJDRho(bool convertParamMode) {
	for (int i = 0; i < numOfCalcElements; i++) {
		calcElementsVector[i]->debug = 0.0; //debug
	}


	dJdRho.setZero();
	dJExceptRoughnessDRho.setZero();

	//term of data misfit	
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		dJdRho.coeffRef(i) += dDataMisfitDRho.coeff(i);

	}

	//For MultipleObjFunc
	double objfuncVal = 0;
	double misfit=0.0;
	//lambda term
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		dJdRho.coeffRef(i) -= lambdaDRDRho.coeff(i).real();
	}

	//save dJdRho for sensitivity output
	if (convertParamMode) {
		for (int i = 0; i < outputSensitivityVector.size(); i++) {
			double a = std::log10(outputSensitivityVector[i]->resistivity);
			int ID = outputSensitivityVector[i]->invertedRhoElementsID;
			dJExceptRoughnessDRho.coeffRef(i) = dJdRho.coeffRef(ID) * invertedRhoIDToElementVector[ID]->resistivity;
		}

	}
	else {
		for (int i = 0; i < outputSensitivityVector.size(); i++) {
			int ID = outputSensitivityVector[i]->invertedRhoElementsID;
			dJExceptRoughnessDRho.coeffRef(i) = dJdRho.coeffRef(ID);
		}
	}

	//distribute gradient from data misfit followed by Avdeev and Avdeeva 2009
	if (invSettings->modifyGradient) {
		ModifyGradient();
	}


	//term of roughning matrix
	Eigen::VectorXd roughnessterm{ numOfInvertedResistivityElements };
	Eigen::VectorXd logRhoVec{ numOfInvertedResistivityElements };
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		logRhoVec.coeffRef(i) = log(invertedRhoIDToElementVector[i]->resistivity);
	}

	double weightRougheningPre = weightRoughening;
	if (FFTSensitivityMode && lambdaForFFT>0) {
		weightRoughening = lambdaForFFT;
	}
	if (FFTSensitivityMode && weightRoughening < 0) {
		cout << "Warning::WeightRoughning is not set!! Assume weightRoughening as zero!!!!" << endl;
		weightRoughening = 0.0;
	}
	if (useL1Norm) {
		for (int i = 0; i < rougheningMatrix->outerSize(); ++i) {
			double val = rougheningMatrix->row(i) * logRhoVec;
			if (val >= epsForL1Norm) {
				for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(*rougheningMatrix, i); it; ++it) {
					double dmdRho = 1 / invertedRhoIDToElementVector[it.col()]->resistivity;
					dJdRho.coeffRef(it.col()) += rateL1Norm * weightRoughening * rougheningMatrix->coeff(i, it.col()) * dmdRho;
				}

			}
			else if (val <= -epsForL1Norm) {
				for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(*rougheningMatrix, i); it; ++it) {
					double dmdRho = 1 / invertedRhoIDToElementVector[it.col()]->resistivity;
					dJdRho.coeffRef(it.col()) -= rateL1Norm * weightRoughening * rougheningMatrix->coeff(i, it.col()) * dmdRho;
				}

			}

		}
		Eigen::VectorXd WTWm{ numOfInvertedResistivityElements };
		WTWm = rougheningMatrix->transpose() * (*rougheningMatrix) * logRhoVec;
		//WTWm = rougheningMatrix->transpose()*(*rougheningMatrix)*rhoVec;

		for (int i = 0; i < numOfInvertedResistivityElements; i++) {
			//double dmdRho = 1 / invertedRhoIDToElementVector[i]->resistivity*log10(exp(1.0));
			double dmdRho = 1 / invertedRhoIDToElementVector[i]->resistivity;
			//double dmdRho = 1.0;

			dJdRho.coeffRef(i) += (1 - rateL1Norm) * weightRoughening * 2 * WTWm.coeff(i) * dmdRho;

		}
	}

	else {
		string filename = "roughenessterm_" + std::to_string(weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".vtk";
		
		Eigen::VectorXd WTWm{ numOfInvertedResistivityElements };
		WTWm = rougheningMatrix->transpose() * (*rougheningMatrix) * logRhoVec;
		//WTWm = rougheningMatrix->transpose()*(*rougheningMatrix)*rhoVec;

		for (int i = 0; i < numOfInvertedResistivityElements; i++) {
			//double dmdRho = 1 / invertedRhoIDToElementVector[i]->resistivity*log10(exp(1.0));
			double dmdRho = 1 / invertedRhoIDToElementVector[i]->resistivity;
			//double dmdRho = 1.0;

			dJdRho.coeffRef(i) += weightRoughening * 2 * WTWm.coeff(i) * dmdRho;
			roughnessterm.coeffRef(i)= weightRoughening * 2 * WTWm.coeff(i) * dmdRho;

		}
		/*Eigen::VectorXd outputRoughnessTerm{ outputSensitivityVector.size() };
		for (int i = 0; i < outputSensitivityVector.size(); i++) {
			int ID = outputSensitivityVector[i]->invertedRhoElementsID;
			outputRoughnessTerm.coeffRef(i) = roughnessterm.coeffRef(ID);
		}*/
		//output->VTKFileOputput(&outputSensitivityVector, &outputRoughnessTerm, filename);
	}

	
	//set Slave Elem Terms to the master 
	for (int i = 0; i < numOfCalcElements; i++) {
		if (calcElementsVector[i]->invertedRhoElementsID >= 0 && calcElementsVector[i]->MPCResistivityCoeff.size() != 0) {
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(calcElementsVector[i]->MPCResistivityCoeff, 0); it; ++it) {
				if (calcElementsVector[it.col()]->invertedRhoElementsID >= 0) {
					dJdRho.coeffRef(calcElementsVector[it.col()]->invertedRhoElementsID) += calcElementsVector[i]->MPCResistivityCoeff.coeff(0, it.col())
						* dJdRho.coeff(calcElementsVector[i]->invertedRhoElementsID);

					roughnessterm.coeffRef(calcElementsVector[it.col()]->invertedRhoElementsID) += calcElementsVector[i]->MPCResistivityCoeff.coeff(0, it.col())
						* roughnessterm.coeff(calcElementsVector[i]->invertedRhoElementsID);
				}
			}
			dJdRho.coeffRef(calcElementsVector[i]->invertedRhoElementsID) = 0.0;
			roughnessterm.coeffRef(calcElementsVector[i]->invertedRhoElementsID) = 0.0;
		}
	}

	//for (int i = 0; i < numOfInvertedResistivityElements; i++) {
	//	if (invertedRhoIDToElementVector[i]->masterResistivityElement != nullptr) {


	//		dJdRho.coeffRef(invertedRhoIDToElementVector[i]->masterResistivityElement->invertedRhoElementsID) += dJdRho.coeff(i);
	//		dJdRho.coeffRef(i) = 0.0;
	//	}
	//}

	if (FFTSensitivityMode) {
		for (int i = 0; i < numOfInvertedResistivityElements; i++) {
			invertedRhoIDToElementVector[i]->dDataMisfitDRho = dJdRho.coeff(i) - roughnessterm.coeff(i);
			invertedRhoIDToElementVector[i]->dRoughnessTermDRho = roughnessterm.coeff(i);
		}
	}

	dUdRho_output.resize(numOfInvertedResistivityElements,2);

	//Convert dJ/dRho ->dJ/dParam
	if (convertParamMode) {
		for (int i = 0; i < numOfInvertedResistivityElements; i++) {
			dUdRho_output.coeffRef(i, 0) = dJdRho.coeff(i); //Including constraint term
			dUdRho_output.coeffRef(i, 1) = dJdRho.coeff(i)- roughnessterm.coeff(i);//excluding constraint term
			dJdRho.coeffRef(i) = dJdRho.coeff(i) * dRhoDParam.coeff(i);

		}
	}
	
	if (isInvertedDistortion) {
		//data misfit term
		CalcDDataMisfitDDistortionParam();
		
		/*double pre = CalcDataMisfit();
		double d = 0.001;
		for (int i = 0; i < numOfObsImpedanceElements; i++) {
			obsImpedanceElements[i]->distortionMatrix.coeffRef(1, 1) += d;
			double post = CalcDataMisfit();
			cout << "numerical:" << (post - pre) / d << endl;
			cout << "autodif:" << dDataMisfitDDistortionParam.coeff(4 * i + 3) << endl;
		}*/
		double funcVal = 1.0;
		double misfitVal = 1.0;

		
		for (int i = 0; i < 4 * numOfObsImpedanceElements; i++) {
			dJdRho.coeffRef(numOfInvertedResistivityElements + i) = dDataMisfitDDistortionParam.coeff(i)/ misfitVal;
		}

		//constraint term
		for (int i = 0; i < numOfObsImpedanceElements; i++) {
			for (int j = 0; j < 2; j++) {
				for (int k = 0; k < 2; k++) {
					if ((j == 0 && k == 0) || (j == 1 && k == 1)) {
						dJdRho.coeffRef(numOfInvertedResistivityElements + 4 * i + 2 * j + k) += weightRougheningForDistortion * 2.0 *
							(obsImpedanceElements[i]->distortionMatrix.coeff(j, k) - 1.0)/ funcVal;
					}
					else {
						dJdRho.coeffRef(numOfInvertedResistivityElements + 4 * i + 2 * j + k) += weightRougheningForDistortion * 2.0 *
							(obsImpedanceElements[i]->distortionMatrix.coeff(j, k))/ funcVal;
					}
				}
			}

		}

	}
	if (FFTSensitivityMode) {
		weightRoughening = weightRougheningPre; //for safety, recover the value.
	}
}

void Analysis::Analysis::CalcRougheningMatrix() {
	
	//rougheningMatrix->reserve(Eigen::VectorXi::Constant(27*numOfInvertedResistivityElements, 27));
	//rougheningMatrix->setZero();

	//Laplacian Filter
	//rougheningMatrix->reserve(Eigen::VectorXi::Constant(numOfInvertedResistivityElements, 4*7));
	//for (int iInvElem = 0; iInvElem < numOfInvertedResistivityElements; iInvElem++) {
	//	Element::Element* element = invertedRhoIDToElementVector[iInvElem];

	//	if (element->isAirGroundBoundaryCell == true) {
	//		continue;
	//	}

	//	for (int i = 0; i < 6; i++) {
	//		Eigen::Vector3i pos;
	//		pos[0] = 0;
	//		pos[1] = 0;
	//		pos[2] = 0;
	//		if (i == 0) pos[0] = -1;
	//		else if (i == 1) pos[0] = 1;
	//		else if (i == 2) pos[1] = -1;
	//		else if (i == 3) pos[1] = 1;
	//		else if (i == 4) pos[2] = -1;
	//		else if (i == 5) pos[2] = 1;
	//		int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
	//		Element::Element* neighborElem = element->neighborElements[ipos];
	//		if (neighborElem == nullptr) {
	//			continue;
	//		}
	//		double unit;
	//		if (invSettings->isUseDistanceInModelConstraint == false) {
	//			unit = 1.0;
	//		}
	//		else {
	//			if (element->layer == neighborElem->layer) {
	//				unit = element->roughenMatrixUnit;
	//			}
	//			else {
	//				unit = std::max(element->roughenMatrixUnit, neighborElem->roughenMatrixUnit);
	//			}
	//			//unit = std::max(std::max(modelNormalizationCoeff[0], modelNormalizationCoeff[1]), modelNormalizationCoeff[2]);
	//		}

	//		double dl;
	//		if (neighborElem->layer == element->layer) {
	//			dl = (element->centerCoord - neighborElem->centerCoord).norm();
	//		}
	//		else {
	//			if (i == 0 || i == 1) dl = std::max(element->dx, neighborElem->dx);
	//			else if (i == 2 || i == 3) dl = std::max(element->dy, neighborElem->dy);
	//			else if (i == 4 || i == 5) dl = std::max(element->dz, neighborElem->dz);
	//		}
	//		if (invSettings->isUseDistanceInModelConstraint == false) {
	//			dl = 1.0;
	//		}

	//		for (int j = 0; j < element->diffResistivitySurfaceCoeff[i]->outerSize(); ++j) {
	//			for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(*element->diffResistivitySurfaceCoeff[i], j); it; ++it)
	//			{
	//				int iCol = it.col();
	//				if (calcElementsVector[iCol]->invertedRhoElementsID >= 0 && calcElementsVector[iCol]->isAirGroundBoundaryCell == false) {
	//					double val;
	//					if (calcElementsVector[iCol]->invertedRhoElementsID >= 0 && calcElementsVector[iCol]->invertedRhoElementsID == element->invertedRhoElementsID) {
	//						val = -abs((element->diffResistivitySurfaceCoeff[i]->coeff(0, iCol) * 1.0 / dl * unit).real());
	//					}
	//					else if (calcElementsVector[iCol]->invertedRhoElementsID >= 0) {
	//						val = +abs((element->diffResistivitySurfaceCoeff[i]->coeff(0, iCol) * 1.0 / dl * unit).real());
	//					}
	//					rougheningMatrix->coeffRef(iInvElem, calcElementsVector[iCol]->invertedRhoElementsID) += val;
	//				}
	//				else {
	//					for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it2(*element->diffResistivitySurfaceCoeff[i], j); it2; ++it2)
	//					{
	//						int iCol2 = it2.col();
	//						if (calcElementsVector[iCol2]->invertedRhoElementsID >= 0 && calcElementsVector[iCol2]->isAirGroundBoundaryCell == false) {
	//							rougheningMatrix->coeffRef( iInvElem, calcElementsVector[iCol2]->invertedRhoElementsID) = 0.0; //reset settings
	//						}
	//					}
	//					break;
	//				}
	//			}
	//		}
	//	}
	//}
	////debug

	//std::ofstream f;
	//f.open("debugWeightMatrix.txt", std::ios::trunc);
	//for (int j = 0; j < rougheningMatrix->outerSize(); ++j) {
	//	for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(*rougheningMatrix, j); it; ++it)
	//	{
	//		f << it.row() << " " << it.col() << " " << rougheningMatrix->coeff(j, it.col()) << endl;;
	//	}
	//}
	//f.close();


	//
	

	////Diff Filter
	 rougheningMatrix->reserve(Eigen::VectorXi::Constant(6 * numOfInvertedResistivityElements, 6));
\
	for (int iInvElem = 0; iInvElem < numOfInvertedResistivityElements; iInvElem++) {
		Element::Element* element = invertedRhoIDToElementVector[iInvElem];

		if (element->isAirGroundBoundaryCell == true) {
			continue;
		}
		
		for (int i = 0; i < 6; i++) {
			Eigen::Vector3i pos;
			pos[0] = 0;
			pos[1] = 0;
			pos[2] = 0;
			if (i == 0) pos[0] = -1;
			else if (i == 1) pos[0] = 1;
			else if (i == 2) pos[1] = -1;
			else if (i == 3) pos[1] = 1;
			else if (i == 4) pos[2] = -1;
			else if (i == 5) pos[2] = 1;
			int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
			Element::Element* neighborElem = element->neighborElements[ipos];
			if (neighborElem == nullptr) {
				continue;
			}
			double unit;
			if (invSettings->isUseDistanceInModelConstraint == false) {
				unit = 1.0;
			}
			else {
				if (element->layer == neighborElem->layer) {
					unit = element->roughenMatrixUnit;
				}
				else {
					unit = std::max(element->roughenMatrixUnit, neighborElem->roughenMatrixUnit);
				}
				//unit = std::max(std::max(modelNormalizationCoeff[0], modelNormalizationCoeff[1]), modelNormalizationCoeff[2]);
			}

			double dl;
			if (neighborElem->layer == element->layer) {
				dl= (element->centerCoord - neighborElem->centerCoord).norm();
			}
			else {
				if (i == 0 || i == 1) dl = std::max(element->dx,neighborElem->dx);
				else if (i == 2 || i == 3) dl = std::max(element->dy, neighborElem->dy);
				else if (i == 4 || i == 5) dl = std::max(element->dz, neighborElem->dz);
			}
			if (invSettings->isUseDistanceInModelConstraint == false) {
				dl = 1.0;
			}
			
			for (int j = 0; j < element->diffResistivitySurfaceCoeff[i]->outerSize(); ++j) {
				for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(*element->diffResistivitySurfaceCoeff[i], j); it; ++it)
				{
					int iCol = it.col();
					
					if (calcElementsVector[iCol]->invertedRhoElementsID >= 0 && calcElementsVector[iCol]->isAirGroundBoundaryCell==false) {
						rougheningMatrix->coeffRef(6 * iInvElem + i, calcElementsVector[iCol]->invertedRhoElementsID) =
							(element->diffResistivitySurfaceCoeff[i]->coeff(0, iCol)*1.0 / dl * unit).real();
					}
					else {
						for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it2(*element->diffResistivitySurfaceCoeff[i], j); it2; ++it2)
						{
							int iCol2 = it2.col();
							if (calcElementsVector[iCol2]->invertedRhoElementsID >= 0 && calcElementsVector[iCol2]->isAirGroundBoundaryCell == false) {
								rougheningMatrix->coeffRef(6 * iInvElem + i, calcElementsVector[iCol2]->invertedRhoElementsID) = 0.0; //reset settings
							}
						}
						break;
					}
				}
			}
		}
	}
	//debug
	
	std::ofstream f;
	f.open("debugWeightMatrix.txt", std::ios::trunc);
	for (int j = 0; j < rougheningMatrix->outerSize(); ++j) {
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(*rougheningMatrix, j); it; ++it)
		{
			f <<it.row()<<" "<<it.col()<<" "<< rougheningMatrix->coeff(j, it.col()) << endl;;
		}
	}
	f.close();


	////


	//for (int iInvElem = 0; iInvElem < numOfInvertedResistivityElements; iInvElem++) {
	//	Element::Element* element = invertedRhoIDToElementVector[iInvElem];
	//	unordered_map<string, Element::Element*> alreadyCalcElem;
	//	for (int i = 0; i < 3; i++) {
	//		for (int j = 0; j < 3; j++) {
	//			for (int k = 0; k < 3; k++) {
	//				if (i == 1 && j == 1 && k == 1) {
	//					continue;
	//				}
	//				int ipos = i + 3 * j + 9 * k;
	//				
	//				Element::Element* neighborElem = element->neighborElements[ipos];
	//				if ( neighborElem != nullptr && neighborElem->isParent == false && neighborElem->invertedRhoElementsID >= 0 && alreadyCalcElem.count(neighborElem->ID) == 0){
	//					alreadyCalcElem[neighborElem->ID] = neighborElem;
	//					double dl = (element->centerCoord - neighborElem->centerCoord).norm();
	//					rougheningMatrix->insert(27 * iInvElem + ipos, element->invertedRhoElementsID) = 1.0/dl
	//						*std::pow(modelNormalizationCoeff[0]* modelNormalizationCoeff[0] + modelNormalizationCoeff[1]* modelNormalizationCoeff[1],0.5);
	//					rougheningMatrix->insert(27 * iInvElem + ipos, neighborElem->invertedRhoElementsID) = -1.0 / dl
	//						* std::pow(modelNormalizationCoeff[0] * modelNormalizationCoeff[0] + modelNormalizationCoeff[1] * modelNormalizationCoeff[1], 0.5);
	//			
	//					//rougheningMatrix->insert(27 * iInvElem + ipos, element->invertedRhoElementsID) = 1.0;
	//					//rougheningMatrix->insert(27 * iInvElem + ipos, neighborElem->invertedRhoElementsID) = -1.0;

	//				}
	//				else if (neighborElem != nullptr && neighborElem->isParent == true) {
	//					bool isAllChildInverted = true;
	//					for (int ii = 0; ii < 2; ii++) {
	//						for (int jj = 0; jj < 2; jj++) {
	//								string childID = neighborElem->ID + Functions::GetBinaryValue(ii, jj);
	//								if (elements[childID]->invertedRhoElementsID < 0) {
	//									isAllChildInverted = false;
	//								}
	//						}
	//					}
	//					if (isAllChildInverted) {
	//						double dl = (element->centerCoord - neighborElem->centerCoord).norm();
	//						rougheningMatrix->insert(27 * iInvElem + ipos, element->invertedRhoElementsID) = 1.0 / dl
	//							* std::pow(modelNormalizationCoeff[0] * modelNormalizationCoeff[0] + modelNormalizationCoeff[1] * modelNormalizationCoeff[1], 0.5);
	//						
	//						//rougheningMatrix->insert(27 * iInvElem + ipos, element->invertedRhoElementsID) = 1.0;

	//						for (int ii = 0; ii < 2; ii++) {
	//							for (int jj = 0; jj < 2; jj++) {
	//									string childID = neighborElem->ID + Functions::GetBinaryValue(ii, jj);
	//									Element::Element* childElem = elements[childID];
	//									if ( alreadyCalcElem.count(childElem->ID) == 0) {
	//										alreadyCalcElem[childElem->ID] = childElem;					
	//										rougheningMatrix->insert(27 * iInvElem + ipos, childElem->invertedRhoElementsID) = -1.0 / dl / 4.0
	//											* std::pow(modelNormalizationCoeff[0] * modelNormalizationCoeff[0] + modelNormalizationCoeff[1] * modelNormalizationCoeff[1], 0.5);

	//										//rougheningMatrix->insert(27 * iInvElem + ipos, childElem->invertedRhoElementsID) = -1.0 / 4.0;

	//									}

	//							}
	//						}
	//					}
	//					
	//				}

	//			}
	//		}
	//	}
	//}

	//follow femtic by usui-san
	//
	//rougheningMatrix->setZero();
	//rougheningMatrix->reserve(4 * 6 * numOfInvertedResistivityElements);


	//for (int iInvElem = 0; iInvElem < numOfInvertedResistivityElements; iInvElem++) {
	//	Element::Element* element = invertedRhoIDToElementVector[iInvElem];

	//	int invertedID = element->invertedRhoElementsID;
	//	int calcID = element->calcID;
	//	//vector<double> testVec(6);

	//	for (int i = 0; i < 6; i++) {
	//		for (int j = 0; j < element->diffOperationOfResistivitySurfaceCoeff[i]->outerSize(); ++j) {
	//			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*(element->diffOperationOfResistivitySurfaceCoeff[i]), j); it; ++it)
	//			{
	//				
	//				int iCol = it.col();
	//				int anotherElemCalcID = iCol;
	//				Element::Element* anotherElement = calcElementsVector[anotherElemCalcID];
	//				//if (anotherElement->masterResistivityElement != nullptr) {
	//				//	//rougheningMatrix.coeffRef(invertedID, anotherElement->masterResistivityElement->invertedRhoElementsID) +=
	//				//	//	element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();
	//				//	double val = element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();
	//				//	rougheningMatrix.coeffRef(invertedID, anotherElement->masterResistivityElement->invertedRhoElementsID) +=
	//				//		val / abs(val);
	//				//}
	//				if (anotherElement->invertedRhoElementsID >= 0) {
	//					//rougheningMatrix.coeffRef(invertedID, anotherElement->invertedRhoElementsID) +=
	//					//	element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();

	//					double val=element->diffOperationOfResistivitySurfaceCoeff[i]->coeff(0, iCol).real();
	//					rougheningMatrix->coeffRef(6*invertedID+i, anotherElement->invertedRhoElementsID) += val;
	//					//cout<<i<<" " << calcID <<" "<< invertedID<< " " << iCol<<" "<< anotherElement->invertedRhoElementsID << " " << element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real() << endl;
	//				}
	//				else {
	//					/*rougheningMatrix.coeffRef(invertedID, invertedID) +=
	//						element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();*/
	//					double val= element->diffOperationOfResistivitySurfaceCoeff[i]->coeff(0, iCol).real();
	//					rougheningMatrix->coeffRef(6*invertedID+i, invertedID) += val;
	//					//	//if neighbor elements is not inverted, this constraint calculation should be 0.(once plus value is added above, and minus value is added here, so total is zero)
	//					//}
	//				}



	//			}
	//		}
	//	}

	//}


	

	//(*rougheningMatrix) *= modelNormalizationCoeff[0]; //rougheningMatrix has unit Of "/m", so convert non-dimensional.
	//constraint by ∂m
	//rougheningMatrix.resize(6*numOfInvertedResistivityElements, numOfInvertedResistivityElements);
	//rougheningMatrix.setZero();
	//rougheningMatrix.reserve(100 * numOfInvertedResistivityElements);//100は適当
	//for (int iInvElem = 0; iInvElem < numOfInvertedResistivityElements; iInvElem++) {
	//	Element::Element* element = invertedRhoIDToElementVector[iInvElem];
	//	int invertedID = element->invertedRhoElementsID;
	//	int calcID = element->calcID;
	//	//vector<double> testVec(6);

	//	for (int i = 0; i < 6; i++) {
	//		for (int j = 0; j < element->diffOperationOfResistivitySurfaceCoeff[i].outerSize(); ++j) {

	//			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->diffOperationOfResistivitySurfaceCoeff[i], j); it; ++it)
	//			{
	//				int iCol = it.col();
	//				int anotherElemCalcID = iCol;
	//				Element::Element* anotherElement = calcElementsVector[anotherElemCalcID];
	//				//if (anotherElement->masterResistivityElement != nullptr) {
	//				//	//rougheningMatrix.coeffRef(invertedID, anotherElement->masterResistivityElement->invertedRhoElementsID) +=
	//				//	//	element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();
	//				//	double val = element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();
	//				//	rougheningMatrix.coeffRef(invertedID, anotherElement->masterResistivityElement->invertedRhoElementsID) +=
	//				//		val / abs(val);
	//				//}
	//				if (anotherElement->invertedRhoElementsID >= 0) {
	//					//rougheningMatrix.coeffRef(invertedID, anotherElement->invertedRhoElementsID) +=
	//					//	element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();

	//					double val = element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();
	//					rougheningMatrix.coeffRef(6*invertedID+i, anotherElement->invertedRhoElementsID) += val;
	//					//cout<<i<<" " << calcID <<" "<< invertedID<< " " << iCol<<" "<< anotherElement->invertedRhoElementsID << " " << element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real() << endl;
	//				}
	//				else {
	//					/*rougheningMatrix.coeffRef(invertedID, invertedID) +=
	//						element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();*/
	//					double val = element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();
	//					rougheningMatrix.coeffRef(6*invertedID+i, invertedID) += val;
	//					//	//if neighbor elements is not inverted, this constraint calculation should be 0.(once plus value is added above, and minus value is added here, so total is zero)
	//					//}
	//				}



	//			}
	//		}
	//	}

	//}

	//debug
	//for (int id = 0; id < numOfInvertedResistivityElements; id++) {
	//	cout << invertedRhoIDToElementVector[id]->boundary << endl;
	//	for (int ii = 0; ii < 3; ii++) {
	//		for (int jj = 0; jj < 3; jj++) {
	//			for (int kk = 0; kk < 3; kk++) {
	//				if (invertedRhoIDToElementVector[id]->neighborElements[ii + 3 * jj + 9*kk] != NULL) {
	//					cout << invertedRhoIDToElementVector[id]->neighborElements[ii + 3 * jj + 9*kk]->invertedRhoElementsID << " " << ii << " " << jj << " " << kk << endl;
	//				}
	//			}
	//		}
	//	}
	//	for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(rougheningMatrix, id); it; ++it)
	//	{
	//		int iCol = it.col();
	//		int iRow = it.row();
	//		cout << iRow << " " << iCol << " " << rougheningMatrix.coeff(iRow, iCol) << endl;
	//	}
	//}
	

	//rougheningMatrix *= modelNormalizationCoeff; //rougheningMatrix has unit Of "/m", so convert non-dimensional.

	//for (int i = 0; i < 6; i++) {
	//	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
	//		Element::Element* element = invertedRhoIDToElementVector[i];
	//		int invertedID = element->invertedRhoElementsID;
	//		int calcID = element->calcID;
	//		//vector<double> testVec(6);
	//		for (int i = 0; i < 6; i++) {
	//			//double test = 0;
	//			for (int j = 0; j < element->diffOperationOfResistivitySurfaceCoeff[i]->outerSize(); ++j) {

	//				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*(element->diffOperationOfResistivitySurfaceCoeff[i]), j); it; ++it)
	//				{
	//					int iCol = it.col();
	//					int anotherElemCalcID = iCol;
	//					Element::Element* anotherElement = calcElementsVector[anotherElemCalcID];
	//					if (anotherElement->invertedRhoElementsID >= 0) {
	//						rougheningMatrix->coeffRef(6 * invertedID + i, anotherElement->invertedRhoElementsID) +=
	//							element->diffOperationOfResistivitySurfaceCoeff[i]->coeff(0, iCol).real();
	//						//cout<<i<<" " << calcID <<" "<< invertedID<< " " << iCol<<" "<< anotherElement->invertedRhoElementsID << " " << element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real() << endl;
	//					}
	//					else {
	//						rougheningMatrix->coeffRef(6 * invertedID + i, invertedID) +=
	//							element->diffOperationOfResistivitySurfaceCoeff[i]->coeff(0, iCol).real();
	//						//if neighbor elements is not inverted, this constraint calculation should be 0.(once plus value is added above, and minus value is added here, so total is zero)
	//					}
	//				}

	//			}

	//			//cout << element->resistivitySurfaceCoeff[i] << endl;
	//			//testVec[i] = test;
	//		}

	//	}
	//}
	//rougheningMatrix *=  modelNormalizationCoeff; //rougheningMatrix has unit Of "/m", so convert non-dimensional.
	rougheningMatrix->makeCompressed();
	rougheningMatrix->data().squeeze();
}

inline double Analysis::Analysis::Optimize(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* opt_data)
{
\
	bool isChangeResis = false;
	Eigen::VectorXd ParamResis(numOfInvertedResistivityElements);
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		ParamResis.coeffRef(i) = vals_inp.coeff(i);
	}
	if (isInvertedDistortion) {
		for (int i = 0; i < numOfObsImpedanceElements; i++) {
			obsImpedanceElements[i]->distortionMatrix.coeffRef(0, 0) = vals_inp.coeff(numOfInvertedResistivityElements + 4 * i);
			obsImpedanceElements[i]->distortionMatrix.coeffRef(0, 1) = vals_inp.coeff(numOfInvertedResistivityElements + 4 * i + 1);
			obsImpedanceElements[i]->distortionMatrix.coeffRef(1, 0) = vals_inp.coeff(numOfInvertedResistivityElements + 4 * i + 2);
			obsImpedanceElements[i]->distortionMatrix.coeffRef(1, 1) = vals_inp.coeff(numOfInvertedResistivityElements + 4 * i + 3);
			
		}
	}
	isChangeResis = CalcRhoFromParamAndDRhoDParam(ParamResis);




	SetSameResistivityToBoundaryCell();
	cout << "isChangeResis" << isChangeResis << endl;
	//if (isChangeResis == true || initObjVal==0) {
	CalcSurfaceResistivityElements(); //Update Resistivity
	time_t start_t = time(NULL);
	cout << "Update SumNCrossRhoRotHdS.." << endl;
	CalcSumNCrossRhoRotHdSElements(); //Update coeffs of Matrix
	time_t end_t = time(NULL);
	std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
	cout << "End Update SumNCrossRhoRotHdS.." << endl;
	bool isNeededGradient = false;
	if (grad_out) {
		dDataMisfitDRho.setZero();
		if (lambdaDRDRho.size() == 0) {
			lambdaDRDRho.resize(numOfInvertedResistivityElements);
		}

		lambdaDRDRho.setZero();

		isNeededGradient = true;
	}


	CalcForward(isNeededGradient);

	//}

	invSettings->ReadManualSettingData(&settings);

	double obj_val;
	obj_val = 0.0;
	double factor = 1.0;
	double RMS;

	obj_val = CalcDataMisfit();
	RMS = std::pow(obj_val / numOfObsData, 0.5);
	
	
	std::cout << "DataMisfit:" << obj_val << std::endl;


	RMScur = RMS;

	double roughningMatrixPenaltyTerm = CalcRoughningMatrixPenalty();


	obj_val += weightRoughening * roughningMatrixPenaltyTerm; 
	if (isInvertedDistortion) {
		CalcConstraintDistortionTerm();

		obj_val += weightRougheningForDistortion * constraintDistortionTerm;
		
	}
	bool isFirstLoop = false;
	if (optMethod == "GD") {
		if (initObjVal == 0 && inheritPreviousObjVal == false) {
			initObjVal = obj_val;
			obj_valPre = 1.0;
			isFirstLoop = true;
		}

		else if (inheritPreviousObjVal == true && isFirstLoopInheritPreviousObjVal) {
			isFirstLoop = true;
			isFirstLoopInheritPreviousObjVal = false;
		}
	}
	else {
		if (initObjVal == 0) {
			initObjVal = obj_val;
			obj_valPre = 1.0;
			isFirstLoop = true;
		}
	}

	if (grad_out) {

		/*vector<int>debugElemID;
		debugElemID.push_back(invertedRhoIDToElementVector[10]->invertedRhoElementsID);
		debugElemID.push_back(invertedRhoIDToElementVector[100]->invertedRhoElementsID);
		debugElemID.push_back(obsPointElements[10]->invertedRhoElementsID);
		debugElemID.push_back(obsPointElements[20]->invertedRhoElementsID);*/
		if (isChangeResis == true || isFirstLoop==true) {
			CalcDDataMisfitDRho();
			CalcDJDRho();
			//CalcJacobian();
		}
		if (optMethod == "GD") {
			(*grad_out) = dJdRho / initObjVal;
		}
		else {
			(*grad_out) = dJdRho;
		}
	}
	obj_valNotNormalized = obj_val;
	if (settings.numOfIteration % invSettings->outputInterval <= 0) {
		if (numOfSameModelWeightCalc == 1) {
			std::string filename = "Rho_" + std::to_string(weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".vtk";
			output->RhoOutput(&elements, filename);
			filename = "Rho_" + std::to_string(weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".txt";
			output->TxtOutputResistivity(&elements, filename);
			filename = "Sensitivity_" + std::to_string(weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".vtk";
			output->VTKFileOputput(&outputSensitivityVector,&dJExceptRoughnessDRho, filename);
			filename = "ObsCalcImpedance_" + std::to_string(weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".txt";
			output->OutputObsCalcImpedance(boundary->omega, &obsPointElements, filename);
			filename = "ObsCalcTipper_" + std::to_string(weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".txt";
			output->OutputObsCalcTipper(boundary->omega, &obsPointElements, filename);
			filename = "gradObjFunc_" + std::to_string(weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".txt";
			output->OutputGradObjFunc(obj_val,&invertedRhoIDToElementVector, &dUdRho_output, filename);

			if (isInvertedDistortion) {
				filename = "Distortion_" + std::to_string(weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".txt";
				output->OutputDistortionMatrix(&obsImpedanceElements, constraintDistortionTerm, filename);
				filename = "DistortionForRestart_" + std::to_string(weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".txt";
				output->OutputDistortionMatrixForRestart(&obsImpedanceElements, filename);
			}
		}
		else {
			std::string filename = "Rho_" + std::to_string(weightRoughening) + "_" + std::to_string(numOfSameModelWeightCalc) + "_" + std::to_string(settings.numOfIteration) + ".vtk";
			output->RhoOutput(&elements, filename);
			filename = "Rho_" + std::to_string(weightRoughening) + "_" + std::to_string(numOfSameModelWeightCalc) + "_" + std::to_string(settings.numOfIteration) + ".txt";
			output->TxtOutputResistivity(&elements, filename);
			filename = "ObsCalcImpedance_" + std::to_string(weightRoughening) + "_" + std::to_string(numOfSameModelWeightCalc) + "_" + std::to_string(settings.numOfIteration) + ".txt";
			output->OutputObsCalcImpedance(boundary->omega, &obsPointElements, filename);
			filename = "ObsCalcTipper_" + std::to_string(weightRoughening) + "_" + std::to_string(numOfSameModelWeightCalc) + "_" + std::to_string(settings.numOfIteration) + ".txt";
			output->OutputObsCalcTipper(boundary->omega, &obsPointElements, filename);
			filename = "Sensitivity_" + std::to_string(weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".vtk";
			output->VTKFileOputput(&outputSensitivityVector, &dJExceptRoughnessDRho, filename);
			filename = "gradObjFunc_" + std::to_string(weightRoughening) + "_" + std::to_string(numOfSameModelWeightCalc) +"_" + std::to_string(settings.numOfIteration) + ".txt";
			output->OutputGradObjFunc(obj_val,&invertedRhoIDToElementVector, &dUdRho_output, filename);
			if (isInvertedDistortion) {
				filename = "Distortion_" + std::to_string(weightRoughening) + "_" + std::to_string(numOfSameModelWeightCalc) + "_" + std::to_string(settings.numOfIteration) + ".txt";
				output->OutputDistortionMatrix(&obsImpedanceElements, constraintDistortionTerm, filename);
				filename = "DistortionForRestart_" + std::to_string(weightRoughening) + "_" + std::to_string(numOfSameModelWeightCalc) + "_" + std::to_string(settings.numOfIteration) + ".txt";
				output->OutputDistortionMatrixForRestart(&obsImpedanceElements, filename);
			}
		}
	}



	if (RMS < thresholdRMS) {
		//cout << "Optimization has finished because RMS is below threshold." << endl;
		//settings.isFinishOptimize = true;
		isBelowRMSThreshold = true;
		//obj_val = 0.0;
		//if (grad_out) {
		//	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		//		(*grad_out).coeffRef(i) = 0.0;
		//	}
		//}
	}
	else {
		isBelowRMSThreshold = false;
	}

	//else if (RMS > RMSpre) { //only for gradient discent
	//	obj_val = 0.0;
	//	if (grad_out) {
	//		for (int i = 0; i < numOfInvertedResistivityElements; i++) {
	//			(*grad_out).coeffRef(i) = 0.0;
	//			RMSpre = 1e30;
	//		}
	//	}
	//}
	//else {
	//	RMSpre = RMS;
	//}
	//if (optMethod == "GD") {
	//	if (settings.iteration > 0 &&
	//		std::abs(obj_val - obj_valPre)/obj_valPre < objFuncChangeThresholdForNextmodelConstraint) {
	//		settings.isFinishOptimize = true;
	//		obj_valPre = obj_val;
	//		obj_val = 0.0;
	//		RMSpre = 1e30;
	//	}
	//	else {
	//		obj_valPre = obj_val;
	//		RMSpre = RMS;
	//		RMSpre = RMS;
	//	}
	//}
	//else {
	
	//}
	//
	if (optMethod == "GD") {
		//double tmp = obj_val / obj_valPre; //nakayama eiji san method
		//obj_valPre = obj_val;
		//obj_val = tmp;
		obj_val = obj_val / initObjVal;
		
	}
	//else {
	//	obj_val = obj_val;

	//}
	
	//if (optMethod == "GD" && obj_valPre < obj_val) {
	//	//if (RMScur > thresholdRMS) {
	//		settings.gd_settings.isRestartAdam = true;
	//	//}
	//}
	//if (optMethod == "GD") {
	//	if (isFirstLoop == false && isChangeResis == true && obj_val!=0.0 && std::abs(obj_val - obj_valPre)/ obj_val < invSettings->objFuncChangeThresholdForNextmodelConstraint) {
	//		cout << "This model constraint Optimization has finished because the change of Objective Function is below threshold." << endl;
	//		cout << "obj_val:" << obj_val << endl;
	//		cout << "pre obj_val:" << obj_valPre << endl;
	//		settings.isFinishOptimize = true;
	//		obj_val = 0.0;
	//		if (grad_out) {
	//			for (int i = 0; i < numOfInvertedResistivityElements; i++) {
	//				(*grad_out).coeffRef(i) = 0.0;
	//			}
	//		}
	//	}
	//}

	std::cout << "RMS:" << RMS << endl;
	//std::cout << "DataMisfit:" << obj_val << std::endl;
	std::cout << "weightRoughening:" << weightRoughening << " PemaltyTerm:" << roughningMatrixPenaltyTerm << std::endl;
	std::cout << "Objective Function Value:" << obj_val << std::endl;
	std::cout << "Objective Function Change:" << obj_val - obj_valPre << std::endl;
	if (optMethod == "GD") {
		std::cout << "GD Step Size:" << settings.gd_settings.par_step_size << std::endl;
	}
	std::cout << "Total Calculation Time:" << time(NULL) - startCalc_t << " Seconds." << endl;

	infofile << "\nnumOfIteration:" << settings.numOfIteration<< endl;
	infofile << "weightRoughening:" << weightRoughening << endl;
	infofile << "RMS:" << RMS << endl;
	//std::cout << "DataMisfit:" << obj_val << std::endl;
	infofile << " PemaltyTerm:" << roughningMatrixPenaltyTerm << std::endl;
	infofile << "Objective Function Value:" << obj_val << std::endl;
	infofile << "Objective Function Change:" << obj_val - obj_valPre << std::endl;
	if (optMethod == "GD") {
		infofile << "GD Step Size:" << settings.gd_settings.par_step_size << std::endl;
	}
	infofile << "Total Calculation Time:" << time(NULL) - startCalc_t << " Seconds." << endl;
	if (!std::isfinite(obj_val)) {
		obj_val = 1e30;
		if (grad_out) {
			grad_out->setOnes();
			(*grad_out) *= 1e30;   // ←こうする
		}
		return obj_val;
	}
	else {
		obj_valPre = obj_val;
	}

	settings.objFuncVal = obj_val;
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		settings.resisVec.coeffRef(i) = log(invertedRhoIDToElementVector[i]->resistivity);
	}
	if(settings.isFinishOptimize) {
		obj_val = 0.0;
		grad_out->setZero();
	}

	isFirstLambdaAndLoop = false;

	return obj_val;
}






inline Eigen::Vector2d Analysis::Analysis::OptimizeUsingJacobian(const Eigen::VectorXd& vals_inp, Eigen::MatrixXd* jac_out)
{
	//bool isChangeResis = false;
	//isChangeResis = CalcRhoFromParamAndDRhoDParam(vals_inp);

	//
	//SetSameResistivityToBoundaryCell();
	////isChangeResis = true; //test
	//cout << "isChangeResis" << isChangeResis << endl;
	//if (isChangeResis == true || initObjVal == 0) {
	//	CalcSurfaceResistivityElements(); //Update Resistivity
	//	CalcSumNCrossRhoRotHdSElements(); //Update coeffs of Matrix

	//	bool isNeededGradient = false;
	//	bool isNeededJacobi = true;
	//	if (jac_out) {
	//		jacobian->setZero();
	//		isNeededGradient = true;
	//	}

	//	CalcForward(isNeededGradient, isNeededJacobi);
	//}
	//double obj_val;
	//obj_val = 0.0;
	// 
	//obj_val += CalcDataMisfit();
	//std::cout << "DataMisfit:" << obj_val << std::endl;

	//double RMS = std::pow(obj_val / numOfObsData, 0.5);
	//

	//double roughningMatrixPenaltyTerm = CalcRoughningMatrixPenalty();


	//obj_val +=  weightRoughening * roughningMatrixPenaltyTerm;

	//bool isFirstLoop = false;
	//if (initObjVal == 0) {
	//	initObjVal = obj_val;
	//	obj_valPre = obj_val;
	//	isFirstLoop = true;
	//}

	//if (jac_out) {
	//	*jac_out = (1.0/ initObjVal)*(*jacobian);

	//}
	//if (numOfSameModelWeightCalc == 1) {
	//	std::string filename = "Rho_" + std::to_string( weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".vtk";
	//	output->RhoOutput(&elements, filename);
	//	filename = "Rho_" + std::to_string( weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".txt";
	//	output->TxtOutputResistivity(&elements, filename);
	//}
	//else {
	//	std::string filename = "Rho_" + std::to_string(weightRoughening) + "_" + std::to_string(numOfSameModelWeightCalc) + "_" + std::to_string(settings.numOfIteration) + ".vtk";
	//	output->RhoOutput(&elements, filename);
	//	filename = "Rho_" + std::to_string(weightRoughening) + "_" + std::to_string(numOfSameModelWeightCalc) + "_" + std::to_string(settings.numOfIteration) + ".txt";
	//	output->TxtOutputResistivity(&elements, filename);
	//}




	//obj_val = obj_val / initObjVal;
	//Eigen::Vector2d returnval;
	//returnval.coeffRef(0) = obj_val;
	//returnval.coeffRef(1) = RMS;
	//return returnval;
}
void Analysis::Analysis::RunOptimize() {
	//mkl_set_num_threads(omp_get_max_threads());

	optMethod = invSettings->optMethod;
	isDirectSolver = invSettings->isDirectSolver;

	

	startCalc_t = time(NULL);
	std::cout << ("Initialize Data") << std::endl;
	Initialize();
	std::cout << ("Initialization End") << std::endl;

	//DEBUG
	//CalcForward(true);
	//CalcDDataMisfitDRho();
	//CalcDJDRho();


	output->RhoOutput(&elements);
	output->VTKObsPointsFileOutput(&obsPointElements);
	//output->VTKFileOputput(0.0, &elements, "ObsPoints");

	
	//CountIndependentInvertedResisElements();

	numOfParameters = numOfInvertedResistivityElements;
	Eigen::VectorXd paramVec;
	if (isInvertedDistortion) {
		numOfParameters += numOfObsImpedanceElements * 4;
		paramVec.resize(numOfParameters);
		Eigen::VectorXd tmp= CalcParamFromRho();
		for (int i = 0; i < numOfInvertedResistivityElements; i++) {
			paramVec.coeffRef(i) = tmp.coeff(i);
		}
		for (int i = 0; i < numOfObsImpedanceElements; i++) { 
			paramVec.coeffRef(numOfInvertedResistivityElements + 4 * i) = obsImpedanceElements[i]->distortionMatrix.coeff(0, 0);
			paramVec.coeffRef(numOfInvertedResistivityElements + 4 * i + 1) = obsImpedanceElements[i]->distortionMatrix.coeff(0, 1);
			paramVec.coeffRef(numOfInvertedResistivityElements + 4 * i + 2) = obsImpedanceElements[i]->distortionMatrix.coeff(1, 0);
			paramVec.coeffRef(numOfInvertedResistivityElements + 4 * i + 3) = obsImpedanceElements[i]->distortionMatrix.coeff(1, 1);
		}
	}
	else {
		paramVec = CalcParamFromRho();
	}
	for (int i = 0; i < numOfParameters; i++) {
		if (!std::isfinite(paramVec.coeff(i))) {
			cout << "ERROR::Some Initial Resistivity is Not Finite or Out Of Range." << std::endl;
				exit(1);
		}
	}

	if (FFTSensitivityMode) {
		std::cout << "FFT Sensitivity Mode is Running!!!!!!!!" << endl;
		RunFFTSensitivityAnalysis();
		exit(0);
	}





	std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, void*)> optFunc = std::bind(
		&Analysis::Optimize,
		this,
		std::placeholders::_1,
		std::placeholders::_2,
		std::placeholders::_3);
	void* opt_data;
	
	settings.print_level = 1;
	settings.gd_settings.method =6; ///adamax
	settings.gd_settings.ada_max = true;
	settings.gd_settings.par_step_size = invSettings->par_step_size;
	settings.gd_settings.loosenFactor = invSettings->loosenFactor;
	settings.gd_settings.decreaseFactor = invSettings->decreaseFactor;
	
	settings.gd_settings.numWarmUp = invSettings->numWarmUp;
	settings.gd_settings.minIterations = invSettings->minIterations;
	settings.gd_settings.averageIterations = invSettings->averageIterations;
	settings.gd_settings.numTrunc = invSettings->numTrunc;

	settings.iter_max = invSettings->maxIterationPerModelConstraint;
	settings.lbfgs_settings.step =1;
	settings.lbfgs_settings.wolfe_cons_1 = 1e-3;
	settings.lbfgs_settings.wolfe_cons_2 = 0.99;
	settings.lbfgs_settings.par_M = 10;
	settings.lbfgs_settings.restart_M = 5;
	settings.lbfgs_settings.loosenFactor = invSettings->loosenFactor;
	settings.lbfgs_settings.decreaseFactor = invSettings->decreaseFactor;
	settings.lbfgs_settings.minStep = invSettings->minStep;
	settings.lbfgs_settings.maxIterationLineSearch = invSettings->maxIterationLineSearch;

	settings.cg_settings.use_rel_sol_change_crit = true;
	settings.rel_objfn_change_tol = invSettings->objFuncChangeThresholdForNextmodelConstraint;
	settings.grad_err_tol = invSettings->grad_err_tol;
	settings.rel_sol_change_tol = invSettings->rel_sol_change_tol;

	settings.resultVector = &resultVector;
	settings.resultAdjointVector = &resultAdjointVector;
	//settings.resultVector_pre = &resultVector_pre;
	//settings.resultAdjointVector_pre = &resultAdjointVector_pre;

	settings.useImpedanceDataVec = &useImpedanceDataArray;
	settings.useTipperDataVec = &useTipperDataArray;

	settings.gd_settings.minibatches = invSettings->minibatches;

	thresholdRMS = invSettings->thresholdRMS;
	opt_data = NULL;

	//Eigen::VectorXd lowerBounds{ numOfInvertedResistivityElements };
	//Eigen::VectorXd upperBounds{ numOfInvertedResistivityElements };
	//for (int i = 0; i < numOfInvertedResistivityElements; i++) {
	//	lowerBounds[i] = -limitOfparamLogNormalization;
	//	upperBounds[i] = +limitOfparamLogNormalization;
	//}
	//settings.lower_bounds = lowerBounds;
	//settings.upper_bounds = upperBounds;

	double adoptModelConstraint = 0.0;
	Eigen::VectorXd rhoVec{ numOfInvertedResistivityElements };
	isBelowRMSThreshold = false;

	



	cout << "Optimization Method is :" << optMethod << endl;
	if (optMethod == "GD") {
		cout << "Number Of Mini-Batches: " << settings.gd_settings.minibatches << endl;
	}
	if (isDirectSolver) {
		cout << "Direct Solver PARDISO is Used." << endl;
	}
	else {
		cout << "Iterative Solver BiCGSafe is Used." << endl;
	}
	initObjVal = 0.0;

	infofile.open("optimizeInfo.txt", std::ios::out);
	
	vector<Eigen::VectorXd> resistivitiesEachLambda(invSettings->lambdaVector.size());
	vector<bool> isBelowRMS(invSettings->lambdaVector.size());
	int adoptedLambdaNumber = -1;

	settings.resisVec.resize(numOfInvertedResistivityElements);
	settings.resisVec_p.resize(numOfInvertedResistivityElements);
	settings.resisVec.setZero();
	settings.resisVec_p.setZero();

	if (invSettings->lambdaVector.size() == 0) {
		//for (int i = 0; i < numOfCalcModelConstraint;i++) {
		

		for (int i = numOfCalcModelConstraint - 1; i >= 0; i--) {
			settings.numOfIteration = -1;
			preParams.resize(numOfInvertedResistivityElements);
			preParams.setZero();
			//weightRoughening = modelConstraintMax - (modelConstraintMax - modelConstraintMin)*i / (numOfCalcModelConstraint - 1);
			weightRoughening = modelConstraintMin * std::pow(10.0, std::log10(modelConstraintMax / modelConstraintMin)*double(i) / double(numOfCalcModelConstraint - 1));
			weightRougheningForDistortion = weightRoughening;
			bool success = false;
			if (optMethod=="LBFGS" && RMScur < invSettings->RMSSwitchingToGD) {
				optMethod = "GD";
				cout << "RMS IS BELOW RMSSwitchingToGD. SWITCH TO GD METHOD!!!!!" << endl;
			}
			if (optMethod == "GD" && RMScur < invSettings->RMSSwitchingToLBFGS) {
				optMethod = "LBFGS";
				cout << "RMS IS BELOW RMSSwitchingToLBFGS. SWITCH TO L-BFGS METHOD!!!!!" << endl;
			}
			if (optMethod == "GD") {
				initObjVal = 0.0;
				settings.gd_settings.isRestartAdam = false;
				success = optim::gd(paramVec, optFunc, opt_data, settings);
			}
			else if (optMethod == "LBFGS") {
				initObjVal = 0.0;
				//settings.lbfgs_settings.par_M = settings.iter_max;
				success = optim::lbfgs(paramVec, optFunc, opt_data, settings);

			}
			else {
				void* dummy = nullptr;

				auto value_grad = [&](const Eigen::VectorXd& m, Eigen::VectorXd& g) -> double {
					return Optimize(m, &g, dummy);
					};

				auto value_only = [&](const Eigen::VectorXd& m) -> double {
					return Optimize(m, nullptr, dummy);
					};

				Eigen::VectorXd m_init= paramVec; // initial model
				Eigen::VectorXd m0= paramVec;     // refference model
				double beta = weightRoughening; //This is temporal.

				fista::Options opt;
				opt.max_iters = settings.iter_max;
				opt.max_backtracks = 10;
				opt.L0 = 1.0;
				opt.use_monotone = true;
				opt.use_fista = true;
				opt.restart = true;

				auto res = fista::solve_l1_shifted_fista_vg(
					m_init, m0, beta,
					value_grad,
					value_only,
					opt
				);
			}


			if (isBelowRMSThreshold == true) {
				cout << "Optimization has finished successfully." << endl;
				break;
			}
			//initObjVal = 0.0;

			//if (isBelowRMSThreshold == true) {
			//	for (int j = 0; j < numOfInvertedResistivityElements; j++) {
			//		rhoVec.coeffRef(j) = invertedRhoIDToElementVector[j]->resistivity;
			//	}
			//	adoptModelConstraint = weightRoughening;
			//}
			//else {
			//	if (i == 0) {
			//		adoptModelConstraint = weightRoughening;
			//	}
			//	else {
			//		for (int j = 0; j < numOfInvertedResistivityElements; j++) {
			//			invertedRhoIDToElementVector[j]->resistivity = rhoVec.coeff(j);
			//		}
			//	}
			//	break;
			//}


		}
	}
	else {
		bool isAscendingOrder = false;
		bool isDescendingOrder = false;
		for (int i = 1; i < invSettings->lambdaVector.size(); i++) {
			if (invSettings->lambdaVector[i - 1] > invSettings->lambdaVector[i]) {
				isDescendingOrder = true;
			}
			else if (invSettings->lambdaVector[i - 1] < invSettings->lambdaVector[i]) {
				isAscendingOrder = true;
			}
		}
		if (isAscendingOrder && isDescendingOrder) {
			cout << "Warning::Mixed Descending and Ascending Order in Model Weight Order is detected.\nContinue as  Ascending Order. " << endl;
		}
		else if (isAscendingOrder) {
			cout << "Model Weight Order is Ascending." << endl;
		}
		else if (isAscendingOrder==false){
			cout << "Model Weight Order is Descending." << endl;
		}
		
		for (int i =  0; i < invSettings->lambdaVector.size(); i++) {
		//for (int i = invSettings->lambdaVector.size() - 1; i >= 0 ; i--) {
			settings.rel_objfn_change_tol = invSettings->objFuncChangeThresholdVector[i];
			settings.rel_resis_change_tol = invSettings->thresholdRelativeResistivityChangeVector[i];
			//invSettings->objFuncChangeThresholdForNextmodelConstraint = invSettings->objFuncChangeThresholdVector[i];
			//invSettings->thresholdResistivityChange = invSettings->thresholdResistivityChangeVector[i];
			preParams.resize(numOfInvertedResistivityElements);
			preParams.setZero();
			isBelowRMS[i] = false;
			weightRoughening = invSettings->lambdaVector[i];
			weightRougheningForDistortion = weightRoughening;
			settings.grad_err_tol = invSettings->grad_err_tolVector[i];
			settings.iter_max = invSettings->maxIterationVector[i];
			bool success = false;
			if (optMethod == "LBFGS" && RMScur < invSettings->RMSSwitchingToGD) {
				optMethod = "GD";
				cout << "RMS IS BELOW RMSSwitchingToGD. SWITCH TO GD METHOD!!!!!" << endl;
			}
			if (optMethod == "GD" && RMScur < invSettings->RMSSwitchingToLBFGS) {
				optMethod = "LBFGS";
				cout << "RMS IS BELOW RMSSwitchingToLBFGS. SWITCH TO L-BFGS METHOD!!!!!" << endl;
			}
			if (optMethod == "GD") {
				
				if (invSettings->inheritPreviousSettingAdam && i >= 1 && weightRoughening == invSettings->lambdaVector[i - 1]) {
					settings.gd_settings.inheritPreviousSettingAdam = true;
					settings.gd_settings.par_step_size = invSettings->par_step_sizeVector[i];
					inheritPreviousObjVal = true;
					isFirstLoopInheritPreviousObjVal = true;
					numOfSameModelWeightCalc++;
				}
				else {
					inheritPreviousObjVal = false;
					initObjVal = 0.0;
					settings.gd_settings.par_step_size = invSettings->par_step_sizeVector[i];
					settings.gd_settings.inheritPreviousSettingAdam = false;
					numOfSameModelWeightCalc = 1;
				}
				success = optim::gd(paramVec, optFunc, opt_data, settings);
			}
			else if (optMethod == "LBFGS") {
				initObjVal = 0.0;
				settings.lbfgs_settings.par_M = settings.iter_max;
				success = optim::lbfgs(paramVec, optFunc, opt_data, settings);
			}
			else if (optMethod == "GD-LBFGS") {
				settings.grad_err_tol = settings.grad_err_tol * 10;
				success = optim::lbfgs(paramVec, optFunc, opt_data, settings);
				settings.grad_err_tol = settings.grad_err_tol / 10;
				success = optim::gd(paramVec, optFunc, opt_data, settings);

			}
			else {
				cout << "FISTA For L1 Norm Start!!!!!!" << endl;

				void* dummy = nullptr;

				auto value_grad = [&](const Eigen::VectorXd& m, Eigen::VectorXd& g) -> double {
					return Optimize(m, &g, dummy);
					};

				auto value_only = [&](const Eigen::VectorXd& m) -> double {
					return Optimize(m, nullptr, dummy);
					};

				Eigen::VectorXd m_init = paramVec; // initial model
				Eigen::VectorXd m0 = paramVec;     // refference model
				double beta = weightRoughening; //This is temporal.

				fista::Options opt;
				opt.max_iters = settings.iter_max;
				opt.max_backtracks = 10;
				opt.L0 = 1.0;
				opt.use_monotone = true;
				opt.use_fista = true;
				opt.restart = true;

				auto res = fista::solve_l1_shifted_fista_vg(
					m_init, m0, beta,
					value_grad,
					value_only,
					opt
				);
			}

			isBelowRMS[i] = isBelowRMSThreshold;
			resistivitiesEachLambda[i].resize(numOfCalcElements);
			for (int j = 0; j < numOfCalcElements; j++) {
				resistivitiesEachLambda[i][j] = calcElementsVector[j]->resistivity;
			}
			if (isAscendingOrder == true) {
				if (i >0 && isBelowRMS[i] == false) {
					if (isBelowRMS[i - 1] == true) {
						adoptedLambdaNumber = i - 1;
						adoptModelConstraint = invSettings->lambdaVector[i - 1];
					}
					else {
						cout << "Optimization is NOT CONVERGED." << endl;
						adoptedLambdaNumber = 0;
						adoptModelConstraint = invSettings->lambdaVector[0];
					}
				}
				else if (i == 0 && isBelowRMS[i] == false) {
					cout << "Optimization is NOT CONVERGED." << endl;
					adoptedLambdaNumber = 0;
					adoptModelConstraint = invSettings->lambdaVector[0];
				}
				else if (i == invSettings->lambdaVector.size() - 1 && isBelowRMS[i] == true) {
					cout << "Optimization is CONVERGED at the Largest model Constraint." << endl;
					adoptedLambdaNumber = invSettings->lambdaVector.size() - 1;
					adoptModelConstraint = invSettings->lambdaVector[invSettings->lambdaVector.size() - 1];
				}
			}
			else if (isAscendingOrder == false) {
				if (isBelowRMSThreshold == true) {
					cout << "Optimization has finished successfully." << endl;
					adoptModelConstraint = invSettings->lambdaVector[i];
					adoptedLambdaNumber = i;
					break;
				}
				else if (i == invSettings->lambdaVector.size() - 1) {
					adoptModelConstraint = invSettings->lambdaVector[i];
					adoptedLambdaNumber = i;
				}
			}

		}
	}
	std::cout << "Adopted Model Constraint:" << adoptModelConstraint << endl;
	for (int j = 0; j < numOfCalcElements; j++) {
		calcElementsVector[j]->resistivity = resistivitiesEachLambda[adoptedLambdaNumber][j];
	}

	CalcForward(false);
	output->RhoOutput(&elements);
	output->TxtOutputResistivity(&elements, "FinalResistivity.txt");
	infofile.close();

	string filename = "gradObjFunc.txt";
	output->OutputGradObjFunc(obj_valNotNormalized, &invertedRhoIDToElementVector, &dUdRho_output, filename);
	if (isInvertedDistortion) {
		filename = "Distortion.txt";
		output->OutputDistortionMatrix(&obsImpedanceElements, constraintDistortionTerm, filename);
		filename = "DistortionForRestart.txt";
		output->OutputDistortionMatrixForRestart(&obsImpedanceElements, filename);
	}

	
	//if (success) {
	//	std::cout << "cg: sphere test completed successfully." << std::endl;
	//}
	//else {
	//	std::cout << "cg: sphere test completed unsuccessfully." << std::endl;
	//}
}
void Analysis::Analysis::CountObsData() {
	numOfObsDataPerOmega.resize(boundary->omega.size());
	for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
		numOfObsDataPerOmega[iOmega] = 0;
	}
	//Count Data
	numOfObsData = 0;
	numOfImpedanceDataset = 0;
	numOfTipperDataset = 0;

	vector<int> lastID;
	lastID.resize(boundary->omega.size());
	for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
		lastID[iOmega] = 0;
	}
	for (int j = 0; j < numOfObsPointElements; j++) {
		Element::Element* element = obsPointElements[j];
		if (element->isInversionImpedance == true) {
			element->impedanceObsData->rowIDsJacobian.resize(boundary->omega.size());
			element->impedanceObsData->rowIDEachOmega.resize(boundary->omega.size());
			for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
				element->impedanceObsData->rowIDsJacobian[iOmega] = numOfObsData;
				numOfObsData += 4 * 2; //real and imag parts
				numOfObsDataPerOmega[iOmega] += 4 * 2;

				element->impedanceObsData->rowIDEachOmega[iOmega] = lastID[iOmega];
				lastID[iOmega] += 4 * 2;
				numOfImpedanceDataset++;

			}
		}
	}
	
	for (int j = 0; j < numOfObsPointElements; j++) {
		Element::Element* element = obsPointElements[j];
		if (element->isInversionTipper == true) {
			element->tipperObsData->rowIDsJacobian.resize(boundary->omega.size());
			element->tipperObsData->rowIDEachOmega.resize(boundary->omega.size());

			for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {	
				element->tipperObsData->rowIDsJacobian[iOmega] = numOfObsData;
				numOfObsData = numOfObsData + 2 * 2; //real and imag parts
				numOfObsDataPerOmega[iOmega] += 2 * 2;
				element->tipperObsData->rowIDEachOmega[iOmega] = lastID[iOmega];
				lastID[iOmega] += 2 * 2;
				numOfTipperDataset++;
			}
		}
		// Todo::他のテンソル量を逆解析する場合はここに足す
	}
	cout << "numOfObsStations:" << numOfObsPointElements << endl;
	cout << "numOfObsData:" << numOfObsData << endl;
	
}
void Analysis::Analysis::SetSameResistivityToBoundaryCell() {
	//Set To AirGroundBoundary
	for (int i = 0; i < numOfCalcElements; i++) {
		if (calcElementsVector[i]->MPCResistivityCoeff.size()!=0) {
			double resis = 0;
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(calcElementsVector[i]->MPCResistivityCoeff, 0); it; ++it) {
				resis += calcElementsVector[i]->MPCResistivityCoeff.coeff(0, it.col()) * calcElementsVector[it.col()]->resistivity;
			}
			calcElementsVector[i]->resistivity = resis;
		}
	}

	//First, decide resistivities of all inside elements.
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	Element::Element* element = calcElementsVector[i];
	//	Eigen::Vector3i pos;
	//	pos.setZero();
	//	if (element->property->type == Property::Property::NORMAL && element->isAirGroundBoundaryCell == true) {
	//	pos.coeffRef(2) = +1;
	//	}
	//	int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
	//	if (pos.coeff(0) == 0 && pos.coeff(1) == 0 && pos.coeff(2) == 0) {
	//		continue;
	//	}
	//	if (element->layer == element->neighborElements[ipos]->layer && element->neighborElements[ipos]->isParent == false) {
	//		element->masterResistivityElement = element->neighborElements[ipos];

	//		//the third subsurface layer 
	//		//if (element->neighborElements[ipos]->neighborElements[ipos]->layer == element->neighborElements[ipos]->layer && element->neighborElements[ipos]->neighborElements[ipos]->isParent == false) {
	//		//	element->neighborElements[ipos]->neighborElements[ipos]->masterResistivityElement = element->neighborElements[ipos];
	//		//}
	//		//else {
	//		//	cout << "The Third Subsurface Cell must be the same layer to the Second One" << endl;
	//		//	exit(-1);
	//		//}
	//		//

	//	}
	//	else {
	//		cout << "Boundary Cell must be the same layer to the neighbor" << endl;
	//		exit(-1);
	//	}
	//}

	//And then, decide boundary elements resistivities.
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	Element::Element* element = calcElementsVector[i];
	//	Eigen::Vector3i pos;
	//	pos.setZero();
	//	if (element->property->type == Property::Property::NORMAL && element->boundary == "-X_BOUNDARY") {
	//		pos.coeffRef(0) = 1;
	//	}
	//	else if (element->property->type == Property::Property::NORMAL && element->boundary == "+X_BOUNDARY") {
	//		pos.coeffRef(0) = -1;
	//	}
	//	else if (element->property->type == Property::Property::NORMAL && element->boundary == "-Y_BOUNDARY") {
	//		pos.coeffRef(1) = 1;
	//	}
	//	else if (element->property->type == Property::Property::NORMAL && element->boundary == "+Y_BOUNDARY") {
	//		pos.coeffRef(1) = -1;
	//	}
	//	else if (element->property->type == Property::Property::NORMAL && element->boundary == "-Z_BOUNDARY") {
	//		pos.coeffRef(2) = 1;
	//	}
	//	else if (element->property->type == Property::Property::NORMAL && element->boundary == "+Z_BOUNDARY") {
	//		pos.coeffRef(2) = -1;
	//	}
	//	int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
	//	if ((pos.coeff(0)==0 && pos.coeff(1)==0 && pos.coeff(2)==0) || element->isAirGroundBoundaryCell==true) {
	//		continue;
	//	}
	//	if (element->layer == element->neighborElements[ipos]->layer && element->neighborElements[ipos]->isParent==false) {
	//		element->masterResistivityElement = element->neighborElements[ipos];
	//	}
	//	else {
	//		cout << "Boundary Cell must be the same layer to the neighbor" << endl;
	//		exit(-1);
	//	}

	//}
	//Second ,set resistivity and foundamental parent elements(for example, master of mater)
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	Element::Element* element = calcElementsVector[i];
	//	element->masterResistivityElement = SearchMasterElement(element);
	//	if (element->masterResistivityElement != nullptr) {
	//		element->resistivity = element->masterResistivityElement->resistivity;
	//	}
	//}

	//At last, set second layer from air ground
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	Element::Element* element = calcElementsVector[i];
	//	if (element->isSecondCellOfAirGroundBoundary ==true) {
	//		element->resistivity = (element->neighborElements[1 + 3 + 0]->resistivity + element->neighborElements[1 + 3 + 9 * 2]->resistivity)/2.0;
	//	}
	//}

	//test, not slave but same resis in sueface layer
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	Element::Element* element = calcElementsVector[i];
	//	Eigen::Vector3i pos;
	//	pos.setZero();
	//	if (element->property->type == Property::Property::NORMAL && element->isAirGroundBoundaryCell == true) {
	//		pos.coeffRef(2) = +1;
	//	}
	//	int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
	//	element->resistivity = element->neighborElements[ipos]->resistivity;
	//}
}
bool Analysis::Analysis::CalcRhoFromParamAndDRhoDParam(Eigen::VectorXd paramVec) {
	
	dRhoDParam.setZero();
	bool isChangeResis = false;
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		kv::autodif<double> x;
		//if (paramVec.coeff(i) > limitOfparamLogNormalization){
		//	x = kv::autodif<double>::init(limitOfparamLogNormalization);
		//	dRhoDParam.coeffRef(i) = 1e-3;
		//}
		//else if (paramVec.coeff(i) < -limitOfparamLogNormalization) {
		//	x = kv::autodif<double>::init(-limitOfparamLogNormalization);
		//	dRhoDParam.coeffRef(i) = 1e-3;
		//}
		//else {

		double logMaxResis = log(maxResis);
		double logMinResis = log(minResis);
		//double logMaxResis = maxResis;
		//double logMinResis = minResis;
		x = kv::autodif<double>::init(paramVec.coeff(i));
		kv::autodif<double> enx = exp(paramLogNormalization*x);
		//kv::autodif<double> m = (maxResis*enx + minResis) / (1 + enx);
		kv::autodif<double> m = (logMaxResis*enx + logMinResis) / (1 + enx);
		kv::autodif<double> resis = pow(std::exp(1.0), m);
		//kv::autodif<double> resis = m;
		if (std::isnan(resis.v) == true || std::isnan(resis.d(0))==true) {
			double resisv;
			if (paramVec.coeff(i) > 0) {
				resisv = maxResis;
			}
			else {
				resisv = minResis;
			}
			invertedRhoIDToElementVector[i]->resistivity = resisv;
			dRhoDParam.coeffRef(i) = 0.0;
		}
		else {
			if (preParams[i] != paramVec[i]) {
				invertedRhoIDToElementVector[i]->resistivity = resis.v;
				isChangeResis = true;
			}
			dRhoDParam.coeffRef(i) = resis.d(0);
		}
		
		//if (paramVec.coeff(i) >= maxResis) {
		//	invertedRhoIDToElementVector[i]->resistivity = maxResis;
		//}
		//else if (paramVec.coeff(i) <= minResis) {
		//	invertedRhoIDToElementVector[i]->resistivity = minResis;
		//}
		//else {
		//	invertedRhoIDToElementVector[i]->resistivity = paramVec.coeffRef(i) ;
		//	isChangeResis = true;
		//}
		//dRhoDParam.coeffRef(i) = 1.0;

	}
	
	preParams = paramVec;

	return isChangeResis;
}
Eigen::VectorXd Analysis::Analysis::CalcParamFromRho() {
	Eigen::VectorXd paramVec;
	paramVec.resize(numOfInvertedResistivityElements);
	paramVec.setZero();
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		double m = log(invertedRhoIDToElementVector[i]->resistivity);
		//double m = invertedRhoIDToElementVector[i]->resistivity;
		//double x = 1 / paramLogNormalization * log((m - minResis) / (maxResis - m));
		double logMaxResis = log(maxResis);
		double logMinResis = log(minResis);
		//double logMaxResis = maxResis;
		//double logMinResis = minResis;
		double x = 1 / paramLogNormalization * log((m - logMinResis) / (logMaxResis - m));
		paramVec.coeffRef(i) = x;


		//paramVec.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity;

	}
	return paramVec;
}
Element::Element* Analysis::Analysis::SearchMasterElement(Element::Element* slaveElement) {
	
	if (slaveElement->masterResistivityElement != nullptr) {
		Element::Element* masterElem = slaveElement->masterResistivityElement;
		while (true) {
			if (masterElem->masterResistivityElement != nullptr) {
				masterElem=masterElem->masterResistivityElement;
			}
			else {
				return masterElem;
			}
		}
	}
	else {
		return nullptr;
	}

	
}

void Analysis::Analysis::CountIndependentInvertedResisElements() {
	numOfIndependentInvertedResisElements = 0;
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		if (invertedRhoIDToElementVector[i]->masterResistivityElement == nullptr) {
			numOfIndependentInvertedResisElements++;
		}
	}
}

void Analysis::Analysis::CalcJacobian(int iOmega) {
	//std::ofstream f2;
	//f2.open("debugZtmpdZdHtmp.txt", std::ios::trunc);
//	Eigen::SparseMatrix < std::complex<double>, Eigen::RowMajor> dRdRhotmp{ 2 * 3 * numOfCalcElements,numOfInvertedResistivityElements };
//
//	for (int i = 0; i < 2 * 3 * numOfCalcElements; i++) {
//		dRdRhotmp.row(i) = dRdRho->row(iOmega * 2 * 3 * numOfCalcElements + i);
//	}
//	Eigen::SparseMatrix<std::complex<double>> M1{ 3 * numOfCalcElements,int(numOfObsDataPerOmega[iOmega] /2) }; //once real part of Z is solved, imag part can be calculated by using it. 
//	Eigen::SparseMatrix<std::complex<double>> M2{ 3 * numOfCalcElements,int(numOfObsDataPerOmega[iOmega] /2) };
//
//	cout << "Calculating dDatadH matrix.." << endl;
//	for (int j = 0; j < numOfObsPointElements; j++) {
//		Element::Element* element = obsPointElements[j];
//		//===Impedance====
//		if (element->isInversionImpedance == true) {
//			for (int ii = 0; ii < 2; ii++) {
//				for (int jj = 0; jj < 2; jj++) {
//					for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dZdH[iOmega](ii, jj), 0); it; ++it)
//					{
//						int iCol = it.col();
//
//						/*cout << "in calcLambda" << ii << " " << jj << " " << element->dZdH[iOmega](ii, jj).coeff(0, 13981) << endl;*/
//						std::complex<double> dZdHtmp = element->dZdH[iOmega](ii, jj).coeff(0, iCol);
//						//f2 << iCol << " " << ii << " " << jj << " " << dZtmp << " " << dZdHtmp << "before" << endl;
//						double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
//						double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));
//						if (iCol < 3 * numOfCalcElements) {
//							//real part
//							M1.coeffRef(iCol, int(element->impedanceObsData->rowIDEachOmega[iOmega] / 2) + ii + 2 * jj).real(dZdHtmp.real() / epsReal);
//							//imag part
//							M1.coeffRef(iCol, int(element->impedanceObsData->rowIDEachOmega[iOmega] / 2) + ii + 2 * jj).imag(dZdHtmp.imag() / epsImag);
//						}
//						else {
//							//real part
//							M2.coeffRef(iCol - 3 * numOfCalcElements, int(element->impedanceObsData->rowIDEachOmega[iOmega] / 2) + ii + 2 * jj).real(dZdHtmp.real() / epsReal);
//							//imag part
//							M2.coeffRef(iCol - 3 * numOfCalcElements, int(element->impedanceObsData->rowIDEachOmega[iOmega] / 2) + ii + 2 * jj).imag(dZdHtmp.imag() / epsImag);
//						}
//					}
//				}
//			}
//		}
//		//===Tipper=====
//		if (element->isInversionTipper == true) {
//			for (int ii = 0; ii < 2; ii++) {
//				//Real Part
//				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dTdH[iOmega](ii), 0); it; ++it)
//				{
//					int iCol = it.col();
//					std::complex<double> dTdHtmp = element->dTdH[iOmega](ii).coeff(0, iCol);
//
//					double epsReal = std::abs(element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii));
//					double epsImag = std::abs(element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii));
//					if (iCol < 3 * numOfCalcElements) {
//						//real part
//						M1.coeffRef(iCol, int(element->tipperObsData->rowIDEachOmega[iOmega] / 2) + ii).real(dTdHtmp.real() / epsReal);
//						//imag part
//						M1.coeffRef(iCol, int(element->tipperObsData->rowIDEachOmega[iOmega] / 2) + ii).imag(dTdHtmp.imag() / epsImag);
//					}
//					else {
//						//real part
//						M2.coeffRef(iCol - 3 * numOfCalcElements, int(element->tipperObsData->rowIDEachOmega[iOmega] / 2) + ii).real(dTdHtmp.real() / epsReal);
//						//imag part
//						M2.coeffRef(iCol - 3 * numOfCalcElements, int(element->tipperObsData->rowIDEachOmega[iOmega] / 2) + ii).imag(dTdHtmp.imag() / epsImag);
//					}
//				}
//
//			}
//		}
//		// Todo::他のテンソル量を逆解析する場合はここに足す
//	}
//
//	cout << "Calculating Lambda.." << endl;
//	cout << "  Calculating Lambda of H1.." << endl;
//	time_t start_t = time(NULL);
//	Eigen::MatrixXcd res1{ 3 * numOfCalcElements,int(numOfObsDataPerOmega[iOmega] / 2) };
//	solver1->pardisoParameterArray()(11) = 1; //set to adjoint mode
//	res1 = solver1->solve(M1.conjugate());
//	solver1->pardisoParameterArray()(11) = 0; //reset to normal mode
//	time_t end_t = time(NULL);
//	std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
//
//	cout << "  Calculating Lambda of H2.." << endl;
//	start_t = time(NULL);
//	Eigen::MatrixXcd res2{ 3 * numOfCalcElements,int(numOfObsDataPerOmega[iOmega] / 2) };
//	solver2->pardisoParameterArray()(11) = 1; //set to adjoint mode
//	res2 = solver2->solve(M2.conjugate());
//	solver2->pardisoParameterArray()(11) = 0; //reset to normal mode
//	end_t = time(NULL);
//	std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
//	cout << "End Calculating Lambda.." << endl;
//
//
//	//Set Lambda, this includes real and imag parts of Z
//	Eigen::MatrixXcd lambdaTmp{2 * 3 * numOfCalcElements,numOfObsDataPerOmega[iOmega] };
//
//	for (int j = 0; j<int(numOfObsDataPerOmega[iOmega] / 2); j++) {
//		lambdaTmp.block(0, 2 * j, 3 * numOfCalcElements, 1) = res1.col(j);
//		lambdaTmp.block(3 * numOfCalcElements, 2 * j, 3 * numOfCalcElements, 1) = std::complex<double>(0, 1)* res2.col(j);
//		//set using Cauchy–Riemann equations
//		lambdaTmp.block(0, 2 * j + 1, 3 * numOfCalcElements, 1) = res1.col(j);
//		lambdaTmp.block(3 * numOfCalcElements, 2 * j + 1, 3 * numOfCalcElements, 1) = std::complex<double>(0, 1)* res2.col(j);
//		cout << j << endl;
//		//for (int i = 0; i < 3 * numOfCalcElements; i++) {
//		//	lambdaTmp.coeffRef(i, 2 * j) = res1.coeffRef(i, j);
//		//	lambdaTmp.coeffRef(3*numOfCalcElements + i, 2 * j) = res2.coeffRef(i, j);
//		//	//set using Cauchy–Riemann equations
//		//	lambdaTmp.coeffRef(i, 2 * j + 1) = std::complex<double>(0,1)* res1.coeffRef(i, j);
//		//	lambdaTmp.coeffRef(3 * numOfCalcElements + i, 2 * j + 1) = std::complex<double>(0, 1)*res2.coeffRef(i, j);
//		//	cout << i << " " << j << endl;
//		//}
//	}
//
//	//calc jacobian
//	Eigen::MatrixXd jac{ numOfObsDataPerOmega[iOmega],numOfInvertedResistivityElements };
//	//∂Z/∂Rho
//	start_t = time(NULL);
//	cout << "Calculating Jacobian #Omega:" << iOmega << endl;
//	for (int j = 0; j < numOfObsPointElements; j++) {
//		Element::Element* element = obsPointElements[j];
//		//===Impedance====
//		if (element->isInversionImpedance == true) {
//			for (int ii = 0; ii < 2; ii++) {
//				for (int jj = 0; jj < 2; jj++) {
//					for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dZdRho[iOmega](ii, jj), 0); it; ++it)
//					{
//						int iCol = calcElementsVector[it.col()]->invertedRhoElementsID;
//						if (iCol >= 0) {
//							std::complex<double> dZdRhotmp = element->dZdRho[iOmega](ii, jj).coeff(0, iCol);
//							double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
//							double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));
//							//d(ReZ)/dRho
//							cout << numOfObsDataPerOmega[iOmega] << " " << element->impedanceObsData->rowIDEachOmega[iOmega] << " " << ii << " " << jj << " " << iCol << endl;
//							jac.coeffRef(element->impedanceObsData->rowIDEachOmega[iOmega] + 2 * ii + 2 * 2 * jj, iCol) += dZdRhotmp.real() / epsReal;
//							//d(ReZ)/dRho
//							jac.coeffRef(element->impedanceObsData->rowIDEachOmega[iOmega] + 2 * ii + 2 * 2 * jj + 1, iCol) += dZdRhotmp.imag() / epsImag;
//						}
//					}
//				}
//			}
//		}
//	}
//	//Tipper, ∂T/∂Rho=0
//	//Eigen::setNbThreads(0);
//	Eigen::initParallel();
//	Eigen::MatrixXd lambdaDRDRho{ numOfObsDataPerOmega[iOmega],numOfInvertedResistivityElements };
//	lambdaDRDRho = (lambdaTmp.adjoint()*dRdRhotmp).real();
//	jac = jac - lambdaDRDRho;
//	//Eigen::setNbThreads(1);
//	Eigen::initParallel();
//	//set jac to global jacobian
//	int startID = 0;
//	for (int i = 0; i < iOmega - 1; i++) {
//		startID += numOfObsDataPerOmega[i];
//	}
//
//	cout << "startID" << startID << endl;
//	for (int i = 0; i < numOfObsDataPerOmega[iOmega]; i++) {
//		jacobian->row(startID + i) = jac.row(i);
//	}
//
//	std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
//	cout << "End Calculating Jacobian #Omega:" << iOmega << endl;
//}
}
namespace {
	struct KDTree3 {
		struct Node {
			int idx;         
			int axis;        
			int left;        
			int right;       
			double split;     
		};

		const std::vector<InitialResistivityData::InitialResistivityData*>& pts;
		std::vector<int> order;                         
		std::vector<Node> nodes;                         
		int root = -1;

		KDTree3(const std::vector<InitialResistivityData::InitialResistivityData*>& points) : pts(points) {
			const int n = static_cast<int>(pts.size());
			order.resize(n);
			std::iota(order.begin(), order.end(), 0);
			nodes.reserve(n * 2); 
			root = build(0, n, 0); 
		}

		inline double coord(int i, int axis) const {

			return pts[i]->coord.coeff(axis);
		}


		int build(int l, int r, int axis) {
			if (l >= r) return -1;
			int m = (l + r) / 2;
			auto comp = [&](int a, int b) { return coord(a, axis) < coord(b, axis); };
			std::nth_element(order.begin() + l, order.begin() + m, order.begin() + r, comp);

			Node node;
			node.idx = order[m];
			node.axis = axis;
			node.split = coord(node.idx, axis);
			node.left = build(l, m, (axis + 1) % 3);
			node.right = build(m + 1, r, (axis + 1) % 3);

			int id = (int)nodes.size();
			nodes.push_back(node);
			return id;
		}


		int nearest(const double q[3]) const {
			int best_idx = -1;
			double best_d2 = std::numeric_limits<double>::infinity();


			struct Frame { int node; };
			std::vector<Frame> st;
			if (root != -1) st.push_back({ root });

			while (!st.empty()) {
				int ni = st.back().node;
				st.pop_back();
				const Node& nd = nodes[ni];

				double dx = q[0] - coord(nd.idx, 0);
				double dy = q[1] - coord(nd.idx, 1);
				double dz = q[2] - coord(nd.idx, 2);
				double d2 = dx * dx + dy * dy + dz * dz;
				if (d2 < best_d2) {
					best_d2 = d2;
					best_idx = nd.idx;
				}


				double diff = q[nd.axis] - nd.split;
				int first = diff <= 0.0 ? nd.left : nd.right;
				int second = diff <= 0.0 ? nd.right : nd.left;

				if (first != -1) st.push_back({ first });


				if (second != -1 && diff * diff < best_d2) {
					st.push_back({ second });
				}
			}
			return best_idx; 
		}
	};
}
void Analysis::Analysis::SetInitialResistivityFromFile() { //Set Nearest Resistivity in InitialData
	if (initialResistivityData.size() == 0) {
		return;
	}
	bool flagSetAir = false;
	int numOfSetNormal = 0;
	bool flagSetMinResis = false;
	bool isThereAirCells = false;
	double maxCellsResis = 0;
	for (int j = 0; j < initialResistivityData.size(); j++) {
		if (initialResistivityData[j]->resistivity > invSettings->maxResis) {
			isThereAirCells = true;
			if (initialResistivityData[j]->resistivity > maxCellsResis) {
				maxCellsResis = initialResistivityData[j]->resistivity;
			}
		}
	}
	if (!isThereAirCells) {
		for (int j = 0; j < initialResistivityData.size(); j++) {
			if (initialResistivityData[j]->resistivity > maxCellsResis) {
				maxCellsResis = initialResistivityData[j]->resistivity;
			}
		}
		cout << "Warning!!!!!:All Resistivity from the file are less than maxResis!!! Set maxResis to maximum of Resistivity from the file - 1e-3!!!" << endl;
		cout << "This Warning appears When You Set Initial Resistivity From the File!!!\n" << endl;
		invSettings->maxResis = maxCellsResis - 1e-3;
		maxResis = invSettings->maxResis;
	}
	
	KDTree3 kdt(initialResistivityData);

	for (int i = 0; i<numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];
		/*double nearestDistance = 1e30;
		int nearestID = -1;
		for (int j = 0; j < initialResistivityData.size(); j++) {
			double distance = (element->centerCoord - initialResistivityData[j]->coord).norm();
			if (distance < nearestDistance) {
				nearestDistance = distance;
				nearestID = initialResistivityData[j]->ID;
			}
		}
		if (nearestID == -1) {
			cout << "Error In SetInitialResistivityFromFile" << endl;
			exit(1);
		}*/

		double q[3] = {
		element->centerCoord.coeff(0),
		element->centerCoord.coeff(1),
		element->centerCoord.coeff(2)
		};

		int ret_index = kdt.nearest(q);
		if (ret_index < 0) {
			std::cout << "Error In SetInitialResistivityFromFile (KD-tree search failed)\n";
			std::exit(1);
		}


		int nearestID = initialResistivityData[ret_index]->ID;

		//cout << initialResistivityData[nearestID]->resistivity << invSettings->maxResis << endl;
		if (initialResistivityData[nearestID]->resistivity > invSettings->maxResis ) {
			for (auto itr2 = propertiesVector.begin(); itr2 != propertiesVector.end(); itr2++) {
				Property::Property* property = *itr2;
				if (property->type == Property::Property::AIR) { //Assume that element resis is set to the first air property
					element->property = property;
					element->resistivity = property->resistivity;
					element->initialResistivity = property->resistivity;
					if (element->property->type != Property::Property::AIR) {
						flagSetAir = true;
					}
					break;
				}
			}
		}
		else {
			if (initialResistivityData[nearestID]->resistivity <= invSettings->minResis) {
				initialResistivityData[nearestID]->resistivity = invSettings->minResis;
				flagSetMinResis = true;
			}
			else {
				element->resistivity = initialResistivityData[nearestID]->resistivity;
			}
			if (element->property->type == Property::Property::AIR) {
				
				for (auto itr2 = propertiesVector.begin(); itr2 != propertiesVector.end(); itr2++) {
					Property::Property* property = *itr2;
					if (property->type == Property::Property::NORMAL) { //Assume that element property is normal
						element->property = property;
						numOfSetNormal++;
						break;
					}
				}
			}
		}
	}
	if (flagSetAir) {
		cout << "WARNING::Elements Whose Resistivities had been Over maxResis were Set To Air!!!" << endl;
	}
	if (flagSetMinResis) {
		cout << "WARNING::Elements Whose Resistivities had been  under mInResis were Set To minResis!!!" << endl;
	}
	

	//initialResistivityFile includes transition zone elements from air to transition, but in the file it cannot be distinguished, so need to reset them.
	Property::Property* airProp = propertiesVector[0];
	for (auto itr2 = propertiesVector.begin(); itr2 != propertiesVector.end(); itr2++) {
		Property::Property* property = *itr2;
		if (property->type == Property::Property::AIR) { //Assume that element resis is set to the first air property
			airProp = property;
			break;
		}
	}

	Eigen::VectorXi isReset{numOfCalcElements};
	isReset.setZero();
	for (int i = 0; i < numOfCalcElements; i++) {
		bool flg = calcElementsVector[i]->ResetTransitionZone(&elements, numOfCalcElements, &propertiesVector);
		if (flg) {
			isReset.coeffRef(i) = 1;
			numOfSetNormal--;
		}
	}
	for (int i = 0; i < numOfCalcElements; i++) {
		if (isReset.coeff(i)>0) {
			calcElementsVector[i]->property = airProp;
		}
	}
	if (numOfSetNormal > 0) {
		cout << "WARNING::Elements Whose Properties had been Air and had under maxResis were Set To Type Normal!!!" << endl;
	}
}

void Analysis::Analysis::CalcDivergenceElements() {

	//SetSameLayerElements();
	vector<vector < Eigen::Triplet<double>>> divHdSTripletEachThread(omp_get_max_threads());
	for (int iLayer = maxLayer; iLayer >= 0; iLayer--) {
//#pragma omp parallel for
		for (int i = 0; i < sameLayerElementsVector[iLayer].size(); i++) {
			int threadID = omp_get_thread_num();
			int ipos;
			//for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
			Element::Element* element = sameLayerElementsVector[iLayer][i];
			if (element->boundary != "NOT_BOUNDARY") {
				continue;
			}


			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> sumNdivHdS{ 3,3 * numOfCalcElements };
			sumNdivHdS.reserve(Eigen::VectorXi::Constant(3, element->numOfRelatedCalcVariables));//reserve as maximum as possible 

			element->CalcSumNDivHdS(&elements, numOfCalcElements, &sumNdivHdS);
			for (int j = 0; j < 3; j++) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(sumNdivHdS, j); it; ++it) {
					Eigen::Triplet<double> val(3*sameLayerElementsVector[iLayer][i]->calcID+j ,it.col(), sumNdivHdS.coeff(j, it.col()).real());
					divHdSTripletEachThread[threadID].push_back(val);
				}
			}

		}
	}


	//assemble
	vector< Eigen::Triplet<double>> divHdSTriplet;

	for (int i = 0; i < omp_get_max_threads(); i++) {
		for (int j = 0; j < divHdSTripletEachThread[i].size(); j++) {
			divHdSTriplet.push_back(divHdSTripletEachThread[i][j]);
		}
	}

	//make matrix
	divergenceCorrection->sumDivHdSMatrix.setFromTriplets(divHdSTriplet.begin(), divHdSTriplet.end());
	divergenceCorrection->sumDivHdSMatrix.pruned();
		
}
vector<vector<vector<complex<double>>>> Analysis::Analysis::Calc1D() {
	vector<vector<vector<complex<double>>>> result;
	result.resize(nx);
	for (int i = 0; i < nx; i++) {
		result[i].resize(ny);
		for (int j = 0; j < ny; j++) {
			result[i][j].resize(nz);
			Eigen::MatrixXcd mat(nz,nz);
			mat.setZero();
			Eigen::VectorXcd vec(nz);
			vec.setZero();
			for (int k = 0; k < nz; k++) {
				if (k == 0) {
					mat.coeffRef(0, 0) = 1.0; //boundary condition
					vec(0) = 1.0;
					continue;
				}
				else if (k == nz - 1) {
					mat.coeffRef(nz-1, nz-1) = 1.0; //boundary condition
					continue;
				}
				Element::Element* elementp = basedElementsSortByNxNyNz[i][j][k+1];
				Element::Element* element = basedElementsSortByNxNyNz[i][j][k];
				Element::Element* elementm = basedElementsSortByNxNyNz[i][j][k-1];
				double dzp = elementp->dz;
				double dzi = element->dz;
				double dzm = elementm->dz;
				mat.coeffRef(k, k + 1) += 2.0 / dzi / (dzp + dzi);
				mat.coeffRef(k, k) -= 2.0 / dzi / (dzp + dzi);
				mat.coeffRef(k, k) -= 2.0 / dzi / (dzi + dzm);
				mat.coeffRef(k, k - 1) += 2.0 / dzi / (dzi + dzm);

				std::complex<double> term;
				term.imag(+omega * mu / element->resistivity);
				mat.coeffRef(k, k) -= term;
			}
			Eigen::VectorXcd tmp = mat.lu().solve(vec);
			for (int k = 0; k < nz; k++) {
				result[i][j][k] = tmp.coeff(k);
			}
			
		}
	}
	return result;
}

void Analysis::Analysis::CalcDDataMisfitDDistortionParam() {
	for (int i = 0; i < numOfObsImpedanceElements; i++) {
		Element::Element* element = obsImpedanceElements[i];
		ub::vector<kv::autodif<double>>distortionParam;
		ub::vector<double> tmp(4);
		tmp(0) = element->distortionMatrix.coeff(0, 0);
		tmp(1) = element->distortionMatrix.coeff(0, 1);
		tmp(2) = element->distortionMatrix.coeff(1, 0);
		tmp(3) = element->distortionMatrix.coeff(1, 1);
		
		distortionParam = kv::autodif<double>::init(tmp);

		
		
		
		kv::autodif<double> dz;
		for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
			
			//1,1
			{
				double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(0, 0));
				double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(0, 0));
				kv::autodif<double> dzR;
				dzR = distortionParam(0) * element->Z[iOmega].coeff(0, 0).real() + distortionParam(1) * element->Z[iOmega].coeff(1, 0).real(); //component 1,1
				dzR -= element->impedanceObsData->ZobsVector[iOmega].coeff(0, 0).real();
				dzR = dzR / epsReal;
				dzR = dzR * dzR;
				kv::autodif<double> dzI;
				dzI = distortionParam(0) * element->Z[iOmega].coeff(0, 0).imag() + distortionParam(1) * element->Z[iOmega].coeff(1, 0).imag(); //component 1,1
				dzI -= element->impedanceObsData->ZobsVector[iOmega].coeff(0, 0).imag();
				dzI = dzI / epsImag;
				dzI = dzI * dzI;
				dz += dzR + dzI;
			}
			//1,2
			{
				double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(0, 1));
				double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(0, 1));
				kv::autodif<double> dzR;
				dzR = distortionParam(0) * element->Z[iOmega].coeff(0, 1).real() + distortionParam(1) * element->Z[iOmega].coeff(1, 1).real(); //component 1,1
				dzR -= element->impedanceObsData->ZobsVector[iOmega].coeff(0, 1).real();
				dzR = dzR / epsReal;
				dzR = dzR * dzR;
				kv::autodif<double> dzI;
				dzI = distortionParam(0) * element->Z[iOmega].coeff(0, 1).imag() + distortionParam(1) * element->Z[iOmega].coeff(1, 1).imag(); //component 1,1
				dzI -= element->impedanceObsData->ZobsVector[iOmega].coeff(0, 1).imag();
				dzI = dzI / epsImag;
				dzI = dzI * dzI;
				dz += dzR + dzI;
			}
			//2,1
			{
				double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(1, 0));
				double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(1, 0));
				kv::autodif<double> dzR;
				dzR = distortionParam(2) * element->Z[iOmega].coeff(0, 0).real() + distortionParam(3) * element->Z[iOmega].coeff(1, 0).real(); //component 1,1
				dzR -= element->impedanceObsData->ZobsVector[iOmega].coeff(1, 0).real();
				dzR = dzR / epsReal;
				dzR = dzR * dzR;
				kv::autodif<double> dzI;
				dzI = distortionParam(2) * element->Z[iOmega].coeff(0, 0).imag() + distortionParam(3) * element->Z[iOmega].coeff(1, 0).imag(); //component 1,1
				dzI -= element->impedanceObsData->ZobsVector[iOmega].coeff(1, 0).imag();
				dzI = dzI / epsImag;
				dzI = dzI * dzI;
				dz += dzR + dzI;
			}
			//2,2
			{
				double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(1, 1));
				double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(1, 1));
				kv::autodif<double> dzR;
				dzR = distortionParam(2) * element->Z[iOmega].coeff(0, 1).real() + distortionParam(3) * element->Z[iOmega].coeff(1, 1).real(); //component 1,1
				dzR -= element->impedanceObsData->ZobsVector[iOmega].coeff(1, 1).real();
				dzR = dzR / epsReal;
				dzR = dzR * dzR;
				kv::autodif<double> dzI;
				dzI = distortionParam(2) * element->Z[iOmega].coeff(0, 1).imag() + distortionParam(3) * element->Z[iOmega].coeff(1, 1).imag(); //component 1,1
				dzI -= element->impedanceObsData->ZobsVector[iOmega].coeff(1, 1).imag();
				dzI = dzI / epsImag;
				dzI = dzI * dzI;
				dz += dzR + dzI;
			}

		}

		dDataMisfitDDistortionParam.coeffRef(4 * element->impedanceObsID) = dz.d(0);
		dDataMisfitDDistortionParam.coeffRef(4 * element->impedanceObsID + 1) = dz.d(1);
		dDataMisfitDDistortionParam.coeffRef(4 * element->impedanceObsID + 2) = dz.d(2);
		dDataMisfitDDistortionParam.coeffRef(4 * element->impedanceObsID + 3) = dz.d(3);
	}
	



}
void Analysis::Analysis::CalcConstraintDistortionTerm() {
	constraintDistortionTerm = 0.0;
	if (isInvertedDistortion) {
		for (int i = 0; i < numOfObsImpedanceElements; i++) {
			for (int j = 0; j < 2; j++) {
				for (int k = 0; k < 2; k++) {
					if ((j == 0 && k == 0) || (j==1 && k==1)) { //Constraint by similar to identity matrix
						constraintDistortionTerm += pow(obsImpedanceElements[i]->distortionMatrix.coeff(j, k) - 1, 2.0);
					}
					else {
						constraintDistortionTerm += pow(obsImpedanceElements[i]->distortionMatrix.coeff(j, k), 2.0);
					}
				}
			}
		}
	}
}
void Analysis::Analysis::SetInitialDistortionFromFile() { //Set Nearest Resistivity in InitialData
	if (initialDistortionData.size() == 0) {
		return;
	}

	for (int i = 0; i < numOfObsImpedanceElements; i++) {
		Element::Element* element = obsImpedanceElements[i];
		//double nearestDistance = 1e30;
		int nearestID = -1;
		double eps = 1e-6;
		for (int j = 0; j < initialDistortionData.size(); j++) {
			/*double distance = pow(pow(element->centerCoord.coeff(0) - initialDistortionData[j]->coord.coeff(0), 2.0) +
				pow(element->centerCoord.coeff(1) - initialDistortionData[j]->coord.coeff(1), 2.0), 0.5);
			if (distance < nearestDistance) {
				nearestDistance = distance;*/
			Eigen::Vector3d X;
			X.coeffRef(0) = initialDistortionData[j]->coord.coeff(0);
			X.coeffRef(1) = initialDistortionData[j]->coord.coeff(1);
			X.coeffRef(2) = element->centerCoord.coeff(2);
			if (element->CheckThePointInside2D(X,4)){
				
				nearestID = initialDistortionData[j]->ID;
				break;
			}
			
		}
		if (nearestID == -1) {
			cout << "Error In SetInitialDistortionFromFile: Could not Find Distortion Data For Obs Point Element::" << endl;
			cout << "ID:" << element->ID << endl;
			exit(1);
		}
		element->distortionMatrix = initialDistortionData[nearestID]->distortionMatrix;
	}
	if (isInvertedDistortion) {
		string filename = "InitialDistortion.txt";
		output->OutputDistortionMatrix(&obsImpedanceElements, 0.0, filename);
	}
}
void Analysis::Analysis::SetSameLayerElements() {
	if (sameLayerElementsVector.size() == 0) {
		for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
			Element::Element* element = *itr;
			if (element->layer > maxLayer) {

				maxLayer = element->layer;
			}
		}
		for (int iLayer = 0; iLayer <= maxLayer; iLayer++) {
			vector < Element::Element* > layerElementsVector;
			for (int i = 0; i < calcElementsVector.size(); i++) {
				Element::Element* element = calcElementsVector[i];
				if (element->layer == iLayer) {
					layerElementsVector.push_back(element);
				}
			}
			sameLayerElementsVector.push_back(layerElementsVector);
		}
	}
	return;
}
void Analysis::Analysis::SetOutput() {
	int count = 0;
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		Element::Element* element = invertedRhoIDToElementVector[i];
		if (element->property->type == Property::Property::NORMAL) {		
			outputSensitivityVector.push_back(element);
			element->outputSensitivityID = count;
			count++;
		}
	}
	dJExceptRoughnessDRho.resize(count);
	dJExceptRoughnessDRho.setZero();

}

























void Analysis::Analysis::RunLocationCalc() {
	//mkl_set_num_threads(omp_get_max_threads());

	isDirectSolver = false;

	std::cout << "omp_get_max_threads:" << omp_get_max_threads() << std::endl;

	startCalc_t = time(NULL);
	std::cout << ("Initialize Data") << std::endl;
	Initialize();
	std::cout << ("Initialization End") << std::endl;
	//for (int i = 0; i < numOfObsImpedanceElements; i++) {
	//	Element::Element* element = obsImpedanceElements[i];
	//	for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
	//		for (int ii = 0; ii < 2; ii++) {
	//			for (int jj = 0; jj < 2; jj++) {
	//				element->impedanceObsData->ZobsVector[iOmega].coeffRef(ii, jj) = 0.0; //Set to Zero On This Calculation.
	//			}
	//		}
	//	}
	//}

	output->VTKObsPointsFileOutput(&obsPointElements);
	//output->VTKFileOputput(0.0, &elements, "ObsPoints");

	Eigen::VectorXd initialResis{ numOfInvertedResistivityElements };
	for (int i = 0; i < numOfInvertedResistivityElements; ++i) {
		initialResis.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity;
	}

	for (int loop = 0; loop < locPointElements.size(); loop++) {
		numOfParameters = numOfInvertedResistivityElements;
		locElement = locPointElements[loop];


		initObjVal = 0.0;

		CalcSurfaceResistivityElements(); //Update Surface Resistivity
		time_t start_t = time(NULL);
		cout << "Update SumNCrossRhoRotHdS.." << endl;
		CalcSumNCrossRhoRotHdSElements(); //Update coeffs of Matrix
		time_t end_t = time(NULL);
		std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
		cout << "End Update SumNCrossRhoRotHdS.." << endl;

		lambdaDRDRho.setZero();
		CalcForward(true,false,true); //First, Calc DZ_locDRho


		for (int i = 0; i < numOfObsImpedanceElements; i++) {
			Element::Element* element = obsImpedanceElements[i];
			for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
				for (int ii = 0; ii < 2; ii++) {
					for (int jj = 0; jj < 2; jj++) {
						element->Zpre[iOmega].coeffRef(ii, jj) = element->Z[iOmega].coeff(ii,jj);
					}
				}
			}
		}

		double initialMisfit = CalcDataMisfit(false);

		double initialOnePointObjValue= CalcDataMisfit(true);

		dDataMisfitDRho.setZero();
		CalcDDataMisfitDRho(true);
		weightRoughening = 0.0;
		CalcDJDRho(false);
		
		Eigen::VectorXd dJDRhoOnlyResis{ numOfInvertedResistivityElements };
		for (int i = 0; i < numOfInvertedResistivityElements; ++i) {
			dJDRhoOnlyResis.coeffRef(i) = dJdRho.coeff(i);
		}
		
		//double unit = 1;// initialOnePointObjValue;
		
		//Now, We could get dJdRho=∑dZ_loc/dRho

		//Next, Calculate data misfit increase using dRho
		
		Eigen::VectorXd dMisfit{ 2*locationCalcSettings->numOfSplit };

		string filename = "Sensitivity_LocPoint_" + std::to_string(loop) +  ".vtk";
		output->VTKFileOputput(&outputSensitivityVector, &dJExceptRoughnessDRho, filename);

		std::ofstream f;
		std::string fn = "locationCalc_" + to_string(loop) + ".txt";
		f.open(fn, std::ios::trunc);
		f << "ID," << loop << endl;
		f << "XY," << locElement->centerCoord.coeff(0) << "," << locElement->centerCoord.coeff(1) << endl;
		f << "InitialFuncVal," << initialMisfit << endl;
		f << "numberOfEstimationPoints,ratio,alpha,delta_dataMisfit,actualRatio,initialDataMisfit,postDataMisfit,maxProbIncreaseInObs" << endl;
		for (int ip = 0; ip < locationCalcSettings->numOfSplit; ip++) {

			//Plus Direction
			double sum = 0;
			for (int i = 0; i < numOfInvertedResistivityElements; ++i) {
				sum +=  dJDRhoOnlyResis.coeff(i) * dJDRhoOnlyResis.coeff(i);
			}
			double ratio = (locationCalcSettings->widthImpedance - 1.0) * double(ip + 1) / double(locationCalcSettings->numOfSplit);
			double alpha = ratio * initialOnePointObjValue / sum; 

			//double ratio = locationCalcSettings->widthImpedance;
			//double alpha = ratio;

			

			for (int i = 0; i < numOfInvertedResistivityElements; ++i) {
				if (invertedRhoIDToElementVector[i]->resistivity + alpha * dJDRhoOnlyResis.coeff(i)  < 0.0) {
					cout << "Warining!!!:Estimated Resistivity is less than zero!!! Set the Resistivity to Minimum:" << minResis << endl;
					cout << "Coord:" << invertedRhoIDToElementVector[i]->centerCoord << endl;
					cout << "Original Replaced Value:" << invertedRhoIDToElementVector[i]->resistivity + alpha * dJDRhoOnlyResis.coeff(i)  << endl;
				}
			}
			//Update Resis
			for (int i = 0; i < numOfInvertedResistivityElements; ++i) {
				invertedRhoIDToElementVector[i]->resistivity += alpha * dJDRhoOnlyResis.coeff(i) ;
				if (invertedRhoIDToElementVector[i]->resistivity < minResis) {
					invertedRhoIDToElementVector[i]->resistivity = minResis;
				}
				if (invertedRhoIDToElementVector[i]->resistivity > maxResis) {
					invertedRhoIDToElementVector[i]->resistivity = maxResis;
				}
			}

			CalcSurfaceResistivityElements(); //Update Surface Resistivity
			time_t start_t = time(NULL);
			cout << "Update SumNCrossRhoRotHdS.." << endl;
			CalcSumNCrossRhoRotHdSElements(); //Update coeffs of Matrix
			time_t end_t = time(NULL);
			std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
			cout << "End Update SumNCrossRhoRotHdS.." << endl;


			//Calc data misfit
			CalcForward(false, false, false); //data misfit


			for (int i = 0; i < numOfObsImpedanceElements; i++) {
				Element::Element* element = obsImpedanceElements[i];
				for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
					for (int ii = 0; ii < 2; ii++) {
						for (int jj = 0; jj < 2; jj++) {
							element->Zpos[iOmega].coeffRef(ii, jj) = element->Z[iOmega].coeff(ii, jj); 
							//element->Z[iOmega].coeffRef(ii, jj) = element->Zpos[iOmega].coeff(ii, jj) - element->Zpre[iOmega].coeff(ii, jj);
						}
					}
				}
			}
			//double dFuncVal = CalcMaxImpedanceDataComparedToVarience();
			/*Eigen::Vector3d returnVal;
			returnVal = CalcMaxDatafitChangeUsingNormalDistribution();
			double dFuncVal = returnVal.coeff(0);*/
			double postMisfit = CalcDataMisfit(false);
			

			//Calc Actual moving value for the direction
			double onePointObjValue = CalcDataMisfit(true);

			//Calc MaxDataMisfit Increase
			Eigen::VectorXd probMaxIncreaseEachObsPoints=CalcEachObservationProbIncrease();

			//Recover Resis and set for output
			Eigen::VectorXd diffRho{ numOfInvertedResistivityElements };
			for (int i = 0; i < numOfInvertedResistivityElements; ++i) {
				diffRho.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity - initialResis.coeff(i);
				invertedRhoIDToElementVector[i]->resistivity = initialResis.coeff(i);
			}

			std::string filename = "DiffRho_LocationCalc_" + std::to_string(loop)+"_" + std::to_string(ip) + ".vtk";
			output->VTKFileOputput(&invertedRhoIDToElementVector, &diffRho, filename);
			filename = "Rho_LocationCalc_" + std::to_string(loop) + "_" + std::to_string(ip) + ".txt";
			output->TxtOutputResistivity(&elements, filename);
			f << ip << "," << ratio<<","<<alpha<<","<< postMisfit-initialMisfit <<","<<onePointObjValue/initialOnePointObjValue- 1.0 <<","<<initialMisfit<<","<< postMisfit<<","<<probMaxIncreaseEachObsPoints.maxCoeff() << endl;
			
			filename = "EachObsPointsProbIncrease_" + std::to_string(loop) + "_" + std::to_string(ip) + ".txt";
			output->TxtOutput(&obsPointElements, &probMaxIncreaseEachObsPoints, filename);
		}



		for (int ip = 0; ip < locationCalcSettings->numOfSplit; ip++) {

			//Minus Direction
			double sum = 0;
			for (int i = 0; i < numOfInvertedResistivityElements; ++i) {
				sum += dJDRhoOnlyResis(i)* dJDRhoOnlyResis.coeff(i);
			}

			double ratio = (1.0 -1.0/locationCalcSettings->widthImpedance) * double(ip + 1) / double(locationCalcSettings->numOfSplit) ;
			double alpha = ratio * initialOnePointObjValue / sum;

			//double ratio = locationCalcSettings->widthImpedance;
			//double alpha = ratio;

			for (int i = 0; i < numOfInvertedResistivityElements; ++i) {
				if (invertedRhoIDToElementVector[i]->resistivity - alpha * dJDRhoOnlyResis.coeff(i) < 0.0) {
					cout << "Warining!!!:Estimated Resistivity is less than zero!!! Set the Resistivity to Minimum:" << minResis << endl;
					cout << "Coord:"  << invertedRhoIDToElementVector[i]->centerCoord << endl;
					cout << "Original Replaced Value:" << invertedRhoIDToElementVector[i]->resistivity - alpha * dJDRhoOnlyResis.coeff(i)  << endl;
				}
			}

			//Update Resis
			for (int i = 0; i < numOfInvertedResistivityElements; ++i) {
				invertedRhoIDToElementVector[i]->resistivity -= alpha * dJDRhoOnlyResis.coeff(i) ;
				if (invertedRhoIDToElementVector[i]->resistivity < minResis) {
					invertedRhoIDToElementVector[i]->resistivity = minResis;
				}
				if (invertedRhoIDToElementVector[i]->resistivity > maxResis) {
					invertedRhoIDToElementVector[i]->resistivity = maxResis;
				}
			}

			CalcSurfaceResistivityElements(); //Update Surface Resistivity
			time_t start_t = time(NULL);
			cout << "Update SumNCrossRhoRotHdS.." << endl;
			CalcSumNCrossRhoRotHdSElements(); //Update coeffs of Matrix
			time_t end_t = time(NULL);
			std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
			cout << "End Update SumNCrossRhoRotHdS.." << endl;


			//Calc data misfit
			
			CalcForward(false, false, false); //data misfit

			for (int i = 0; i < numOfObsImpedanceElements; i++) {
				Element::Element* element = obsImpedanceElements[i];
				for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
					for (int ii = 0; ii < 2; ii++) {
						for (int jj = 0; jj < 2; jj++) {
							element->Zpos[iOmega].coeffRef(ii, jj) = element->Z[iOmega].coeff(ii, jj); //Set to Zero On This Calculation.
							//element->Z[iOmega].coeffRef(ii, jj) = element->Zpos[iOmega].coeff(ii, jj) - element->Zpre[iOmega].coeff(ii, jj);
						}
					}
				}
			}
			//double dFuncVal = CalcMaxImpedanceDataComparedToVarience();
			/*Eigen::Vector3d returnVal;
			returnVal = CalcMaxDatafitChangeUsingNormalDistribution();
			double dFuncVal = returnVal.coeff(0);*/
			double postMisfit = CalcDataMisfit(false);

			//Calc Actual moving value for the direction
			double onePointObjValue = CalcDataMisfit(true);

			//Calc MaxDataMisfit Increase
			Eigen::VectorXd probMaxIncreaseEachObsPoints = CalcEachObservationProbIncrease();

			//Recover Resis and set for output
			Eigen::VectorXd diffRho{ numOfInvertedResistivityElements };
			for (int i = 0; i < numOfInvertedResistivityElements; ++i) {
				diffRho.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity - initialResis.coeff(i);
				invertedRhoIDToElementVector[i]->resistivity = initialResis.coeff(i);
			}

			std::string filename = "DiffRho_LocationCalc_" + std::to_string(loop) + "_" + std::to_string(locationCalcSettings->numOfSplit+ip) + ".vtk";
			output->VTKFileOputput(&invertedRhoIDToElementVector, &diffRho, filename);
			filename = "Rho_LocationCalc_" + std::to_string(loop) + "_" + std::to_string(locationCalcSettings->numOfSplit+ip) + ".txt";
			output->TxtOutputResistivity(&elements, filename);
			f << ip << "," << ratio << "," << alpha << "," << postMisfit-initialMisfit << "," << -onePointObjValue / initialOnePointObjValue + 1.0 << "," << initialMisfit << "," <<postMisfit << "," << probMaxIncreaseEachObsPoints.maxCoeff() << endl;

			filename = "EachObsPointsProbIncrease_" + std::to_string(loop) + "_" + std::to_string(locationCalcSettings->numOfSplit + ip) + ".txt";
			output->TxtOutput(&obsPointElements, &probMaxIncreaseEachObsPoints, filename);

		}
		f.close();

	}

}







void Analysis::Analysis::SetLocationDataToElement() {

	locPointElements.reserve(locationData.size());// max size we should consider is  locationData.size().

	for (auto itr = elements.begin(); itr != elements.end(); itr++) {
		Element::Element* element = itr->second;
		bool isLocElement = false;
	
		if (element->property->type == Property::Property::AIR) {
			continue;
		}
		Eigen::Vector3i pos;
		//for (int i = 0; i < 6; i++) {
		pos.setZero();
		pos.coeffRef(2) = -1;
		
		int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
		if (!(element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false && elements[element->alreadyFoundNeighborID[ipos]]->isAirGroundBoundaryCell == true && element->property->type != Property::Property::AIR)) {
			continue;
		}

		Element::Element* tmpElement = element;
		Element::Element* locElement = element;

		bool upsideIsSea = true;
		while (true) {
			if (tmpElement->property->type == Property::Property::SEA) {
				tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1つ深いセルへ

			}
			else if (upsideIsSea) {
				upsideIsSea = false;
				tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1つ深いセルへ
			}
			else {
				locElement = tmpElement;
				break;
				//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1]; 
			}
		}
		//while (true) {
		//	bool isFoundElement = true;
		//	for (int i = 0; i < 3; i++) {
		//		for (int j = 0; j < 3; j++) {
		//			for (int k = 0; k < 3; k++) {
		//				ipos = i + 3 * j + 9 * k;
		//				if (tmpElement->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && tmpElement->neighborElements[ipos]->property->type == Property::Property::AIR) {
		//					isFoundElement = false;
		//				}
		//			}
		//		}
		//	}
		//	if (isFoundElement) {
		//		locElement = tmpElement;
		//		break;
		//	}
		//	else {
		//		tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1つ深いセルへ
		//		//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1]; 
		//	}
		//}


		for (int i = 0; i < locationData.size(); i++) {
			Eigen::Vector3d X;
			X.coeffRef(0) = locationData[i]->coord.coeff(0);
			X.coeffRef(1) = locationData[i]->coord.coeff(1);
			X.coeffRef(2) = locElement->centerCoord.coeff(2);

			if (locElement->CheckThePointInside2D(X,4)) {
				if (locationData[i]->isAlreadyFoundElement == false) {
					locationData[i]->isAlreadyFoundElement = true;
					locElement->locationData = locationData[i];
					locPointElements.push_back(locElement);
						
					if (locElement->boundary != "NOT_BOUNDARY") {
						std::cout << "ERROR:Location Data Location is in Boundary Cell." << std::endl;
						exit(1);
					}
				}
			}

		}
	}
	numOfLocationCalcElements = locPointElements.size();

	for (int i = 0; i < locationData.size(); i++) {
		if (locationData[i]->isAlreadyFoundElement == false) {
			std::cout << "Location Data has data out of range." << std::endl;
			std::cout << std::fixed;
			std::cout << "ID:" << i << endl;
			std::cout << "X:" << std::setprecision(5) << locationData[i]->coord.coeff(0) << endl;
			std::cout << "Y:" << std::setprecision(5) << locationData[i]->coord.coeff(1) << endl;
			exit(1);
		}
	}
}



double Analysis::Analysis::CalcMaxImpedanceDataComparedToVarience() {
	dataMisfit = 0.0;

	for (int i = 0; i < numOfObsPointElements; i++) {
		Element::Element* element = obsPointElements[i];
		if (element->isInversionImpedance == true) {
			for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
				Eigen::Matrix2cd Zcalc;


				Zcalc = element->Z[iOmega];

				
				for (int ii = 0; ii < 2; ii++) {
					for (int jj = 0; jj < 2; jj++) {
						std::complex<double> dZtmp = Zcalc.coeff(ii, jj);
						double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
						double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));

						if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0) {
							dZtmp.real(dZtmp.real() / epsReal);
						}
						else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
							dZtmp.real(0.0);
						}
						else {
							//そのまま
						}
						if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0) {
							dZtmp.imag(dZtmp.imag() / epsImag);
						}
						else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
							dZtmp.imag(0.0);
						}
						else {
							//そのまま
						}


						dataMisfit = std::max(dataMisfit,(dZtmp * conj(dZtmp)).real());

					}
				}
			}
		}
	}

		
	return dataMisfit;
}




Eigen::Vector3d Analysis::Analysis::CalcMaxDatafitChangeUsingNormalDistribution() {
	Eigen::Vector3d returnVal;
	returnVal.coeffRef(0) = 0.0;
	//Impedance Tensor
	for (int i = 0; i < numOfObsPointElements; i++) {
		Element::Element* element = obsPointElements[i];

		for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
			Eigen::Matrix2cd Zcalc;


			if (isInvertedDistortion) {
				Zcalc = element->distortionMatrix * element->Zpos[iOmega];

			}
			else {
				Zcalc = element->Zpos[iOmega];

			}
			Eigen::Matrix2d fNormalDistriburtionPos;
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					std::complex<double> dZtmp = Zcalc.coeff(ii, jj) - element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj);
					double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
					double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));

					if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0) {
						dZtmp.real(dZtmp.real() / epsReal);
					}
					else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
						dZtmp.real(0.0);
					}
					else {
						//そのまま
					}
					if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0) {
						dZtmp.imag(dZtmp.imag() / epsImag);
					}
					else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
						dZtmp.imag(0.0);
					}
					else {
						//そのまま
					}

					fNormalDistriburtionPos.coeffRef(ii, jj) = 1.0 / sqrt(2.0 * ConstantValues::pi ) * std::exp(-0.5 * (dZtmp * conj(dZtmp)).real());
					//Varience is considered as average of epsReal and epsImag
				}
			}

			if (isInvertedDistortion) {
				Zcalc = element->distortionMatrix * element->Zpre[iOmega];

			}
			else {
				Zcalc = element->Zpre[iOmega];

			}
			Eigen::Matrix2d fNormalDistriburtionPre;
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					std::complex<double> dZtmp = Zcalc.coeff(ii, jj) - element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj);
					double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
					double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));

					if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0) {
						dZtmp.real(dZtmp.real() / epsReal);
					}
					else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
						dZtmp.real(0.0);
					}
					else {
						//そのまま
					}
					if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0) {
						dZtmp.imag(dZtmp.imag() / epsImag);
					}
					else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
						dZtmp.imag(0.0);
					}
					else {
						//そのまま
					}

					fNormalDistriburtionPre.coeffRef(ii, jj) = 1.0 / sqrt(2.0 * ConstantValues::pi ) * std::exp(-0.5 * (dZtmp * conj(dZtmp)).real());
					//Varience is considered as average of epsReal and epsImag

				}
			}

			Eigen::Matrix2d DiffNormalDistriburtion = fNormalDistriburtionPre - fNormalDistriburtionPos;
			if (returnVal.coeff(0) < DiffNormalDistriburtion.maxCoeff()) {
				Eigen::VectorXf::Index maxRow;
				Eigen::VectorXf::Index maxCol;
				returnVal.coeffRef(0) = DiffNormalDistriburtion.maxCoeff(&maxRow, &maxCol);
				returnVal.coeffRef(1) = fNormalDistriburtionPre.coeff(maxRow, maxCol);
				returnVal.coeffRef(2) = fNormalDistriburtionPos.coeff(maxRow, maxCol);

			}

		}



	}
	return returnVal;
}

void Analysis::Analysis::RunUncertaintyAnalysis() {
	//This Function is followed to PEST slides by Doherty 
	isDirectSolver = false;

	std::cout << "omp_get_max_threads:" << omp_get_max_threads() << std::endl;

	startCalc_t = time(NULL);
	std::cout << ("Initialize Data") << std::endl;
	Initialize();
	std::cout << ("Initialization End") << std::endl;

	output->VTKObsPointsFileOutput(&obsPointElements);
	//output->VTKFileOputput(0.0, &elements, "ObsPoints");

	initObjVal = 0.0;

	Eigen::VectorXd initialResis{ numOfInvertedResistivityElements };
	for (int i = 0; i < numOfInvertedResistivityElements; ++i) {
		initialResis.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity;
	}

	CalcSurfaceResistivityElements(); //Update Surface Resistivity
	time_t start_t = time(NULL);
	cout << "Update SumNCrossRhoRotHdS.." << endl;
	CalcSumNCrossRhoRotHdSElements(); //Update coeffs of Matrix
	time_t end_t = time(NULL);
	std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
	cout << "End Update SumNCrossRhoRotHdS.." << endl;

	lambdaDRDRho.setZero();
	CalcForward(true, false, false);

	double initialMisfit = CalcDataMisfit(false);

	double initialRMS = std::pow(initialMisfit / numOfObsData, 0.5);

	dDataMisfitDRho.setZero();
	CalcDDataMisfitDRho(false);
	weightRoughening = 0.0;
	CalcDJDRho(false);


	//variables are log10(Rho)
	Eigen::VectorXd dJDLogRhoOnlyResis{ numOfInvertedResistivityElements };
	dJDLogRhoOnlyResis.setZero();
	for (int i = 0; i < numOfInvertedResistivityElements; ++i) {
		dJDLogRhoOnlyResis.coeffRef(i) = dJdRho.coeff(i)* invertedRhoIDToElementVector[i]->resistivity *log(10.0);
		
	}

	Eigen::SparseMatrix<double, Eigen::RowMajor> priorCovMat{ numOfInvertedResistivityElements,numOfInvertedResistivityElements };
	priorCovMat.reserve(numOfInvertedResistivityElements);
	

	double objFuncCov = numOfObsData; //rms is 1, from definition
	double coeffForPosteriorCovMat = objFuncCov;
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		double ratio = 1.0 + uncertaintyAnalysis->priorModelStandardDeviation;
		double priorCov = log10(ratio)* log10(ratio);
		priorCovMat.coeffRef(i, i) = priorCov;
		coeffForPosteriorCovMat += dJDLogRhoOnlyResis.coeff(i) * priorCov * dJDLogRhoOnlyResis.coeff(i);
		
	}
	cout << objFuncCov << " " << coeffForPosteriorCovMat << endl;
	coeffForPosteriorCovMat = 1.0 / coeffForPosteriorCovMat;
	
	
	
	Eigen::MatrixXd CJt{ numOfInvertedResistivityElements,1 };
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		CJt.coeffRef(i,0) = priorCovMat.coeff(i,i) * dJDLogRhoOnlyResis.coeff(i); // because priorCovMat is diagonal matrix
	}

	Eigen::VectorXd CJtMaxNValues{ uncertaintyAnalysis->numOfValues };
	CJtMaxNValues.setZero();

	Eigen::VectorXi CJtMaxNValuesIndex{ uncertaintyAnalysis->numOfValues };
	CJtMaxNValuesIndex.setOnes();
	CJtMaxNValuesIndex = -CJtMaxNValuesIndex;

	Eigen::VectorXi alReadySetected{ numOfInvertedResistivityElements };
	alReadySetected.setZero();

	
	for (int i = 0; i < uncertaintyAnalysis->numOfValues; i++) {
		double maxval = 0;
		int k = 0;
		for (int j = 0; j < numOfInvertedResistivityElements; j++) {
			if (maxval < std::abs(CJt.coeff(j,0)) && alReadySetected.coeff(j) == 0) {
				maxval = std::abs(CJt.coeff(j,0));
				k = j;
			}
		}
		if (maxval != 0.0) {
			CJtMaxNValues.coeffRef(i) = CJt.coeff(k,0);
			CJtMaxNValuesIndex.coeffRef(i) = k;
			alReadySetected.coeffRef(k) = 1;
		}
	}

	Eigen::MatrixXd posteriorCovMat_largestN{ uncertaintyAnalysis->numOfValues,uncertaintyAnalysis->numOfValues };
	int size = numOfInvertedResistivityElements - uncertaintyAnalysis->numOfValues;
	Eigen::SparseMatrix<double> posteriorCovMat_other{ size ,size };
	posteriorCovMat_largestN.setZero();
	posteriorCovMat_other.reserve(Eigen::VectorXi::Constant(size, 1));

	for (int i = 0; i < uncertaintyAnalysis->numOfValues; i++) {
		for (int j = 0; j < uncertaintyAnalysis->numOfValues; j++) {
			double diag = 0;
			if (i == j) {
				diag = priorCovMat.coeff(CJtMaxNValuesIndex.coeff(i), CJtMaxNValuesIndex.coeff(i));
			}
			posteriorCovMat_largestN.coeffRef(i, j) = diag - coeffForPosteriorCovMat * CJtMaxNValues.coeff(i) * CJtMaxNValues.coeff(j);
		}
	}

	Eigen::VectorXi posteriorCovMat_otherIndex{ size };
	posteriorCovMat_otherIndex.setZero();
	int j = 0;
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		if (alReadySetected.coeff(i) == 0) {
			posteriorCovMat_other.coeffRef(j, j) = priorCovMat.coeff(i, i) - coeffForPosteriorCovMat * CJt.coeff(i,0)* CJt.coeff(i,0);
			posteriorCovMat_otherIndex.coeffRef(j) = i;
			j++;

		}
	}

	//SVD Decomposition, this is coresponded to Eigenvalue Decomposition because of posteriorCovMat_largestN is symmetric.
	cout << "Start Singular Decomposition..." << endl;

	Eigen::BDCSVD < Eigen::MatrixXd> svd;
	svd.compute(posteriorCovMat_largestN);
	Eigen::MatrixXd rmsMat = svd.singularValues().asDiagonal();
	int diagMatSize = rmsMat.cols();
	for (int i = 0; i < diagMatSize; i++) {
		rmsMat.coeffRef(i, i) = std::pow(rmsMat.coeff(i, i), 0.5);
	}

	std::ofstream fmat;
	fmat.open("DebugRMSMAT.txt", std::ios::trunc);
	rmsMat = svd.matrixU() * rmsMat;
	rmsMat = rmsMat * svd.matrixV().transpose();
	for (int i = 0; i < max(uncertaintyAnalysis->numOfValues,100); i++) {
		for (int j = 0; j < max(uncertaintyAnalysis->numOfValues, 100); j++) {
			fmat << i << " " << j << " " << rmsMat.coeff(i, j) << endl;
		}
	}
	fmat.close();

	cout << "Finished Start Singular Decomposition." << endl;

	//initial settings
	Eigen::MatrixXd resistivityForAllRandomVector{ uncertaintyAnalysis->numOfSamples,numOfInvertedResistivityElements };
	resistivityForAllRandomVector.setZero();
	Eigen::VectorXd posteriorRMSVector{ uncertaintyAnalysis->numOfSamples };
	posteriorRMSVector.setZero();
	std::ofstream f;
	f.open("RMSUncertaintyAnalysis.txt", std::ios::trunc);
	f << "initiallRMS:" << initialRMS << endl;
	f << "threshldDeltaRMS:" << uncertaintyAnalysis->thresHoldDeltaRMS << endl;
	//loop for random vectors
	for (int loop = 0; loop < uncertaintyAnalysis->numOfSamples; loop++) {
		cout << "Random Sampling... itetation#:" << loop << endl;
		//create random vector distriburted from posteriorCovMat_largestN
		std::random_device seed_gen;
		std::default_random_engine engine(seed_gen());

		std::normal_distribution<> dist(0.0, 1.0);

		Eigen::VectorXd randomVec{ uncertaintyAnalysis->numOfValues };
		for (int i = 0; i < uncertaintyAnalysis->numOfValues; i++) {
			randomVec.coeffRef(i) = dist(engine);
		}


		Eigen::VectorXd posteriorDeltaLogRho_largestN = rmsMat * randomVec;


		//create random vector distriburted from posteriorCovMat_otherIndex
		Eigen::VectorXd posteriorDeltaLogRho_other{ size };
		for (int i = 0; i < size; i++) {
			posteriorDeltaLogRho_other.coeffRef(i) = std::pow(posteriorCovMat_other.coeff(i,i),0.5) * dist(engine);
		}

		//set to resistivity 
		for (int i = 0; i < uncertaintyAnalysis->numOfValues; i++) {
			invertedRhoIDToElementVector[CJtMaxNValuesIndex.coeff(i)]->resistivity = initialResis.coeff(CJtMaxNValuesIndex.coeff(i)) *
				std::pow(10.0,posteriorDeltaLogRho_largestN.coeff(i));
			if (invertedRhoIDToElementVector[CJtMaxNValuesIndex.coeff(i)]->resistivity > maxResis) {
				invertedRhoIDToElementVector[CJtMaxNValuesIndex.coeff(i)]->resistivity = maxResis;
			}
			else if (invertedRhoIDToElementVector[CJtMaxNValuesIndex.coeff(i)]->resistivity < minResis) {
				invertedRhoIDToElementVector[CJtMaxNValuesIndex.coeff(i)]->resistivity = minResis;
			}
		}

		for (int i = 0; i < size; i++) {
			invertedRhoIDToElementVector[posteriorCovMat_otherIndex.coeff(i)]->resistivity = initialResis.coeff(posteriorCovMat_otherIndex.coeff(i)) *
				std::pow(10.0,posteriorDeltaLogRho_other.coeff(i));
			if (invertedRhoIDToElementVector[posteriorCovMat_otherIndex.coeff(i)]->resistivity > maxResis) {
				invertedRhoIDToElementVector[posteriorCovMat_otherIndex.coeff(i)]->resistivity = maxResis;
			}
			else if (invertedRhoIDToElementVector[posteriorCovMat_otherIndex.coeff(i)]->resistivity < minResis) {
				invertedRhoIDToElementVector[posteriorCovMat_otherIndex.coeff(i)]->resistivity = minResis;
			}
		}

		
		//Update Matrix
		initObjVal = 0.0;

		CalcSurfaceResistivityElements(); //Update Surface Resistivity
		time_t start_t = time(NULL);
		cout << "Update SumNCrossRhoRotHdS.." << endl;
		CalcSumNCrossRhoRotHdSElements(); //Update coeffs of Matrix
		time_t end_t = time(NULL);
		std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
		cout << "End Update SumNCrossRhoRotHdS.." << endl;

		lambdaDRDRho.setZero();
		
		//Calc Forward
		CalcForward(false); 

		//calc Data Misfit
		double d = CalcDataMisfit();

		double postRMS = std::pow(d / numOfObsData, 0.5);

		posteriorRMSVector.coeffRef(loop) = postRMS;
		for (int i = 0; i < numOfInvertedResistivityElements; i++) {
			resistivityForAllRandomVector.coeffRef(loop, i) = invertedRhoIDToElementVector[i]->resistivity;
		}
		
		//output
		f <<loop<<","<< postRMS  << endl;

		Eigen::VectorXd outputRho{ numOfInvertedResistivityElements };
		for (int i = 0; i < numOfInvertedResistivityElements; i++) {
			outputRho.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity;
		}
		std::string ori = "UncertaintyAnalysis_Rho_" + std::to_string(loop);
		std::string trueOrFalse;
		if (postRMS - initialRMS < uncertaintyAnalysis->thresHoldDeltaRMS) {
			trueOrFalse = "True";
		}
		else {
			trueOrFalse = "False";
		}
		std::string filename = ori + "_" + trueOrFalse+".vtk";
		output->RhoOutput(&elements, filename);
		filename = ori + "_" + trueOrFalse + ".txt";
		output->TxtOutputResistivity(&elements, filename);



		//output Max
		Eigen::VectorXd keepOriginalResistivity{ numOfCalcElements };
		for (int i = 0; i < numOfCalcElements; i++) {
			keepOriginalResistivity.coeffRef(i) = calcElementsVector[i]->resistivity;
			calcElementsVector[i]->resistivity = 0.0;
		}
		outputRho.setZero();
		for (int i = 0; i < numOfInvertedResistivityElements; i++) {
			double max = 0;
			for (int j = 0; j < loop + 1; j++) {	
				if (posteriorRMSVector.coeff(j) - initialRMS <= thresholdRMS) {
					double tmp = std::max(resistivityForAllRandomVector.coeff(j, i) - initialResis.coeff(i), invertedRhoIDToElementVector[i]->resistivity - initialResis.coeff(i));
					outputRho.coeffRef(i) = std::max(outputRho.coeffRef(i), tmp);
				}
			}
		}
		for (int i = 0; i < numOfInvertedResistivityElements; i++) {
			invertedRhoIDToElementVector[i]->resistivity = outputRho.coeffRef(i);
		}
		ori = "UncertaintyAnalysis_MaxDiffRho";
		filename = ori + ".vtk";
		output->RhoOutput(&elements, filename);
		filename = ori + ".txt";
		output->TxtOutputResistivity(&elements, filename);
		//Recover resis
		for (int i = 0; i < numOfCalcElements; i++) {
			calcElementsVector[i]->resistivity = keepOriginalResistivity.coeffRef(i);
		}

		//output Min
		for (int i = 0; i < numOfCalcElements; i++) {
			calcElementsVector[i]->resistivity = 0.0;
		}
		outputRho.setZero();
		for (int i = 0; i < numOfInvertedResistivityElements; i++) {
			double max = 0;
			for (int j = 0; j < loop + 1; j++) {
				if (posteriorRMSVector.coeff(j) - initialRMS <= thresholdRMS) {
					double tmp = std::min(resistivityForAllRandomVector.coeff(j, i) - initialResis.coeff(i), invertedRhoIDToElementVector[i]->resistivity - initialResis.coeff(i));
					outputRho.coeffRef(i) = std::min(outputRho.coeffRef(i), tmp);
				}
			}
		}
		for (int i = 0; i < numOfInvertedResistivityElements; i++) {
			invertedRhoIDToElementVector[i]->resistivity = outputRho.coeffRef(i);
		}
		ori = "UncertaintyAnalysis_MinDiffRho";
		filename = ori + ".vtk";
		output->RhoOutput(&elements, filename);
		filename = ori + ".txt";
		output->TxtOutputResistivity(&elements, filename);
		
		//Recover resis
		for (int i = 0; i < numOfCalcElements; i++) {
			calcElementsVector[i]->resistivity = keepOriginalResistivity.coeffRef(i);
		}

		//standard deviation
		for (int i = 0; i < numOfCalcElements; i++) {
			calcElementsVector[i]->resistivity = 0.0;
		}
		Eigen::VectorXd sd{ numOfInvertedResistivityElements };
		sd.setZero();
		int numOfAcceptedModel = 0;
		for (int j = 0; j < loop + 1; j++) {
			if (posteriorRMSVector.coeff(j) - initialRMS <= thresholdRMS) {
				numOfAcceptedModel++;
				for (int i = 0; i < numOfInvertedResistivityElements; i++) {
					sd.coeffRef(i) += std::pow(resistivityForAllRandomVector.coeff(j, i) - initialResis.coeff(i), 2.0);
				}
			}
		}
		if (numOfAcceptedModel > 0) {
			for (int i = 0; i < numOfInvertedResistivityElements; i++) {
				sd.coeffRef(i) = pow(sd.coeffRef(i) / double(numOfAcceptedModel), 0.5);
			}
			for (int i = 0; i < numOfInvertedResistivityElements; i++) {
				invertedRhoIDToElementVector[i]->resistivity = sd.coeffRef(i);
			}
			ori = "UncertaintyAnalysis_StandardDeviation";
			filename = ori + ".vtk";
			output->RhoOutput(&elements, filename);
			filename = ori + ".txt";
			output->TxtOutputResistivity(&elements, filename);
		}
		//Recover resis
		for (int i = 0; i < numOfCalcElements; i++) {
			calcElementsVector[i]->resistivity = keepOriginalResistivity.coeffRef(i);
		}


	}
	f.close();
	
}

void Analysis::Analysis::CalcNumOfReserveNeededInRow() {
	reservedVector.resize(3 * numOfCalcElements);
	reservedVector.setZero();
	for (int i = 0; i < numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];
		if (element->boundary == "NOT_BOUNDARY") {
			for (int k = 0; k < 3; k++) {
				std::map<int, int> dict;
				for (int j = 0; j < 6; j++) {
					for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(element->nCrossRotHdS[j], k); it; ++it) {
						int iCol = it.col();
						if (dict.find(iCol) == dict.end()) {
							dict[iCol] = 1;
							reservedVector[3 * i + k] += 1;
						}
						
					}
				}
			}
		}
		else {
			if (element->boundary == "-Z_BOUNDARY") {
				reservedVector[3 * i] += 1;
				reservedVector[3 * i + 1] += 1;
				reservedVector[3 * i + 2] += 1;
			}
			else {
				reservedVector[3 * i] += 2;
				reservedVector[3 * i + 1] += 2;
				reservedVector[3 * i + 2] += 2;
			}
		}
		//calcElementsVector[i]->CalcNumOfRelatedCalcVariables();
		//reservedVector[3 * i] = calcElementsVector[i]->numOfRelatedCalcVariables;
		//reservedVector[3 * i + 1] = calcElementsVector[i]->numOfRelatedCalcVariables;
		//reservedVector[3 * i + 2] = calcElementsVector[i]->numOfRelatedCalcVariables;
	}
	for (int i = 0; i < 3 * numOfCalcElements; i++) {
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(divergenceCorrection->sumDivHdSMatrix, i); it; ++it) {
			int iCol = it.col();
			Element::Element* element = calcElementsVector[int(i / 3)];
			bool isZero = true;
			for (int j = 0; j < 6; j++) {
				if (element->nCrossRotHdS[j].coeff(i % 3, it.col()) != 0.0) {
					isZero = false;
					break;
				}
			}
			if (isZero) {
				reservedVector[i] += 1;
			}
		}
	}
}


void Analysis::Analysis::CalcNumOfRelatedCalcVariablesElements() {
	reservedVector_rough.resize(3 * numOfCalcElements);
	reservedVector_rough.setZero();
	for (int i = 0; i < numOfCalcElements; i++) {
		calcElementsVector[i]->CalcNumOfRelatedCalcVariables();
		reservedVector_rough[3 * i] = calcElementsVector[i]->numOfRelatedCalcVariables;
		reservedVector_rough[3 * i + 1] = calcElementsVector[i]->numOfRelatedCalcVariables;
		reservedVector_rough[3 * i + 2] = calcElementsVector[i]->numOfRelatedCalcVariables;
	}

}

void Analysis::Analysis::CheckElements() {
	//Check Bounday
	for (int i = 0; i < numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];
		if (element->boundary != "NOT_BOUNDARY") {
			for (int j = 0; j < 27; j++) {
				Element::Element* neighborElem = element->neighborElements[j];
				if (neighborElem != NULL && (element->layer != neighborElem->layer || neighborElem->isParent)) {
					cout << "ERROR!!!!:Boundary Cells and its neighbors MUST be Same Layer!!!!" << endl;
					cout << "ElementID:" << element->ID << endl;
					cout << "Neighbor ElementID:" << neighborElem->ID << endl;
					exit(1);
				}
			}
		}
		else {
			//Check Limitation Of Layer Difference (must equal to zero or one)
			int layer1 = element->CalcDeepestChildElementLayer(&elements);
			for (int j = 0; j < 27; j++) {
				Element::Element* neighborElem = element->neighborElements[j];
				int layer2= neighborElem->CalcDeepestChildElementLayer(&elements);
				if (abs(layer1 - layer2) >= 2) {
					cout << "ERROR!!!!:Layer Difference Between Neighbor Cells Must be One Or Zero!!!!" << endl;
					cout << "ElementID:" << element->ID << endl;
					cout << "Neighbor ElementID:" << neighborElem->ID << endl;
					exit(1);
				}
			}
		}
	}

	
}

void Analysis::Analysis::SetCalcDivGradCells() {
	for (int i = 0; i < numOfCalcElements; i++) {
		
		Element::Element* element = calcElementsVector[i];

		/*bool flg = false;
		for (int j = 0; j < 27; j++) {
			Element::Element* neighborElem = calcElementsVector[i]->neighborElements[j];
			if (neighborElem != nullptr && element->property->type == Property::Property::AIR && neighborElem->isAirGroundBoundaryCell) {
				element->calcGradDivOperationElement = false;
				flg = true;
			}
		}
		
		if (flg) {
			continue;
		}*/

		if (element->property->type == Property::Property::AIR && !element->isAirGroundBoundaryCell) {
			element->calcGradDivOperationElement = true;
		}
		else {
			element->calcGradDivOperationElement = false;
		}
		//element->calcGradDivOperationElement = false;
	}
}

void Analysis::Analysis::CountNumOfAirCells() {
	numOfAirCells = 0;
	for (int i = 0; i < numOfCalcElements; i++) {
		if (calcElementsVector[i]->calcGradDivOperationElement) {
			numOfAirCells++;
		}
	}
	cout << "Num Of Air Cells:" << numOfAirCells << endl;
}

void Analysis::Analysis::SetSolverOrder() {
	originalOrderToSolverOrder.resize(3 * numOfCalcElements);
	solverOrderToOriginalOrder.resize(3 * numOfCalcElements);

	int countAir = 0;
	int countNonAir = 0;
	for (int i = 0; i < numOfCalcElements; i++) {
		if (calcElementsVector[i]->calcGradDivOperationElement) {
			originalOrderToSolverOrder[3 * i] = countAir;
			originalOrderToSolverOrder[3 * i + 1] = numOfCalcElements + countAir;
			originalOrderToSolverOrder[3 * i + 2] = 2*numOfCalcElements + countAir;

			solverOrderToOriginalOrder[countAir] = 3 * i;
			solverOrderToOriginalOrder[numOfCalcElements + countAir] = 3 * i + 1;
			solverOrderToOriginalOrder[2*numOfCalcElements + countAir] = 3 * i + 2;
			countAir++;
		}
		else {
			originalOrderToSolverOrder[3 * i] = numOfAirCells + countNonAir;
			originalOrderToSolverOrder[3 * i + 1] = numOfCalcElements + numOfAirCells + countNonAir;
			originalOrderToSolverOrder[3 * i + 2] = 2 * numOfCalcElements + numOfAirCells + countNonAir;

			solverOrderToOriginalOrder[numOfAirCells + countNonAir] = 3 * i;
			solverOrderToOriginalOrder[numOfCalcElements + numOfAirCells + countNonAir] = 3 * i + 1;
			solverOrderToOriginalOrder[2 * numOfCalcElements + numOfAirCells + countNonAir] = 3 * i + 2;
			countNonAir++;
		}

	}
}


//This function is followed to "3D magnetotelluric inversion using a limited-memory quasi - Newton optimization" Avdeev and Avdeeva (2009)
void Analysis::Analysis::ModifyGradient() {
	Eigen::VectorXd dJdRho_pos{ numOfInvertedResistivityElements };
		
	dJdRho_pos.setZero();

	double thres = 1e-8;
	int considerSigma = 8;
#pragma omp parallel for
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		Element::Element* element = invertedRhoIDToElementVector[i];

		if (needCalcCoeffForModifyGradient) {
			double sum = 0.0;
			vector<double> distVec;
			vector<Element::Element*>relatedElemVec;
			for (int ii = -considerSigma; ii < considerSigma + 1; ii++) {
				for (int jj = -considerSigma; jj < considerSigma + 1; jj++) {
					Element::Element* pairElement = element;
					for (int iii = 0; iii < abs(ii); iii++) {
						int ipos;
						if (ii < 0) {
							ipos = (-1 + 1) + 3 * (0 + 1) + 9 * (0 + 1);
						}
						else {
							ipos = (1 + 1) + 3 * (0 + 1) + 9 * (0 + 1);
						}
						pairElement = pairElement->neighborElements[ipos];
						if (pairElement == NULL) {
							break;
						}

					}
					if (pairElement == NULL) {
						continue;
					}

					for (int jjj = 0; jjj < abs(jj); jjj++) {
						int ipos;
						if (jj < 0) {
							ipos = (0 + 1) + 3 * (-1 + 1) + 9 * (0 + 1);
						}
						else {
							ipos = (0 + 1) + 3 * (1 + 1) + 9 * (0 + 1);
						}
						pairElement = pairElement->neighborElements[ipos];
						if (pairElement == NULL) {
							break;
						}

					}
					if (pairElement == NULL) {
						continue;
					}

					vector<Element::Element*> elemVec;
					pairElement->GetChildrenElements(&elements, elemVec);

					for (int iii = 0; iii < elemVec.size(); iii++) {
						Element::Element* targetElement = elemVec[iii];
						if (targetElement->invertedRhoElementsID < 0) {
							continue;
						}
						double dist2 = std::pow((element->centerCoord.coeff(0) - targetElement->centerCoord.coeff(0)) / minDx / invSettings->axForModifyGradient, 2.0)
							+ std::pow((element->centerCoord.coeff(1) - targetElement->centerCoord.coeff(1)) / minDy / invSettings->ayForModifyGradient, 2.0);
						if (std::exp(-0.5 * dist2) < thres) {
							continue;
						}
						sum += std::exp(-0.5 * dist2);
						distVec.push_back(std::exp(-0.5 * dist2));
						relatedElemVec.push_back(targetElement);

					}

				}
			}
			coeffsForModifiedGradient[i].resize(distVec.size());
			elementsForModifiedGradient[i].resize(distVec.size());
			for (int ii = 0; ii < coeffsForModifiedGradient[i].size(); ii++) {
				coeffsForModifiedGradient[i].coeffRef(ii) = distVec[ii] / sum;
				elementsForModifiedGradient[i][ii] = relatedElemVec[ii];
				
			}
			


		}
	}

	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		for (int ii = 0; ii < coeffsForModifiedGradient[i].size(); ii++) {
			dJdRho_pos.coeffRef(i) += coeffsForModifiedGradient[i].coeff(ii) * dJdRho.coeff(elementsForModifiedGradient[i][ii]->invertedRhoElementsID);
		}
	}

	/*for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		if (coeffsForModifiedGradient[i].size() == 0.0) {
			Element::Element* element = invertedRhoIDToElementVector[i];
			int ic = 0;
			for (int ii = max(0, element->IDX - considerSigma); ii < min(nx, element->IDX + considerSigma + 1); ii++) {
				for (int jj = max(0, element->IDY - considerSigma); jj < min(ny, element->IDY + considerSigma + 1); jj++) {
					for (int kk = 0; kk < sameIDXYZElements[ii][jj][element->IDZ].size(); kk++) {
						Element::Element* pairElement = sameIDXYZElements[ii][jj][element->IDZ][kk];
						if (pairElement->invertedRhoElementsID < 0) {
							continue;
						}
						ic++;
					}
				}
			}
			coeffsForModifiedGradient[i].resize(ic);
		}
	}
	
	

#pragma omp parallel for
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		
		
		Element::Element* element = invertedRhoIDToElementVector[i];
		
		if (needCalcCoeffForModifyGradient) {
			double sum = 0.0;
			int count = 0;
			vector<double> distVec;
			vector<Element::Element*>relatedElemVec;
			for (int ii = -considerSigma; ii < considerSigma + 1; ii++) {
				for (int jj = -considerSigma; jj < considerSigma + 1; jj++) {
					Element::Element* pairElement = element;
					for (int iii = 0; iii < abs(ii); iii++) {
						int ipos;
						if (ii < 0) {
							ipos=(-1 + 1) + 3 * (0 + 1) + 9 * (0 + 1);
						}
						else {
							ipos = (1 + 1) + 3 * (0 + 1) + 9 * (0 + 1);
						}
						pairElement = pairElement->neighborElements[ipos];
						if (pairElement == NULL) {
							break;
						}
						
					}
					if (pairElement == NULL) {
						continue;
					}

					for (int jjj = 0; jjj < abs(jj); jjj++) {
						int ipos;
						if (jj < 0) {
							ipos = (0 + 1) + 3 * (-1 + 1) + 9 * (0 + 1);
						}
						else {
							ipos = (0 + 1) + 3 * (1 + 1) + 9 * (0 + 1);
						}
						pairElement = pairElement->neighborElements[ipos];
						if (pairElement == NULL) {
							break;
						}

					}
					if (pairElement == NULL) {
						continue;
					}

					vector<Element::Element*> elemVec;
					pairElement->GetChildrenElements(&elements, elemVec);

					for (int iii = 0; iii < elemVec.size(); iii++) {
						Element::Element* targetElement = elemVec[iii];
						if (targetElement->invertedRhoElementsID < 0) {
							continue;
						}
						double dist2 = std::pow((element->centerCoord.coeff(0) - targetElement->centerCoord.coeff(0)) / minDx / invSettings->axForModifyGradient, 2.0)
							+ std::pow((element->centerCoord.coeff(1) - targetElement->centerCoord.coeff(1)) / minDy / invSettings->ayForModifyGradient, 2.0);
						sum += std::exp(-0.5 * dist2);
						count++;
						distVec.push_back(dist2);
						relatedElemVec.push_back(targetElement);

					}

				}
			}


			double sum = 0.0;
			for (int ii = max(0, element->IDX - considerSigma); ii < min(nx, element->IDX + considerSigma + 1); ii++) {
				for (int jj = max(0, element->IDY - considerSigma); jj < min(ny, element->IDY + considerSigma + 1); jj++) {
					for (int kk = 0; kk < sameIDXYZElements[ii][jj][element->IDZ].size(); kk++) {
						Element::Element* pairElement = sameIDXYZElements[ii][jj][element->IDZ][kk];
						if (pairElement->invertedRhoElementsID < 0) {
							continue;
						}
						double dist2 = std::pow((element->centerCoord.coeff(0) - pairElement->centerCoord.coeff(0)) / minDx / invSettings->axForModifyGradient, 2.0)
							+ std::pow((element->centerCoord.coeff(1) - pairElement->centerCoord.coeff(1)) / minDy / invSettings->ayForModifyGradient, 2.0);
						sum += std::exp(-0.5 * dist2);
					}
				}
			}
			int count = 0;
			for (int ii = max(0, element->IDX - considerSigma); ii < min(nx, element->IDX + considerSigma + 1); ii++) {
				for (int jj = max(0, element->IDY - considerSigma); jj < min(ny, element->IDY + considerSigma + 1); jj++) {
					for (int kk = 0; kk < sameIDXYZElements[ii][jj][element->IDZ].size(); kk++) {
						Element::Element* pairElement = sameIDXYZElements[ii][jj][element->IDZ][kk];
						if (pairElement->invertedRhoElementsID < 0) {
							continue;
						}
						double dist2 = std::pow((element->centerCoord.coeff(0) - pairElement->centerCoord.coeff(0)) / minDx / invSettings->axForModifyGradient, 2.0)
							+ std::pow((element->centerCoord.coeff(1) - pairElement->centerCoord.coeff(1)) / minDy / invSettings->ayForModifyGradient, 2.0);

						double coeff = std::exp(-0.5 * dist2) / sum;
						coeffsForModifiedGradient[i].coeffRef(count) = coeff;
						count++;
					}
				}
			}
		}
		int count = 0;
		for (int ii = max(0, element->IDX - considerSigma); ii < min(nx, element->IDX + considerSigma + 1); ii++) {
			for (int jj = max(0, element->IDY - considerSigma); jj < min(ny, element->IDY + considerSigma + 1); jj++) {
				for (int kk = 0; kk < sameIDXYZElements[ii][jj][element->IDZ].size(); kk++) {
					Element::Element* pairElement = sameIDXYZElements[ii][jj][element->IDZ][kk];
					if (pairElement->invertedRhoElementsID < 0) {
						continue;
					}
					dJdRho_pos.coeffRef(i) += coeffsForModifiedGradient[i].coeff(count) * dJdRho.coeff(pairElement->invertedRhoElementsID);
					count++;
				}
			}
		}

	}*/

	needCalcCoeffForModifyGradient = false;

//#pragma omp parallel for
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		dJdRho.coeffRef(i) = dJdRho_pos.coeff(i);
	}
	
	

}

//void Analysis::Analysis::SetMultiGridPreconditioner() {
//	/*int numOfBoundary = 0;
//	for (int i = 0; i < numOfCalcElements; i++) {
//		if (calcElementsVector[i]->boundary == "-Z_BOUNDARY" || calcElementsVector[i]->boundary == "+Z_BOUNDARY") {
//			numOfBoundary++;
//		}
//	}*/
//
//	
//	int thinning= iterativeSolverVector[0]->precond_multi.numOfThinning;
//	int nxCoarse = nx / thinning;
//	int nyCoarse = ny / thinning;
//	int nzCoarse = nz / thinning;
//
//	if (nxCoarse < 2 || nyCoarse < 2 || nzCoarse < 2) {
//		cout << "numOfThinning in Multi Grid Preconditioner must be more than or equal to 2." << endl;
//		exit(1);
//	}
//	int numOfTotalCoarseMeshes = 0;
//	std::unordered_map<int, Eigen::Vector3d> controlPointsAir;
//	std::unordered_map<int, Eigen::Vector3d> controlPointsGround;
//	int IDAir = 0;
//	int IDGround = 0;
//	for (int ii = 0; ii < nxCoarse; ii++) {
//		for (int jj = 0; jj < nyCoarse; jj++) {
//			bool foundAirCell=false;
//			bool foundGroundCell = false;
//			for (int kk = 0; kk < nzCoarse-1; kk++) {
//				Element::Element* element = basedElementsSortByNxNyNz[ii * thinning][jj * thinning][1 +kk * thinning];
//				if (element->property->type == Property::Property::AIR && !element->isAirGroundBoundaryCell) {
//					controlPointsAir[IDAir] = element->centerCoord;
//					IDAir++;
//				}
//				else {
//					controlPointsGround[IDGround] = element->centerCoord;
//					IDGround++;
//				}
//			}
//		}
//	}
//
//	int numOfAirCellsCoarse = IDAir;
//	int numOfGroundCellsCoarse = IDGround;
//	int numOfParamsCoarse = 3 * numOfAirCellsCoarse + 3 * numOfGroundCellsCoarse;
//
//	Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* W= new Eigen::SparseMatrix < complex<double>, Eigen::RowMajor>;
//	W->resize(3 * numOfCalcElements, numOfParamsCoarse);
//	W->reserve(Eigen::VectorXi::Constant(3 * numOfCalcElements, 27));
//	int ib = 0;
//	for (int j = 0; j < numOfCalcElements; j++) {
//		if (calcElementsVector[j]->boundary == "-Z_BOUNDARY" || calcElementsVector[j]->boundary == "+Z_BOUNDARY") {
//			continue;
//		}
//
//		Element::Element* element = calcElementsVector[j];
//		std::unordered_map<int, Eigen::Vector3d>* map;
//		int st;
//		if (element->property->type == Property::Property::AIR && !element->isAirGroundBoundaryCell) {
//			map = &controlPointsAir;
//			st = 0;
//		}
//		else {
//			map = &controlPointsGround;
//			st = 3 * numOfAirCellsCoarse;
//		}
//
//		double dist = 1e30;
//		int ID;
//		for (auto it = map->begin(); it != map->end(); it++)
//		{
//			if ((it->second - element->centerCoord).norm() < dist) {
//				dist = (it->second - element->centerCoord).norm();
//				ID = it->first;
//			}
//		}
//		for (int k = 0; k < 3; k++) {
//			int oriRow = 3 * j + k;
//			int oriCol = st + 3 * ID + k;
//
//			int solRow = originalOrderToSolverOrder[oriRow];
//			int solCol = oriCol;
//			W->coeffRef(solRow, solCol) = 1.0;
//		}
//	}
//	W->makeCompressed();
//	for (int i = 0; i < iterativeSolverVector.size(); i++) {
//		iterativeSolverVector[i]->precond_multi.W = W;
//		iterativeSolverVector[i]->useMultiGrid = true;
//	}
//}

Eigen::VectorXd Analysis::Analysis::CalcEachObservationProbIncrease() {
	Eigen::VectorXd probIncreaseEachObs;
	probIncreaseEachObs.resize(numOfObsPointElements);

	probIncreaseEachObs.setZero();
	//Impedance Tensor
	for (int i = 0; i < numOfObsPointElements; i++) {
		Element::Element* element = obsPointElements[i];

		for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
			Eigen::Matrix2cd Zcalc;


			if (isInvertedDistortion) {
				Zcalc = element->distortionMatrix * element->Zpos[iOmega];

			}
			else {
				Zcalc = element->Zpos[iOmega];

			}
			Eigen::Matrix2d fNormalDistriburtionPos;
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					std::complex<double> dZtmp = Zcalc.coeff(ii, jj) - element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj);
					double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
					double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));

					if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0) {
						dZtmp.real(dZtmp.real() / epsReal);
					}
					else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
						dZtmp.real(0.0);
					}
					else {
						//そのまま
					}
					if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0) {
						dZtmp.imag(dZtmp.imag() / epsImag);
					}
					else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
						dZtmp.imag(0.0);
					}
					else {
						//そのまま
					}

					fNormalDistriburtionPos.coeffRef(ii, jj) = 1.0 / sqrt(2.0 * ConstantValues::pi) * std::exp(-0.5 * (dZtmp * conj(dZtmp)).real());
					//Varience is considered as average of epsReal and epsImag
				}
			}

			if (isInvertedDistortion) {
				Zcalc = element->distortionMatrix * element->Zpre[iOmega];

			}
			else {
				Zcalc = element->Zpre[iOmega];

			}
			Eigen::Matrix2d fNormalDistriburtionPre;
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					std::complex<double> dZtmp = Zcalc.coeff(ii, jj) - element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj);
					double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
					double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));

					if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0) {
						dZtmp.real(dZtmp.real() / epsReal);
					}
					else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
						dZtmp.real(0.0);
					}
					else {
						//そのまま
					}
					if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0) {
						dZtmp.imag(dZtmp.imag() / epsImag);
					}
					else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
						dZtmp.imag(0.0);
					}
					else {
						//そのまま
					}

					fNormalDistriburtionPre.coeffRef(ii, jj) = 1.0 / sqrt(2.0 * ConstantValues::pi) * std::exp(-0.5 * (dZtmp * conj(dZtmp)).real());
					//Varience is considered as average of epsReal and epsImag

				}
			}
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					probIncreaseEachObs.coeffRef(i) = std::max(probIncreaseEachObs.coeff(i), fNormalDistriburtionPos.coeff(ii, jj) - fNormalDistriburtionPre.coeff(ii, jj));
				}
			}

		}



	}
	return probIncreaseEachObs;
}

void Analysis::Analysis::ReadInitialGuess(string filename, Eigen::VectorXcd& resultVector) {
	try {
		std::ifstream file(filename);
		if (!file) {
			std::cout << "Warning::No Initial Guess File!!!!" << std::endl;
			return;
		}

		std::string line;
		int current = 0;
		Eigen::VectorXcd readResult;
		readResult.resize(resultVector.size());
		readResult.setZero();
		while (std::getline(file, line)) {

			std::stringstream ss(line);
			std::string cell;
			std::vector<double> values;

			while (std::getline(ss, cell, ',')) {
				try {
					values.push_back(std::stod(cell));  // 数値に変換
				}
				catch (...) {
					std::cout << "Cannot Read Initial Guess From File!!!!!"  << std::endl;
					return;
				}
			}
			readResult.coeffRef(current) = std::complex<double>(values[0], values[1]);
			current++;
		}
		for (int i = 0; i < readResult.size(); i++) {
			resultVector.coeffRef(i) = readResult.coeff(i);
		}
	}
	catch (...) {
		std::cout << "Cannot Read Initial Guess From File!!!!!"  << std::endl;
		return;
	}
}

void Analysis::Analysis::SetLogScale() {


	for (int i = 0; i < numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];
		element->interpolateLogScale = false;
		if (useLogScaleInElement && element->property->type != Property::Property::AIR) {
			element->interpolateLogScale = true;
		}
		Eigen::Vector3i pos;
		//for (int i = 0; i < 6; i++) {
		pos.setZero();
		pos.coeffRef(2) = -1;
		int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
		if (!(element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false && elements[element->alreadyFoundNeighborID[ipos]]->isAirGroundBoundaryCell == true && element->property->type != Property::Property::AIR)) {
			continue;
		}
		element->notInterpolateLogScaleSurfaces.push_back(4); //In cells which share the surface with isAirGroundBoundaryCell, the interpolation on the surface should not be LogScale. 


	}
}