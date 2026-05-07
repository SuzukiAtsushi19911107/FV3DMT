/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#pragma once
#define OPTIM_ENABLE_EIGEN_WRAPPERS
//#include "cg.hpp"
//#include "misc/optim_options.hpp"
#include "optim.hpp"
#include <iostream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Property.h"
#include "ReadData.h"
#include "Output.h"
#include "Element.h"
#include "InvSettings.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <kv/autodif.hpp>
#include <kv/complex.hpp>
#include <time.h>
#include "DivergenceCorrection.h"
#include "BiCGSafe.h"
#include "LocationData.h"
#include "UncertaintyAnalysis.h"
namespace ub = boost::numeric::ublas;



namespace Analysis {
	class Analysis {
	public:
		int nx = -1;
		int ny = -1;
		int nz = -1;

		//double minResis = log10(0.1); //This should be given as input
		//double maxResis = log10(1e4); //This should be given as input
		double minResis = 0.001; //This should be given as input
		double maxResis = 1e5; //This should be given as input
		double thresholdRMS = 1.;
		double RMScur = 1e30;
		double weightRoughening = -1;
		double weightRougheningForDistortion = 10;
		Analysis(ReadData::ReadData* readData);
		std::unordered_map<std::string, Element::Element*> elements;
		std::vector<Element::Element*> elementsVector;
		std::unordered_map<int, Property::Property*> properties;
		std::vector<Property::Property*> propertiesVector;
		Boundary::Boundary* boundary;
		InvSettings::InvSettings* invSettings;
		std::vector<ObsData::ObsData*> obsData;
		std::vector<Element::Element*> obsPointElements;
		std::vector<LocationData::LocationData*> locationData;
		std::vector<Element::Element*> locPointElements;
		UncertaintyAnalysis::UncertaintyAnalysis* uncertaintyAnalysis;
		int numOfObsPointElements;
		int numOfObsTensorComponents = 4; 
		std::vector<InitialResistivityData::InitialResistivityData*> initialResistivityData;
		std::vector<InitialDistortionData::InitialDistortionData*> initialDistortionData;
		Output::Output* output;
		Eigen::SparseMatrix<double , Eigen::RowMajor>* globalMatrixNoOmegaTerm;
		Eigen::SparseMatrix< double , Eigen::RowMajor>* globalMatrixNoOmegaTermAdjoint;
		vector<vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>> m_res;
		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> dJdH;
		vector<Eigen::VectorXcd> lambdaDRDRhoEachThread;

		//vector< Eigen::PardisoLU<Eigen::SparseMatrix<std::complex<double>>>*>solverVector;
		//vector< Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>> globalMatrixVector;
		//Eigen::VectorXcd*  globalVector;
		Eigen::SparseMatrix<std::complex< double >, Eigen::ColMajor>* globalVector;
		Eigen::SparseMatrix<complex<double> , Eigen::RowMajor>* dRdRho;
		Eigen::VectorXd dUdRho;
		Eigen::SparseMatrix<double , Eigen::RowMajor>* modelWeightMatrix;
		Eigen::VectorXd dRhoDParam;
		Eigen::MatrixXd dUdRho_output;
		Eigen::MatrixXd* jacobian;
		//vector<vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>> globalVectorEachThread;

		//vector<vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>> globalVectorAdjointEachThread;

		std::vector<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>>> globalMatrixForParallel; //for parallel
		std::vector < Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> globalVectorForParallel;

		vector<ub::matrix < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>> dZdHCalc;
		ub::matrix < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> dZdRhoCalc;

		Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* diagMatrixResistivity;

		Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* resisMatDotSumDivHdSMat;

		vector<int> numOfObsDataPerOmega;

		int maxLayer = 0;
		vector < vector < Element::Element* >> sameLayerElementsVector;

		Eigen::VectorXi reservedVector;
		Eigen::VectorXi reservedVector_rough;
		Eigen::VectorXi reservedVectorReordering;
		Eigen::VectorXi reservedVector_roughReordering;
		int numOfAirCells = 0;

		vector<int> originalOrderToSolverOrder;
		vector<int> solverOrderToOriginalOrder;
		double initObjVal=0;
		double dataMisfit;
		double mWTWm;
		vector<double> modelNormalizationCoeff{ 3 };
		int numOfObsData;
		int numOfImpedanceDataset;
		int numOfTipperDataset;
		bool isBelowRMSThreshold;
		double RMSpre = 1e30;
		double paramLogNormalization = 0.5;
		double limitOfparamLogNormalization = 10; //to prevent divergence
		double modelConstraintMax = 10;
		double modelConstraintMin = 0.001;
		int numOfCalcModelConstraint = 5;
		double objFuncChangeThresholdForNextmodelConstraint = 0.01;
		int maxIterationPerModelConstraint = 40;
		std::string optMethod = "GD";
		Eigen::VectorXd dDataMisfitDRho;
		Eigen::VectorXd dJdRho;
		vector < Eigen::VectorXd> coeffsForModifiedGradient;
		vector < vector<Element::Element*>> elementsForModifiedGradient;
		Eigen::VectorXd dJExceptRoughnessDRho;
		//Eigen::VectorXcd resultVector_pre;
		//Eigen::VectorXcd resultAdjointVector_pre;
		Eigen::VectorXcd resultVector;
		Eigen::VectorXcd resultAdjointVector;
		Eigen::VectorXd useImpedanceDataArray;
		Eigen::VectorXd useTipperDataArray;
		//Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>* solver1;
		//Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>* solver2;
		vector<Eigen::VectorXcd*> lambdaEachOmega;
		bool isInitializedSolver1 = false;
		bool isInitializedSolver2 = false;
		double omega;
		int numOfCalcElements;
		int numOfInvertedResistivityElements;
		int numOfDirichletConditionCells;
		int numOfIndependentInvertedResisElements;
		std::unordered_map<std::string, Element::Element*> calcElements;
		std::vector<Element::Element*> calcElementsVector;
		std::vector<Element::Element*> outputSensitivityVector;
		std::unordered_map<int, Element::Element*> invertedRhoIDToElementMap;
		std::vector<Element::Element*> invertedRhoIDToElementVector;
		Eigen::SparseMatrix<double,Eigen::RowMajor>* rougheningMatrix;
		optim::algo_settings_t settings;
		double obj_valPre=1e30;
		double obj_valNotNormalized = 0.0;
		std::vector<Element::Element*> notBoundaryElements;
		Eigen::VectorXcd lambdaDRDRho;
		time_t startCalc_t = time(NULL);
		struct CalcLambdaDRDRhoParameters {
			vector<int>startIDVector;
			vector<int>endIDVector;
			vector<Eigen::VectorXcd> lambdaDRDRhoEachThread;
			bool isInitialized = false;
			vector<vector<int>>threadIDGroup;
			vector < vector < Element::Element* >> sameLayerElementsVector;
			int maxLayer;
		};
		Eigen::VectorXd preParams;
		
		std::ofstream infofile;

		
		bool isDirectSolver = false;

		bool isAlreadyMadeMatrix = false;

		bool useLogScaleInElement = false;

		bool inheritPreviousObjVal = false;
		int numOfSameModelWeightCalc = 1;
		bool isFirstLoopInheritPreviousObjVal = false;

		bool isFirstLambdaAndLoop = true;

		//BiCGSafe::BiCGSafe* iterativeSolver;


		DivergenceCorrection::DivergenceCorrection* divergenceCorrection;
		vector<vector<vector<Element::Element*>>> basedElementsSortByNxNyNz;

		vector<BiCGSafe::BiCGSafe*> iterativeSolverVector;

		int numOfObsImpedanceElements = 0;

		int numOfParameters;

		bool isInvertedDistortion = true;

		vector<Element::Element*> obsImpedanceElements;

		Eigen::VectorXd dDataMisfitDDistortionParam;

		double constraintDistortionTerm = 0.0;

		bool alreadySetGlobalMatrixNonOmegaTerm = false;

		bool useL1Norm = false;
		double rateL1Norm = 1.0;
		double epsForL1Norm = 1e-3;

		LocationCalcSettings::LocationCalcSettings* locationCalcSettings;
		int numOfLocationCalcElements=0;
		Element::Element* locElement;

		vector<vector<Element::Element*>> sameIDZElements;
		unordered_map<int, unordered_map<int, unordered_map<int,vector<Element::Element*>>>> sameIDXYZElements;
		double minDx;
		double minDy;
		double minDz;

		bool needCalcCoeffForModifyGradient = true;

		bool calcJustDataMisfit = false;


		//for FFTSensitivityAnalysis
		double attenuation = 0.1;
		int Nx = 51;
		int Ny = 51;
		int Nz = 51;
		int Kx = 25;
		int Ky = 25;
		int Kz = 25;
		int cells_window = 3;
		int cells_window_fft_x = -1;
		int cells_window_fft_y = -1;
		int cells_window_fft_z = -1;
		int cells_window_out_x = -1;
		int cells_window_out_y = -1;
		int cells_window_out_z = -1;
		int numEnsemble = 100;
		int corr_cells_x = -1;
		int corr_cells_y = -1;
		int corr_cells_z = -1;

		double minX = -10000;
		double maxX = 10000;
		double minY = -10000;
		double maxY = 10000;
		double minZ = -10000;
		double maxZ = 10000;

		double epsR = 0.1;
		double epsT = 0.0;
		double eps_window = 0.;
		double eps_window_fft_x = -1.0;
		double eps_window_fft_y = -1.0;
		double eps_window_fft_z = -1.0;
		double fd_eps = 1.0e-1;
		double null_sv_ratio_thresh = 1.0e-2;
		int max_null_directions = 5;
		int min_null_directions = 3;
		int max_redraw_trials = 100;
		int candidate_pool_target = -1;
		double max_allowed_selected_cosine = 0.7071067811865476;
		bool first_direction_random = false;
		double confidenceLevel1 = 0.05; //For line search to seek the solution within these values
		double confidenceLevel2 = 0.01;
		double deltaRMSLevel1 = -1.0;
		double deltaRMSLevel2 = -1.0;
		double initWidth = 0.1;
		double lambdaForFFT = -1.0;
		string replacedResistivityFile = "";
		bool usePreviousResult = false;
		Eigen::VectorXcd resultVector_init;
		string orthogonalize = "objectiveFunction";
		bool FFTSensitivityMode = false;

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		//std::vector< Eigen::SparseMatrix < std::complex< double >, Eigen::ColMajor>>jacobiH;
		//Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> dZdH;
		//Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> dZdRho;
		int Hpolarization = 0;
		void SetLogScale();
		void ClearHAndE();
		void ClearZ();
		void RunAnalysis();
		void Initialize();
		void MakeMatrix(bool isRebuildMatrix = true, bool isCalcInversionValues=true);
		void solve(int iOmega,int itr);
		void Solve_iterative(int iOmega,int threadID);
		void SetH(int iOmega);
		void CalcE(int itr,int iOmega);
		void CalcZ(int iOmega);
		void CalcT(int iOmega);
		void AssociationPropertiesToElements();
		void SetLayerOfElements();
		void SetNeighborElements();
		void SetNotBoundaryElements();
		void SetNumOfCalcElementsAndCalcElementsAndElementsVector();
		void CalcSumNCrossRhoRotHdSElements();
		void SetTransitionZoneElements();
		void CalcNumOfDirichletConditionCells();
		void CalcSurfaceResistivityElements();
		void SetObsDataToElement();
		void CalcLambda(int iOmega,int threadID=0,bool onePointMode=false);
		void SearchRelatedCalcElements();
		void CalcLambdaDRDRho(const ub::vector<complex<double>>* rhoVec, const vector<Eigen::VectorXcd>* HresultItr,int iOmega);
		void SetInvertedElements();
		void CalcDDataMisfitDRho(bool onePointMode = false);
		void CalcDUDRho();
		double CalcDataMisfit(bool onePointMode=false);
		double CalcMaxImpedanceDataComparedToVarience();
		Eigen::Vector3d CalcMaxDatafitChangeUsingNormalDistribution();
		void SetDKDRhoElements();
		void CalcDKDRhoElements();
		void CalcForward(bool isCalcInversionValues = true, bool isCalcJacobiMatrix=false,bool onePointMode=false);
		void CalcDiffSmoothing();
		void CalcDZDHElements(const ub::vector<kv::complex<double>>* HVecUb,int iOmega);
		void CalcDZDRhoElements(const ub::vector<kv::complex<double>>* rhoVecUb , const ub::vector<kv::complex<double>>*, const int iOmega);
		void CalcDTDHElements(int iOmega);
		void CalcDJDRho(bool convertParamMode=true);
		void CalcRougheningMatrix();
		double CalcRoughningMatrixPenalty();
		double Optimize(const Eigen::VectorXd& vals_inp,Eigen::VectorXd* grad_out, void* opt_data);
		Eigen::Vector2d  OptimizeUsingJacobian(const Eigen::VectorXd& vals_inp, Eigen::MatrixXd* jac_out);
		void RunOptimize();
		void CountObsData();
		void SetSameResistivityToBoundaryCell();
		bool CalcRhoFromParamAndDRhoDParam(Eigen::VectorXd paramVec);
		Eigen::VectorXd CalcParamFromRho();
		Element::Element* SearchMasterElement(Element::Element* slaveElement);
		void CountIndependentInvertedResisElements();
		void CalcJacobian(int iOmega);
		void SetInitialResistivityFromFile();
		void SetInitialDistortionFromFile();
		void CalcDivergenceElements();
		vector<vector<vector<complex<double>>>> Calc1D();
		void CalcDDataMisfitDDistortionParam();
		void CalcConstraintDistortionTerm();
		void SetSameLayerElements();
		void SetOutput();
		void CalcNumOfReserveNeededInRow();
		void CalcNumOfRelatedCalcVariablesElements();

		void RunLocationCalc();
		void SetLocationDataToElement();

		void RunUncertaintyAnalysis();

		void CheckElements();

		void SetCalcDivGradCells();
		void CountNumOfAirCells();
		void SetSolverOrder();

		void ModifyGradient();

		void SetMultiGridPreconditioner();

		Eigen::VectorXd CalcEachObservationProbIncrease();

		void ReadInitialGuess(string file, Eigen::VectorXcd& resultVector);

		
		//for FFTSensitivityAnalysis
		void RunFFTSensitivityAnalysis();
		double  RunFowardCalc(std::vector<double> x, bool isCalcGradient = false);
		vector<double>   RunFowardCalcForJacobian(std::vector<double> x);
		vector<double> CalcDataMisfitEachData();

	};
}