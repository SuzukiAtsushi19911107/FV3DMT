/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#pragma once
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"
#include <iostream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
namespace InvSettings {
	class InvSettings {
	public:
		InvSettings(int numOfLambda);
		std::string impedanceFile="";
		std::string tipperFile="";
		double par_step_size = 0.1;
		double paramLogNormalization = 0.1;
		double limitOfparamLogNormalization = 10; //to prevent divergence
		double modelConstraintMax = 10;
		double modelConstraintMin = 0.001;
		int numOfCalcModelConstraint = 5;
		double objFuncChangeThresholdForNextmodelConstraint = 0.0;
		vector<double> objFuncChangeThresholdVector;
		int maxIterationPerModelConstraint = 40;
		std::string optMethod = "GD";


		double grad_err_tol = 0.0;
		double rel_sol_change_tol = 0.0;

		double minResis = 0.001; 
		double maxResis = 1e5; 

		double toleranceIterativeSolver = 1e-12;
		double toleranceIterativeSolverAdjoint = -1e12;
		int maxIterationBiCGSafe=20000;

		double thresholdResistivityChange = 0.0;
		vector<double> thresholdRelativeResistivityChangeVector;
		std::vector<double> lambdaVector;
		std::vector<double> grad_err_tolVector;
		std::vector<double> par_step_sizeVector;
		std::vector<int> maxIterationVector;

		std::string manualSettingFile = "None";

		double coeffForSearchStepSize = 0;

		bool inheritPreviousSettingAdam = true;

		bool isDirectSolver = false;

		double loosenFactor = 1.1;
		double decreaseFactor = 0.1;
		double minStep = 0.0001;
		int maxIterationLineSearch = 5;
		int outputInterval = 1;

		double automaticLambdaMin = 0.0001;
		double automaticLambdaMax = 100;
		double automaticLambdaFirstStepRatio = 0.1;
		double automaticLambdaFirstValue = 1;
		double automaticLambdaMaxChange = 0.01;
		int automaticLambdaMaxIteration = 100;
		bool isUseDistanceInModelConstraint = true;

		int minibatches = 1;

		double RMSSwitchingToGD = -1;
		double RMSSwitchingToLBFGS = -1;
		bool isInvertedDistortion = true;

		bool useL1Norm = false;
		double rateL1Norm = 1.0;

		int fillFactorForILU = 1;

		double safetyFactor = 0.01;

		double thresholdRMS = 1.0;


		// below ones are followed "3D magnetotelluric inversion using a limited-memory quasi - Newton optimization" Avdeev and Avdeeva (2009)
		bool modifyGradient = false;

		double axForModifyGradient = 1.0;

		double ayForModifyGradient = 1.0;
		int numWarmUp = 5;
		int minIterations = 10;
		int averageIterations = 5;
		int numTrunc = 5;

		bool useLogScaleInterpolation = false;

		std::string initialGuessFile = "None";
		std::string InitialGuessOutputFile = "None";

		double epsRatio = 0.1;
		double eps_modelConstraint = 0.;
		double eps_distortionConstraint = 0.;
		std::vector<std::string> split(std::string str);
		std::vector<std::string> readNext(std::ifstream* f);
		void ReadManualSettingData(optim::algo_settings_t *settings);
	};
}