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
namespace UncertaintyAnalysis {
	class UncertaintyAnalysis {
	public:
		bool isCalc = false;
		double thresHoldDeltaRMS = 0.1;
		double priorModelStandardDeviation = 0.1;
		int numOfValues = 1000;
		int numOfSamples = 100;
	};
}