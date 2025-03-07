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
namespace LocationCalcSettings {
	class LocationCalcSettings {
	public:
		bool isCalc = false;
		std::string locationFile ="";
		std::string resistivityFile = "";
		double widthImpedance = 0.001;
		int numOfSplit = 1;
		double ratioResistivity = 0.1;
	};
}