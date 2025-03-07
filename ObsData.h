/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#pragma once
#include <iostream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
namespace ObsData {
	class ObsData {
	public:
		ObsData();
		string name = "";
		int ID;
		vector<int> rowIDsJacobian;
		vector<int> rowIDEachOmega;
		Eigen::Vector2d coord;
		vector<Eigen::Matrix2cd> ZobsVector;
		bool isAlreadyFoundElementImpedance = false;
		vector<Eigen::Matrix2d> varianceZobsVectorReal;
		vector<Eigen::Matrix2d> varianceZobsVectorImag;
		bool isImpedanceData = false;
		int impedanceID;

		vector<Eigen::Vector2cd> TobsVector;
		bool isAlreadyFoundElementTipper = false;
		vector<Eigen::Vector2d> varianceTobsVectorReal;
		vector<Eigen::Vector2d> varianceTobsVectorImag;
		bool isTipperData = false;
		int tipperID;
	};
}