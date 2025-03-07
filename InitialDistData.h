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
namespace InitialDistortionData {
	class InitialDistortionData {
	public:
		InitialDistortionData();
		int ID;
		Eigen::Vector3d coord;
		Eigen::Matrix2d distortionMatrix;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};
}
