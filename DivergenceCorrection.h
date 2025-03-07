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
#include <time.h>
#include <unordered_map>  
#include "Element.h"
using namespace std;
namespace DivergenceCorrection {
	class DivergenceCorrection {
	public:
		DivergenceCorrection(int n, Eigen::VectorXi reservedVector) {
			isDirectSolver = true;
			isAlreadyFactorized = false;
			Eigen::VectorXi rvSmallRow{ n };
			Eigen::VectorXi rvSmallCol{ 3 * n };
			Eigen::VectorXi rvSmallRowCol{ n };
			for (int i = 0; i < n; i++) {
				rvSmallRow.coeffRef(i) = max(1, reservedVector.coeff(i));

				rvSmallCol.coeffRef(3 * i) = max(1, int(reservedVector.coeff(i) / 3));
				rvSmallCol.coeffRef(3 * i + 1) = max(1, int(reservedVector.coeff(i) / 3));
				rvSmallCol.coeffRef(3 * i + 2) = max(1, int(reservedVector.coeff(i) / 3));

				rvSmallRowCol.coeffRef(i) = max(1, int(reservedVector.coeff(i) / 3));
			}


			divGradMatrix.resize(n, n);
			divGradMatrix.reserve(rvSmallRowCol);

			//divergenceOperatorMatrix.resize(n, 3 * n);
			//divergenceOperatorMatrix.reserve(rvSmallRow);
			//gradOperatorMatrix.resize(3 * n, n);
			//gradOperatorMatrix.reserve(rvSmallCol);
			sumDivHdSMatrix.resize(3 * n,3* n);
			sumDivHdSMatrix.reserve(reservedVector);
			size = n;
		}
		void factorize();
		void initialize(unordered_map<string, Element::Element*>* elements, vector<Element::Element*>* calcElementsVector);
		Eigen::VectorXcd correction(Eigen::VectorXcd& H);
		Eigen::SparseMatrix<double, Eigen::RowMajor> divGradMatrix;
		Eigen::SparseMatrix<double, Eigen::RowMajor> divergenceOperatorMatrix;
		Eigen::SparseMatrix<double, Eigen::RowMajor> sumDivHdSMatrix;
		Eigen::SparseMatrix<double, Eigen::RowMajor> gradOperatorMatrix;
		double thresholdKeepVal = 0.0001;
		bool isDirectSolver;
		bool isAlreadyFactorized;
		int size;
		vector<Element::Element*>* m_calcElementsVector;
	};
}