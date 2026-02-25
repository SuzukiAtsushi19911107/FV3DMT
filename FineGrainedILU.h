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
#include <unordered_map>    



/*
This follows a paper below
FINE-GRAINED PARALLEL INCOMPLETE LU FACTORIZATION∗
EDMOND CHOW AND AFTAB PATEL
SIAM J. SCI. COMPUT. 2015 Society for Industrial and Applied Mathematics
Vol. 37, No. 2, pp. C169–C193
*/
using namespace std;
namespace FineGrainedILU {
	class FineGrainedILU  {
	public:
		FineGrainedILU() {
			Ls.resize(3);
			Us.resize(3);
			sparsePatterns.resize(3);
		}
		// Release internal memory buffers.
		// After calling this, compute() will rebuild patterns/L/U on demand.
		void release_memory(bool shrink_vectors = true);

		// Ensure internal buffers match the given matrix size.
		// If size changed, it releases memory and resets initialization state.
		void reset_if_size_mismatch(int n_rows);
		bool isInitialized = false;
		int ratioToOriginalMat = 1;
		int sweep = 10;
		vector<Eigen::SparseMatrix < complex<double>, Eigen::RowMajor>> sparsePatterns;
		vector<Eigen::SparseMatrix < complex<double>, Eigen::RowMajor>> Ls;
		vector<Eigen::SparseMatrix < complex<double>, Eigen::ColMajor>> Us;
		virtual void solve(Eigen::VectorXcd& b, Eigen::VectorXcd& res);
		virtual void compute( const Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>& mat);
		virtual void compute(const Eigen::SparseMatrix<double, Eigen::RowMajor>& matR, const Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>& matI);
	};
}