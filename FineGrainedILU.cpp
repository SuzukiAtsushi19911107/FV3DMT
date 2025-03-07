/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#pragma once
#include "optim.hpp"
#include <iostream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <time.h>
#include "FineGrainedILU.h"
#include <time.h>
/*
This follows a paper below
FINE-GRAINED PARALLEL INCOMPLETE LU FACTORIZATION∗
EDMOND CHOW AND AFTAB PATEL
SIAM J. SCI. COMPUT. 2015 Society for Industrial and Applied Mathematics
Vol. 37, No. 2, pp. C169–C193
*/
using namespace std;
void FineGrainedILU::FineGrainedILU::compute(const Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>& mat) {
	time_t start_t = time(NULL);
	int n = mat.rows();
	int nc = n / 3;
	if (!isInitialized) {
		vector<vector<Eigen::Triplet<complex<double>>>> LsTrip;
		vector<vector<Eigen::Triplet<complex<double>>>> UsTrip;
		LsTrip.resize(3);
		UsTrip.resize(3);
		for (int i = 0; i < 3; i++) {
			Ls[i].resize(nc, nc);
			Us[i].resize(nc, nc);
		}
		for (int i = 0; i < n; i++) {
			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it) {
				int j = it.col();
				if (i < nc && j < nc) {

					if (i > j) {
						Eigen::Triplet <complex<double>> val(i, j, mat.coeff(i, j));
						LsTrip[0].push_back(val);
					}
					else {
						Eigen::Triplet <complex<double>> val(i, j, mat.coeff(i, j));
						UsTrip[0].push_back(val);
					}
					if (i == j) {
						Eigen::Triplet <complex<double>> val(i, j, 1.0);
						LsTrip[0].push_back(val);
					}
				}
				else if (i >= nc && i < 2 * nc && j >= nc && j < 2 * nc) {
					if (i > j) {
						Eigen::Triplet <complex<double>> val(i - nc, j - nc, mat.coeff(i, j));
						LsTrip[1].push_back(val);
					}
					else {
						Eigen::Triplet <complex<double>> val(i - nc, j - nc, mat.coeff(i, j));
						UsTrip[1].push_back(val);
					}
					if (i == j) {
						Eigen::Triplet <complex<double>> val(i - nc, i - nc, 1.0);
						LsTrip[1].push_back(val);
					}

				}
				else if (i >= 2 * nc && i < 3 * nc && j >= 2 * nc && j < 3 * nc) {
					if (i > j) {
						Eigen::Triplet <complex<double>> val(i - 2 * nc, j - 2 * nc, mat.coeff(i, j));
						LsTrip[2].push_back(val);
					}
					else {
						Eigen::Triplet <complex<double>> val(i - 2 * nc, j - 2 * nc, mat.coeff(i, j));
						UsTrip[2].push_back(val);
					}
					if (i == j) {
						Eigen::Triplet <complex<double>> val(i - 2 * nc, i - 2 * nc, 1.0);
						LsTrip[2].push_back(val);
					}
				}
			}
		}


		for (int i = 0; i < 3; i++) {
			Ls[i].setFromTriplets(LsTrip[i].begin(), LsTrip[i].end());
			Us[i].setFromTriplets(UsTrip[i].begin(), UsTrip[i].end());
			Ls[i].pruned(0.0);
			Ls[i].makeCompressed();
			Us[i].pruned(0.0);
			Us[i].makeCompressed();
		}

		for (int ii = 0; ii < 3; ii++) {

			sparsePatterns[ii].resize(nc, nc);
			sparsePatterns[ii].reserve(Eigen::VectorXi::Constant(nc, 100));
			for (int i = 0; i < nc; i++) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(Ls[ii], i); it; ++it) {
					int j = it.col();
					if (Us[ii].coeff(j, i) != 0.0) {
						sparsePatterns[ii].coeffRef(i, j) = true;
					}

				}
			}
			sparsePatterns[ii].makeCompressed();
		}
		isInitialized = true;
	}


	for (int ic = 0; ic < 3; ic++) {
		for (int ii = 0; ii < sweep; ii++) {
			for (int ia = ic * nc; ia < (ic + 1) * nc; ia++) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(mat, ia); it; ++it) {
					int ja = it.col();
					int j = it.col() - nc * ic;
					int i = ia - nc * ic;
					if (j >= nc || j < 0) {
						continue;
					}
					if (i > j) {
						complex<double> sum = 0.0;
						/*Eigen::VectorXcd tmp{ 1 };
						tmp =(Ls[ic].block(i,0,1,j)*
							Us[ic].block(0,j,j, 1));
						sum = tmp.coeff(0);*/
						for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(sparsePatterns[ic], i); it; ++it) {
							int k = it.col();
							if (k >= j) {
								break;
							}
							sum += Ls[ic].coeff(i, k) * Us[ic].coeff(k, j);
						}

						Ls[ic].coeffRef(i, j) = (mat.coeff(ia, ja) - sum) / Us[ic].coeff(j, j);
					}
					else {
						complex<double> sum = 0.0;
						/*Eigen::VectorXcd tmp{ 1 };
						tmp = (Ls[ic].block(i, 0, 1, i) *
							Us[ic].block(0, j, i, 1));
						sum = tmp.coeff(0);*/
						for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(sparsePatterns[ic], i); it; ++it) {
							int k = it.col();
							if (k >= i) {
								break;
							}
							sum += Ls[ic].coeff(i, k) * Us[ic].coeff(k, j);
						}
						/*for (int k = 0; k < i; k++) {
							sum += Ls[ic].coeff(i, k) * Us[ic].coeff(k, j);
						}*/
						Us[ic].coeffRef(i, j) = mat.coeff(ia, ja) - sum;
					}
				}
			}
		}
	}
	time_t end_t = time(NULL);
	std::cout << "Calculation Time for ILU Decomposition:" << end_t - start_t << " Seconds." << endl;
}
void FineGrainedILU::FineGrainedILU::compute(const Eigen::SparseMatrix<double, Eigen::RowMajor>& matR, const Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>& matI) {
	time_t start_t = time(NULL);
	int n = matR.rows();
	int nc = n / 3;
    if (!isInitialized) {
		vector<vector<Eigen::Triplet<complex<double>>>> LsTrip;
		vector<vector<Eigen::Triplet<complex<double>>>> UsTrip;
		LsTrip.resize(3);
		UsTrip.resize(3);
		for (int i = 0; i < 3; i++) {
			Ls[i].resize(nc, nc);
			Us[i].resize(nc, nc);
		}
		for (int i = 0; i < n; i++) {
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(matR, i); it; ++it) {
				int j = it.col();
				if (i < nc && j < nc) {

					if (i > j) {
						Eigen::Triplet <complex<double>> val(i, j, matR.coeff(i, j));
						LsTrip[0].push_back(val);
					}
					else {
						Eigen::Triplet <complex<double>> val(i, j, matR.coeff(i, j) + matI.coeff(i, j));
						UsTrip[0].push_back(val);
					}
					if (i == j) {
						Eigen::Triplet <complex<double>> val(i, j, 1.0);
						LsTrip[0].push_back(val);
					}
				}
				else if (i >= nc && i < 2 * nc && j >= nc && j < 2 * nc) {
					if (i > j) {
						Eigen::Triplet <complex<double>> val(i - nc, j - nc, matR.coeff(i, j));
						LsTrip[1].push_back(val);
					}
					else {
						Eigen::Triplet <complex<double>> val(i - nc, j - nc, matR.coeff(i, j) + matI.coeff(i, j));
						UsTrip[1].push_back(val);
					}
					if (i == j) {
						Eigen::Triplet <complex<double>> val(i - nc, i - nc, 1.0);
						LsTrip[1].push_back(val);
					}

				}
				else if (i >= 2 * nc && i < 3 * nc && j >= 2 * nc && j < 3 * nc) {
					if (i > j) {
						Eigen::Triplet <complex<double>> val(i - 2 * nc, j - 2 * nc, matR.coeff(i, j));
						LsTrip[2].push_back(val);
					}
					else {
						Eigen::Triplet <complex<double>> val(i - 2 * nc, j - 2 * nc, matR.coeff(i, j) + matI.coeff(i, j));
						UsTrip[2].push_back(val);
					}
					if (i == j) {
						Eigen::Triplet <complex<double>> val(i - 2 * nc, i - 2 * nc, 1.0);
						LsTrip[2].push_back(val);
					}
				}
			}
		}


		for (int i = 0; i < 3; i++) {
			Ls[i].setFromTriplets(LsTrip[i].begin(), LsTrip[i].end());
			Us[i].setFromTriplets(UsTrip[i].begin(), UsTrip[i].end());
			Ls[i].pruned(0.0);
			Ls[i].makeCompressed();
			Us[i].pruned(0.0);
			Us[i].makeCompressed();
		}

		for (int ii = 0; ii < 3; ii++) {

			sparsePatterns[ii].resize(nc, nc);
			sparsePatterns[ii].reserve(Eigen::VectorXi::Constant(nc, 100));
			for (int i = 0; i < nc; i++) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(Ls[ii], i); it; ++it) {
					int j = it.col();
					if (Us[ii].coeff(j, i) != 0.0) {
						sparsePatterns[ii].coeffRef(i, j) = true;
					}

				}
			}
			sparsePatterns[ii].makeCompressed();
		}
		isInitialized = true;
	}


	for (int ic = 0; ic < 3; ic++) {
		for (int ii = 0; ii < sweep; ii++) {
			for (int ia = ic * nc; ia < (ic + 1) * nc; ia++) {
				for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(matR, ia); it; ++it) {
					int ja = it.col();
					int j = it.col() - nc * ic;
					int i = ia - nc * ic;
					if (j >= nc || j < 0) {
						continue;
					}
					if (i > j) {
						complex<double> sum = 0.0;
						/*Eigen::VectorXcd tmp{ 1 };
						tmp =(Ls[ic].block(i,0,1,j)*
							Us[ic].block(0,j,j, 1));
						sum = tmp.coeff(0);*/
						for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(sparsePatterns[ic], i); it; ++it) {
							int k = it.col();
							if (k >= j) {
								break;
							}
							sum += Ls[ic].coeff(i, k) * Us[ic].coeff(k, j);
						}

						Ls[ic].coeffRef(i, j) = (matR.coeff(ia, ja) + matI.coeff(ia, ja) - sum) / Us[ic].coeff(j, j);
					}
					else {
						complex<double> sum = 0.0;
						/*Eigen::VectorXcd tmp{ 1 };
						tmp = (Ls[ic].block(i, 0, 1, i) *
							Us[ic].block(0, j, i, 1));
						sum = tmp.coeff(0);*/
						for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(sparsePatterns[ic], i); it; ++it) {
							int k = it.col();
							if (k >= i) {
								break;
							}
							sum += Ls[ic].coeff(i, k) * Us[ic].coeff(k, j);
						}
						/*for (int k = 0; k < i; k++) {
							sum += Ls[ic].coeff(i, k) * Us[ic].coeff(k, j);
						}*/
						Us[ic].coeffRef(i, j) = matR.coeff(ia, ja) + matI.coeff(ia, ja) - sum;
					}
				}
			}
		}
	}
	time_t end_t = time(NULL);
	std::cout << "Calculation Time for ILU Decomposition:" << end_t - start_t << " Seconds." << endl;
}
void FineGrainedILU::FineGrainedILU::solve(Eigen::VectorXcd& b, Eigen::VectorXcd& res) {
	int nc = res.size() / 3;
	
	for (int ic = 0; ic < 3; ic++) {
		//Ly=b
		for (int i = 0; i < nc; i++) {
			complex<double> sum = 0.0;
			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(Ls[ic], i); it; ++it) {
				int j = it.col();
				if (j >= i) { continue; }
				sum += res.coeff(ic * nc + j) * Ls[ic].coeff(i, j);
			}

			res.coeffRef(ic * nc + i) = (b.coeff(ic * nc + i) - sum) / Ls[ic].coeff(i, i);

			
		}

		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmp = Us[ic];
		for (int i = nc-1; i >= 0; i--) {
			complex<double> sum = 0.0;
			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(tmp, i); it; ++it) {
				int j = it.col();
				if (j <= i) { continue; }
				sum += res.coeff(ic * nc + j) * tmp.coeff(i, j);
			}
			res.coeffRef(ic * nc + i) = (res.coeff(ic * nc + i) - sum) / tmp.coeff(i, i);

			
		}
	}
}

