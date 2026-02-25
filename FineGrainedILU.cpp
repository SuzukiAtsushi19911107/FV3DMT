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
	reset_if_size_mismatch(mat.rows());
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
	reset_if_size_mismatch(matR.rows());
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
	//int nc = res.size() / 3;
	//
	//for (int ic = 0; ic < 3; ic++) {
	//	//Ly=b
	//	for (int i = 0; i < nc; i++) {
	//		complex<double> sum = 0.0;
	//		for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(Ls[ic], i); it; ++it) {
	//			int j = it.col();
	//			if (j >= i) { continue; }
	//			sum += res.coeff(ic * nc + j) * Ls[ic].coeff(i, j);
	//		}

	//		res.coeffRef(ic * nc + i) = (b.coeff(ic * nc + i) - sum) / Ls[ic].coeff(i, i);

	//		
	//	}

	//	Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmp = Us[ic];
	//	for (int i = nc-1; i >= 0; i--) {
	//		complex<double> sum = 0.0;
	//		for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(tmp, i); it; ++it) {
	//			int j = it.col();
	//			if (j <= i) { continue; }
	//			sum += res.coeff(ic * nc + j) * tmp.coeff(i, j);
	//		}
	//		res.coeffRef(ic * nc + i) = (res.coeff(ic * nc + i) - sum) / tmp.coeff(i, i);

	//		
	//	}
	//}

	const int nc = static_cast<int>(res.size()) / 3;

	for (int ic = 0; ic < 3; ++ic) {
		// 連続メモリをMap。bは読み取り専用でOK
		Eigen::Map<const Eigen::VectorXcd> rb(b.data() + ic * nc, nc);
		Eigen::Map<      Eigen::VectorXcd> rx(res.data() + ic * nc, nc);

		// L は RowMajor（行志向前進代入）
		const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>& L = Ls[ic];

		// --- 前進代入: L * y = b（L(ii)=1 前提）
		for (int i = 0; i < nc; ++i) {
			std::complex<double> sum = 0.0;
			for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(L, i); it; ++it) {
				const int j = it.col();
				if (j < i) sum += it.value() * rx[j];   // strict-lower だけ使う
			}
			rx[i] = rb[i] - sum; // L(ii)=1
		}

		// U は ColMajor（列志向後退代入）
		const Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor>& U = Us[ic];

		for (int j = nc - 1; j >= 0; --j) {
			// 対角取得（列j内）
			std::complex<double> diag = 0.0;
			for (Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor>::InnerIterator it(U, j); it; ++it) {
				if (it.row() == j) { diag = it.value(); break; }
			}
			// 安全策（必要に応じてエラー処理）
			// assert(std::abs(diag) > 0);
			rx[j] /= diag;  // 正規化

			// 列jの i<j 成分で上三角の寄与を一気に撒く
			for (Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor>::InnerIterator it(U, j); it; ++it) {
				const int i = it.row();
				if (i < j) rx[i] -= it.value() * rx[j];
			}
		}
	}
}
void FineGrainedILU::FineGrainedILU::release_memory(bool shrink_vectors)
{
    // Release Eigen sparse matrices completely (capacity included).
    for (int i = 0; i < 3; ++i) {
        {
            // Release RowMajor complex sparse matrix
            Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> empty;
            Ls[i].swap(empty);
        }
        {
            // Release ColMajor complex sparse matrix
            Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor> empty;
            Us[i].swap(empty);
        }
        {
            // Release sparsePatterns (currently stored as complex<double> RowMajor in your header)
            Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> empty;
            sparsePatterns[i].swap(empty);
        }
    }

    // Optionally shrink the std::vector containers themselves (minor, but deterministic).
    if (shrink_vectors) {
        std::vector<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>(Ls.begin(), Ls.end()).swap(Ls);
        std::vector<Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor>>(Us.begin(), Us.end()).swap(Us);
        std::vector<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>(sparsePatterns.begin(), sparsePatterns.end()).swap(sparsePatterns);
    }

    // Keep the vectors size=3 (constructor expects that).
    if ((int)Ls.size() != 3) Ls.resize(3);
    if ((int)Us.size() != 3) Us.resize(3);
    if ((int)sparsePatterns.size() != 3) sparsePatterns.resize(3);

    // Reset state.
    isInitialized = false;
}

void FineGrainedILU::FineGrainedILU::reset_if_size_mismatch(int n_rows)
{
    // Defensive: n_rows must be a multiple of 3 for your block split.
    if (n_rows <= 0 || (n_rows % 3) != 0) {
        // If invalid, just reset to a clean state.
        release_memory(false);
        return;
    }

    const int nc = n_rows / 3;

    if (!isInitialized) return;

    // If internal matrices do not match the current size, reinitialize.
    // Note: only checking Ls[0]/Us[0] is enough because all are set together.
    const bool size_ok =
        (Ls.size() == 3 && Us.size() == 3 && sparsePatterns.size() == 3) &&
        (Ls[0].rows() == nc && Ls[0].cols() == nc) &&
        (Us[0].rows() == nc && Us[0].cols() == nc) &&
        (sparsePatterns[0].rows() == nc && sparsePatterns[0].cols() == nc);

    if (!size_ok) {
        release_memory(false);
    }
}

