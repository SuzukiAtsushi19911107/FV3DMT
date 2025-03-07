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
#include "DivergenceCorrection.h"

#include "Element.h"
#include "FineGrainedILU.h"

//#include "MultiGridPreconditioner.h"
using namespace std;
namespace BiCGSafe {
	class BiCGSafe {
	public:
		BiCGSafe(int numOfSolsInp,int nInp) {
			
			n = nInp;
			numOfSols = numOfSolsInp;


			m_lastRelativeSolChangeVector.resize(numOfSols);
			m_errorVector.resize(numOfSols);

			m_maxIteration = 5000;
			m_relSolTol = 1e-8;
			m_relSolTolForAdjoint = 1e-8;
			m_l = 4;
			divergenceCorrection = nullptr;
			m_numIterCorrection = 500;
			m_minNumIterCorrection= 1000;
			r0star.resize(numOfSols);
			x0star.resize(numOfSols);
			r.resize(numOfSols);
			x0.resize(numOfSols);
			x_p.resize(numOfSols);
			v.resize(numOfSols);
			p.resize(numOfSols); 
			u.resize(numOfSols);
			y.resize(numOfSols);
			z.resize(numOfSols);
			Ap.resize(numOfSols);
			psiApPlusEtaY.resize(numOfSols);
			precondr.resize(numOfSols);
			r_p.resize(numOfSols);
			matKpr.resize(numOfSols);
			matu.resize(numOfSols);
			
			r0_sqnorm.resize(numOfSols);
			rhs_sqnorm.resize(numOfSols);
			alpha.resize(numOfSols);
			w.resize(numOfSols);
			relatedSolChange.resize(numOfSols);
			beta.resize(numOfSols);
			rho.resize(numOfSols);
			rho_old.resize(numOfSols);
			psi.resize(numOfSols);
			eta.resize(numOfSols);
			restart.resize(numOfSols);
			rReturn.resize(numOfSols);



			for (int i = 0; i < numOfSols; i++) {
				r0star[i].resize(n);
				r0star[i].setZero();
				x0star[i].resize(n);
				x0star[i].setZero();
				r[i].resize(n);
				r[i].setZero();
				x0[i].resize(n);
				x0[i].setZero();
				x_p[i].resize(n);
				x_p[i].setZero();
				v[i].resize(n);
				v[i].setZero();
				p[i].resize(n);
				p[i].setZero();
				u[i].resize(n);
				u[i].setZero();
				y[i].resize(n);
				y[i].setZero();
				z[i].resize(n);
				z[i].setZero();
				Ap[i].resize(n);
				Ap[i].setZero();
				precondr[i].resize(n);
				precondr[i].setZero();
				r_p[i].resize(n);
				r_p[i].setZero();
				matKpr[i].resize(n);
				matKpr[i].setZero();
				matu[i].resize(n);
				matu[i].setZero();
				psiApPlusEtaY[i].resize(n);
				psiApPlusEtaY[i].setZero();
				rReturn[i].resize(n);
				rReturn[i].setZero();
			}
		}
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		int m_maxIteration;
		int m_l;
		double m_lastRelativeSolChange;
		std::vector<double> m_lastRelativeSolChangeVector;
		int m_iters;
		double m_error;
		vector<double>  m_errorVector;
		double m_relSolTol;
		double m_relSolTolForAdjoint;
		int m_numIterCorrection;
		int m_minNumIterCorrection;
		//BlockILU::BlockILU precond;
		FineGrainedILU::FineGrainedILU precond;
		//ILU0::ILU0 precond;
		//Eigen::DiagonalPreconditioner<std::complex<double>> precond;
//		MultiGridPreconditioner::MultiGridPreconditioner precond_multi;
//		bool useMultiGrid = false;

		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd> > r0star;
		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> x0star;
		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> r;
		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> x0;
		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> x_p;
		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> v, p, u,psiApPlusEtaY;
		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> y, z;

		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> Ap;
		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> precondr;
		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> r_p;

		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> matKpr;
		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> matu;

		vector<double> r0_sqnorm;
		vector<double> rhs_sqnorm;
		vector<std::complex<double>> alpha;
		vector<std::complex<double>> w;
		vector<double> relatedSolChange;
		vector<std::complex<double>> beta;
		vector<std::complex<double>> rho;
		vector<std::complex<double>> rho_old;
		vector<std::complex<double>> psi;
		vector<std::complex<double>> eta;
		vector<bool> restart;

		vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> rReturn;

		int n;
		int numOfSols;

		double omega= -1;

		DivergenceCorrection::DivergenceCorrection* divergenceCorrection;
		bool solve(Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* mat, Eigen::VectorXcd* x,Eigen::VectorXcd* b,double tol,bool isDividedByDiagonal=false);
		bool solve(const Eigen::SparseMatrix< std::complex<double>, Eigen::RowMajor>* mat, vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* x,const vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* b, double tol, bool reuseILUByAdjoint = false);
		bool solve(const Eigen::SparseMatrix< std::complex<double>, Eigen::ColMajor>* mat, vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* x,const vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* b, double tol, bool reuseILUByAdjoint = false);

		bool solve(const Eigen::SparseMatrix< double, Eigen::RowMajor>* matR, const Eigen::SparseMatrix< std::complex<double>, Eigen::RowMajor>* matI, vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* x, const vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* b, double tol, bool reuseILUByAdjoint = false);


		bool solveWithSSOR(const Eigen::SparseMatrix< std::complex<double>, Eigen::RowMajor>* mat, vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* x, const vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* b, double tol, bool isDividedByDiagonal = false);
		bool solveWithSSOR(const Eigen::SparseMatrix< std::complex<double>, Eigen::ColMajor>* mat, vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* x, const vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* b, double tol, bool isDividedByDiagonal = false);

		void solveWithCorrection(Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* mat, Eigen::VectorXcd* x, Eigen::VectorXcd* b, double tol, int numIteration);
		void calculation(Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* mat, Eigen::VectorXcd* x, Eigen::VectorXcd* b, double tol, bool isDividedByDiagonal = false);
		std::complex<double> dot(Eigen::VectorXcd& a, Eigen::VectorXcd& b);
		std::vector<Element::Element*>* calcElementsVector;
		std::vector<int>* solverToOriginal;
		class JudgeComplex {
		public:
			double m_tol;
			JudgeComplex(double tol) {
				m_tol = tol;
			}
			bool operator() (const int& row, const int& col, const std::complex<double> value) const {
				if (value.real() < m_tol && value.imag() < m_tol) {
					return false;
				}
				else {
					return true;
				}
			}
		};
	};
}