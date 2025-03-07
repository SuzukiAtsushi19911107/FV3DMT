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
#include "Property.h"
#include "ConstantValues.h"
#include "ObsData.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <kv/autodif.hpp>
#include <kv/complex.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include "LocationData.h"
#include "Element.h"
#include "Node.h"
namespace ub = boost::numeric::ublas;
using namespace std;

namespace UnstructuredElement {
	class UnstructuredElement : public Element::Element
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		UnstructuredElement();
		void CalcSurfaceResistivity(unordered_map<string, Element*>* elements, vector<Element*>* calcElementsVector, int numOfCalcElements);
		void CalcE(Eigen::SparseMatrix<std::complex< double >, Eigen::ColMajor>* Hresult, unordered_map<string, Element*>* elements, int numOfCalcElements, int itr);
		void CalcE(Eigen::VectorXcd* Hresult, unordered_map<string, Element*>* elements, int numOfCalcElements, int itr);
		void CalcSumNCrossRhoRotHdS(unordered_map<string, Element*>* elements, int numOfCalcElements);
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> CalcRotHdS(unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::Vector3i pos);
		void CalcNearestNeighborVectorEdge(Eigen::Vector3d x0, unordered_map<string, Element*>* elements, int numChildElements, Eigen::SparseMatrix<double, Eigen::RowMajor>* row, double* distance, int numOfPoints);
		Eigen::Vector3cd CalcDEDH(int derID, int numOfCalcElements, unordered_map<string, Element*>* elements);
		ub::matrix<kv::autodif<kv::complex<double>>> CalcDEDH(ub::vector<kv::autodif<kv::complex<double>>>* HresultTwoItr, std::vector<int>nonZeroRowIndices, unordered_map<string, Element*>* elements, int numOfCalcElements);
		ub::matrix<kv::autodif<kv::complex<double>>> CalcDEDRho(ub::vector<kv::autodif<kv::complex<double>>>* rhoVec, ub::vector<kv::complex<double>>* HresultTwoItr, std::vector<int>nonZeroRowIndices, int numOfCalcElements);
		void CalcDivHdS(Eigen::Vector3i pos, unordered_map<string, Element*>*elements, int numOfCalcElements, Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>&divHdS);
		void CalcSumNDivHdS(unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* sumNDivHdS);
		
		void  CalcOperatorSurface(Eigen::Vector3i pos, unordered_map<string, Element*>* elements, int numOfCalcElements,
			vector<Eigen::Triplet<double>>& divGradDsOperator, int& locDivGrad,
			vector<Eigen::Triplet<double>>& divDsOperator, int& locDiv, vector<Eigen::Triplet<double>>& gradOperator, int& locgrad);
		void  CalcOperator(unordered_map<string, Element*>* elements, int numOfCalcElements, vector<Eigen::Triplet<double>>& sumNGradOperatorDs, int& locDivGrad,
			vector<Eigen::Triplet<double>>& divDsOperator, int& locDiv, vector<Eigen::Triplet<double>>& gradOperator, int& locgrad);
		void CalcSumNDotHdSOperator(int numOfCalcElements, Eigen::SparseMatrix<double, Eigen::RowMajor>& sumNDotHdsOperator);
		void CalcNDotHdSOperator(Eigen::Vector3i pos, int numOfCalcElements, Eigen::SparseMatrix<double, Eigen::RowMajor>& sumNDotHdsOperator);
		void CalcInterpolatedVectorInElement(Eigen::Vector3i val, Eigen::Vector3d x0, unordered_map < string, Element* >* elements, int numChildElements, Eigen::SparseMatrix<double, Eigen::RowMajor>& row);

	};
}