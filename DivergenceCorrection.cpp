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

using namespace std;
void DivergenceCorrection::DivergenceCorrection::factorize() {
	cout<<"Direct Solver in Non Intel MKL Version is not Implemented."<<endl;
		exit(1);	
	return;
}
Eigen::VectorXcd DivergenceCorrection::DivergenceCorrection::correction(Eigen::VectorXcd& H) {
	Eigen::VectorXd colVec(size);
	colVec.setZero();
	Eigen::VectorXcd res(3*size);
	res.setZero();
	cout<<"Direct Solver in Non Intel MKL Version is not Implemented."<<endl;
		exit(1);	
	return res;

}
void DivergenceCorrection::DivergenceCorrection::initialize(unordered_map<string, Element::Element*>* elements, vector<Element::Element*>* calcElementsVector) {

	cout << "     Assembling matrix for Divergence Correction..." << endl;

	m_calcElementsVector = calcElementsVector;

	
	vector<vector< Eigen::Triplet<double>>> divGradMatrixTripletEachThread(omp_get_max_threads());
	vector < vector< Eigen::Triplet<double>>> gradOperatorMatrixTripletEachThread(omp_get_max_threads());
	vector < vector< Eigen::Triplet<double>>> divergenceOperatorMatrixTripletEachThread(omp_get_max_threads());


	vector<int> locationPushBackForDivGrad(omp_get_max_threads());
	vector<int> locationPushBackForDiv(omp_get_max_threads());
	vector<int> locationPushBackForGrad(omp_get_max_threads());
	for (int i = 0; i < omp_get_max_threads(); i++) {
		// Reserve only: avoid default-constructing a huge number of Triplets.
		divGradMatrixTripletEachThread[i].reserve(int(m_calcElementsVector->size() * 1000 / omp_get_max_threads())); // ~1000 per row as a rough upper bound
		//gradOperatorMatrixTripletEachThread[i].resize(int(3 * m_calcElementsVector->size() * 1000 / omp_get_max_threads()));
		//divergenceOperatorMatrixTripletEachThread[i].resize(int(3 * m_calcElementsVector->size() * 1000 / omp_get_max_threads()));
		locationPushBackForDivGrad[i] = 0;
		locationPushBackForDiv[i] = 0;
		locationPushBackForGrad[i] = 0;
	}

#pragma omp parallel for
	for (int i = 0; i < calcElementsVector->size(); i++) {
		int threadID = omp_get_thread_num();
		int ipos;
		Element::Element* element = (*calcElementsVector)[i];

		if (element->boundary == "NOT_BOUNDARY") {

			//if (sameLayerElementsVector[iLayer][i]->boundary == "NOT_BOUNDARY" && sameLayerElementsVector[iLayer][i]->property->type == Property::Property::AIR) {
			//if (sameLayerElementsVector[iLayer][i]->boundary == "NOT_BOUNDARY" && !sameLayerElementsVector[iLayer][i]->isAirGroundBoundaryCell) {
			//Eigen::SparseMatrix<double, Eigen::RowMajor> divGradMatrixDsElements{ 1,size };
			//divGradMatrixDsElements.reserve(Eigen::VectorXi::Constant(1, element->numOfRelatedCalcVariables));
			//Eigen::SparseMatrix<double, Eigen::RowMajor> gradientDsOperator{ 3,size };
			//gradientDsOperator.reserve(Eigen::VectorXi::Constant(3, element->numOfRelatedCalcVariables));
			element->CalcOperator(elements, size, divGradMatrixTripletEachThread[threadID], locationPushBackForDivGrad[threadID],
				divergenceOperatorMatrixTripletEachThread[threadID], locationPushBackForDiv[threadID],
				gradOperatorMatrixTripletEachThread[threadID], locationPushBackForGrad[threadID]);


			//Eigen::SparseMatrix<double, Eigen::RowMajor> divergenceOperatorDsMatrixElements{ 1,3 * size };
			//divergenceOperatorDsMatrixElements.reserve(Eigen::VectorXi::Constant(1, 100));
			//sameLayerElementsVector[iLayer][i]->CalcSumNDotHdSOperator(size, divergenceOperatorDsMatrixElements);

			//prune
			//divGradMatrixDsElements.pruned();
			//gradientDsOperator.pruned();
			//divergenceOperatorDsMatrixElements.pruned();
			//divGradMatrixDsElements.makeCompressed();
			//gradientDsOperator.makeCompressed();
			//divergenceOperatorDsMatrixElements.makeCompressed();
			//divGradMatrixDsElements.data().squeeze();
			//gradientDsOperator.data().squeeze();
			//divergenceOperatorDsMatrixElements.data().squeeze();

			//double dv = element->dv;
			//for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(divGradMatrixDsElements, 0); it; ++it) {
			//	//divGradMatrix.insert(sameLayerElementsVector[iLayer][i]->calcID, it.col()) = divGradMatrixElements.coeff(0, it.col());

			//	Eigen::Triplet<double> val(element->calcID, it.col(), divGradMatrixDsElements.coeff(0, it.col()));
			//	divGradMatrixTripletEachThread[threadID].push_back(val);
			//}
			//for (int j = 0; j < 3; j++) {
			//	for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(gradientDsOperator, j); it; ++it) {
			//		//gradOperatorMatrix.insert(size*(j%3)+ sameLayerElementsVector[iLayer][i]->calcID, it.col()) = gradientOperator.coeff(j, it.col());

			//		Eigen::Triplet<double> val(3 * sameLayerElementsVector[iLayer][i]->calcID+j, it.col(), gradientDsOperator.coeff(j, it.col()));
			//		gradOperatorMatrixTripletEachThread[threadID].push_back(val);

			//	}
			//}
			//for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(divergenceOperatorDsMatrixElements, 0); it; ++it) {
			//	//divergenceOperatorMatrix.insert(sameLayerElementsVector[iLayer][i]->calcID, size*(it.col()%3)+int(it.col()/3)) = divergenceOperatorMatrixElements.coeff(0, it.col());

			//	Eigen::Triplet<double> val(sameLayerElementsVector[iLayer][i]->calcID,it.col(), divergenceOperatorDsMatrixElements.coeff(0, it.col()));
			//	divergenceOperatorMatrixTripletEachThread[threadID].push_back(val);
			//}
		}
		////Boundary Condition
		//else {
		//	Eigen::Triplet<double> val(sameLayerElementsVector[iLayer][i]->calcID, sameLayerElementsVector[iLayer][i]->calcID, 1.0);
		//	divGradMatrixTripletEachThread[threadID].push_back(val);
		//}


	}
	

	// assemble
	vector<int> numVal(3);
	numVal[0] = 0;
	numVal[1] = 0;
	numVal[2] = 0;
	for (int i = 0; i < omp_get_max_threads(); i++) {
		numVal[0] += static_cast<int>(divGradMatrixTripletEachThread[i].size());
		//numVal[1] += static_cast<int>(gradOperatorMatrixTripletEachThread[i].size());
		//numVal[2] += static_cast<int>(divergenceOperatorMatrixTripletEachThread[i].size());
	}
	vector< Eigen::Triplet<double>> divGradMatrixTriplet;
	divGradMatrixTriplet.reserve(numVal[0]);
	cout << "     Making matrix From Triplets..." << endl;

	for (int i = 0; i < omp_get_max_threads(); i++) {
		// size() already reflects the number of Triplets actually pushed.
		divGradMatrixTriplet.insert(divGradMatrixTriplet.end(),
			divGradMatrixTripletEachThread[i].begin(),
			divGradMatrixTripletEachThread[i].end());
		/*for (int j = 0; j < gradOperatorMatrixTripletEachThread[i].size(); j++) {
			gradOperatorMatrixTriplet[numVal[1]]=gradOperatorMatrixTripletEachThread[i][j];
			numVal[1]++;
		}
		for (int j = 0; j < divergenceOperatorMatrixTripletEachThread[i].size(); j++) {
			divergenceOperatorMatrixTriplet[numVal[2]]=divergenceOperatorMatrixTripletEachThread[i][j];
			numVal[2]++;
		}*/
	}

	//make matrix
	divGradMatrix.setFromTriplets(divGradMatrixTriplet.begin(), divGradMatrixTriplet.end());
	//gradOperatorMatrix.setFromTriplets(gradOperatorMatrixTriplet.begin(), gradOperatorMatrixTriplet.end());
	//divergenceOperatorMatrix.setFromTriplets(divergenceOperatorMatrixTriplet.begin(), divergenceOperatorMatrixTriplet.end());

	divGradMatrix.pruned();
	//gradOperatorMatrix.pruned();
	//divergenceOperatorMatrix.pruned();
}