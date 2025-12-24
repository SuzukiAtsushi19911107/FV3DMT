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
#include "Element.h"
#include "Functions.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <kv/autodif.hpp>
#include <kv/complex.hpp>
#include "UnstructuredElement.h"

namespace ub = boost::numeric::ublas;
using namespace std;

UnstructuredElement::UnstructuredElement::UnstructuredElement() {
	numOfCalcSurfaceForE = 4;
}
void UnstructuredElement::UnstructuredElement::CalcSurfaceResistivity(unordered_map<string, Element*>* elements, vector<Element*>* calcElementsVector, int numOfCalcElements) {
	
	if (isAlreadyCalcResisCoeff==false) {
		
		resistivitySurface = new vector<double>(6);
		//for (int j = 0; j < 6; j++) {
		//	(*resistivitySurface)[j] = 0.0;
		//}

		resistivitySurfaceCoeff.resize(6);
		diffResistivitySurfaceCoeff.resize(6);
		for (int j = 0; j < 6; j++) {
			resistivitySurfaceCoeff[j] = new Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor >{1, numOfCalcElements };
			resistivitySurfaceCoeff[j]->reserve(Eigen::VectorXi::Constant(1, int(numOfRelatedCalcVariables/3)));
			diffResistivitySurfaceCoeff[j] = new Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor >{ 1, numOfCalcElements };
			diffResistivitySurfaceCoeff[j]->reserve(Eigen::VectorXi::Constant(1,int(numOfRelatedCalcVariables/3)));
			

		}

	}
	if (boundary == "NOT_BOUNDARY") {

		for (int isurf = 0; isurf < 6; isurf++) {
			(*resistivitySurface)[isurf] = 0.0;
			if (isAlreadyCalcResisCoeff == false) {

				Eigen::Vector3i pos;
				pos[0] = 0;
				pos[1] = 0;
				pos[2] = 0;
				if (isurf == 0) pos[0] = -1;
				else if (isurf == 1) pos[0] = 1;
				else if (isurf == 2) pos[1] = -1;
				else if (isurf == 3) pos[1] = 1;
				else if (isurf == 4) pos[2] = -1;
				else if (isurf == 5) pos[2] = 1;
				int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
				string neighborID = alreadyFoundNeighborID[ipos];
				//string neighborID = Functions::GetNeighborElement(*elements, this, pos);

				//First, Calc Diff

				Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> diffRhoCoeff{ 1,numOfCalcElements };
				//rhoCoeff.makeCompressed();
				//rhoCoeff.reserve(100);
				//harmonicOperationRhoCoeff.makeCompressed();
				//harmonicOperationRhoCoeff.reserve(100);
				diffRhoCoeff.reserve(Eigen::VectorXi::Constant(1, 100));
				Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>rho1Coeff{ 1,numOfCalcElements };
				Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>rho2Coeff{ 1,numOfCalcElements };
				rho1Coeff.reserve(Eigen::VectorXi::Constant(1, 100));
				rho2Coeff.reserve(Eigen::VectorXi::Constant(1, 100));
				rho1Coeff.setZero();
				rho2Coeff.setZero();
				if (layer == neighborElements[ipos]->layer) {
					CalcCenterCoeff(elements, numOfCalcElements, &rho1Coeff, 1.0);
					neighborElements[ipos]->CalcSurfaceRelatedResistivityCoeff(elements, numOfCalcElements, &rho2Coeff, 1.0, -1 * pos);
					Functions::SetAtoResultCoef1DotBPlusCoef2DotC(diffResistivitySurfaceCoeff[isurf], &rho1Coeff, &rho2Coeff, 1.0, -1.0);
				}
				else if (layer > neighborElements[ipos]->layer) {
					int j;
					if (isurf == 0) j = 1; //reverse
					else if (isurf == 1) j = 0;
					else if (isurf == 2) j = 3;
					else if (isurf == 3) j = 2;
					else if (isurf == 4) j = 5;
					else if (isurf == 5) j = 4;
					Functions::PlusEqual(diffResistivitySurfaceCoeff[isurf], neighborElements[ipos]->diffResistivitySurfaceCoeff[j], 1.0);
				}
				else{
					cout << "CalcSurfaceResistivity Must Be From Small Layer number to Large!!!" << endl;
					exit(1);
				}
				



				//Next, calc surface resis
				if (layer == neighborElements[ipos]->layer) {
					vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> edgeVal(4);
					//vector<Eigen::MatrixXd> edgeVal(4);
					for (int i = 0; i < 4; i++) {
						edgeVal[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 1,1 * numOfCalcElements };
						edgeVal[i].reserve(Eigen::VectorXi::Constant(1, 81));
						//edgeVal[i].resize(3, 3 * numOfCalcElements);
					}
					for (int i = 0; i < 4; i++) {
						edgeVal[i].setZero();
					}

					int surfaceNum;
					if (pos[0] == -1) {
						surfaceNum = 0;
					}
					else if (pos[0] == 1) {
						surfaceNum = 1;
					}
					else if (pos[1] == -1) {
						surfaceNum = 2;
					}
					else if (pos[1] == 1) {
						surfaceNum = 3;
					}
					else if (pos[2] == -1) {
						surfaceNum = 4;
					}
					else {
						surfaceNum = 5;
					}



					vector<Eigen::Vector3d> edgeCenters(4);
					for (int i = 0; i < 4; i++) { //Edge loop
						Eigen::Vector3d edgeCenter;
						bool possibilityTwoPoints = false;
						if (pos[0] == -1) {
							if (i == 0) { edgeCenter = (nodes[0]->x + nodes[3]->x) / 2.0; possibilityTwoPoints = true; }
							else if (i == 1) { edgeCenter = (nodes[3]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = false; }
							else if (i == 2) { edgeCenter = (nodes[4]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = true; }
							else if (i == 3) { edgeCenter = (nodes[0]->x + nodes[4]->x) / 2.0; possibilityTwoPoints = false; }
						}
						else if (pos[0] == 1) {
							if (i == 0) { edgeCenter = (nodes[1]->x + nodes[2]->x) / 2.0; possibilityTwoPoints = true; }
							else if (i == 1) { edgeCenter = (nodes[2]->x + nodes[6]->x) / 2.0; possibilityTwoPoints = false; }
							else if (i == 2) { edgeCenter = (nodes[6]->x + nodes[5]->x) / 2.0; possibilityTwoPoints = true; }
							else if (i == 3) { edgeCenter = (nodes[5]->x + nodes[1]->x) / 2.0; possibilityTwoPoints = false; }
						}
						else if (pos[1] == -1) {
							if (i == 0) { edgeCenter = (nodes[0]->x + nodes[1]->x) / 2.0; possibilityTwoPoints = true; }
							else if (i == 1) { edgeCenter = (nodes[1]->x + nodes[5]->x) / 2.0; possibilityTwoPoints = false; }
							else if (i == 2) { edgeCenter = (nodes[5]->x + nodes[4]->x) / 2.0; possibilityTwoPoints = true; }
							else if (i == 3) { edgeCenter = (nodes[4]->x + nodes[0]->x) / 2.0; possibilityTwoPoints = false; }
						}
						else if (pos[1] == 1) {
							if (i == 0) { edgeCenter = (nodes[3]->x + nodes[2]->x) / 2.0; possibilityTwoPoints = true; }
							else if (i == 1) { edgeCenter = (nodes[2]->x + nodes[6]->x) / 2.0; possibilityTwoPoints = false; }
							else if (i == 2) { edgeCenter = (nodes[6]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = true; }
							else if (i == 3) { edgeCenter = (nodes[7]->x + nodes[3]->x) / 2.0; possibilityTwoPoints = false; }
						}
						else if (pos[2] == -1) {
							if (i == 0) { edgeCenter = (nodes[0]->x + nodes[1]->x) / 2.0; possibilityTwoPoints = false; }
							else if (i == 1) { edgeCenter = (nodes[1]->x + nodes[2]->x) / 2.0; possibilityTwoPoints = false; }
							else if (i == 2) { edgeCenter = (nodes[2]->x + nodes[3]->x) / 2.0; possibilityTwoPoints = false; }
							else if (i == 3) { edgeCenter = (nodes[3]->x + nodes[0]->x) / 2.0; possibilityTwoPoints = false; }
						}
						else if (pos[2] == 1) {
							if (i == 0) { edgeCenter = (nodes[4]->x + nodes[5]->x) / 2.0; possibilityTwoPoints = false; }
							else if (i == 1) { edgeCenter = (nodes[5]->x + nodes[6]->x) / 2.0; possibilityTwoPoints = false; }
							else if (i == 2) { edgeCenter = (nodes[6]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = false; }
							else if (i == 3) { edgeCenter = (nodes[7]->x + nodes[4]->x) / 2.0; possibilityTwoPoints = false; }
						}
						edgeCenters[i] = edgeCenter;

						unordered_map<string, Element*>nodeElementsDict;
						nodeElementsDict.reserve(8);
						vector<Element*> nodeElements;
						Eigen::Vector3i tmp;

						for (int j = 0; j < 4; j++) {
							tmp.setZero();
							if (pos.coeffRef(0) == -1) {
								if (i == 0) { //edgeNode No.1
									if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(0) = -1;
									if (j == 2) { tmp.coeffRef(0) = -1; tmp.coeffRef(2) = -1; }
									if (j == 3) tmp.coeffRef(2) = -1;
								}
								else if (i == 1) { //edgeNode No.2
									if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(0) = -1;
									if (j == 2) { tmp.coeffRef(0) = -1; tmp.coeffRef(1) = +1; }
									if (j == 3) tmp.coeffRef(1) = +1;
								}
								else if (i == 2) { //edgeNode No.3
									if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(0) = -1;
									if (j == 2) { tmp.coeffRef(0) = -1; tmp.coeffRef(2) = +1; }
									if (j == 3) tmp.coeffRef(2) = +1;
								}
								else if (i == 3) { //edgeNode No.4
									if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(0) = -1;
									if (j == 2) { tmp.coeffRef(0) = -1; tmp.coeffRef(1) = -1; }
									if (j == 3) tmp.coeffRef(1) = -1;
								}
							}
							else if (pos.coeffRef(0) == 1) {
								if (i == 0) { //edgeNode No.1
									if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(0) = 1;
									if (j == 2) { tmp.coeffRef(0) = 1; tmp.coeffRef(2) = -1; }
									if (j == 3) tmp.coeffRef(2) = -1;
								}
								else if (i == 1) { //edgeNode No.2
									if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(0) = 1;
									if (j == 2) { tmp.coeffRef(0) = 1; tmp.coeffRef(1) = +1; }
									if (j == 3) tmp.coeffRef(1) = +1;
								}
								else if (i == 2) { //edgeNode No.3
									if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(0) = 1;
									if (j == 2) { tmp.coeffRef(0) = 1; tmp.coeffRef(2) = +1; }
									if (j == 3) tmp.coeffRef(2) = +1;
								}
								else if (i == 3) { //edgeNode No.4
									if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(0) = 1;
									if (j == 2) { tmp.coeffRef(0) = 1; tmp.coeffRef(1) = -1; }
									if (j == 3) tmp.coeffRef(1) = -1;
								}
							}
							else if (pos.coeffRef(1) == -1) {
								if (i == 0) { //edgeNode No.1
									if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(1) = -1;
									if (j == 2) { tmp.coeffRef(1) = -1; tmp.coeffRef(2) = -1; }
									if (j == 3) tmp.coeffRef(2) = -1;
								}
								else if (i == 1) { //edgeNode No.2
									if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(1) = -1;
									if (j == 2) { tmp.coeffRef(1) = -1; tmp.coeffRef(0) = 1; }
									if (j == 3) tmp.coeffRef(0) = 1;
								}
								else if (i == 2) { //edgeNode No.3
									if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(1) = -1;
									if (j == 2) { tmp.coeffRef(1) = -1; tmp.coeffRef(2) = 1; }
									if (j == 3) tmp.coeffRef(2) = 1;
								}
								else if (i == 3) { //edgeNode No.4
									if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(1) = -1;
									if (j == 2) { tmp.coeffRef(1) = -1; tmp.coeffRef(0) = -1; }
									if (j == 3) tmp.coeffRef(0) = -1;
								}
							}
							else if (pos.coeffRef(1) == 1) {
								if (i == 0) { //edgeNode No.1
									if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(1) = 1;
									if (j == 2) { tmp.coeffRef(1) = 1; tmp.coeffRef(2) = -1; }
									if (j == 3) tmp.coeffRef(2) = -1;
								}
								else if (i == 1) { //edgeNode No.2
									if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(1) = 1;
									if (j == 2) { tmp.coeffRef(1) = 1; tmp.coeffRef(0) = 1; }
									if (j == 3) tmp.coeffRef(0) = 1;
								}
								else if (i == 2) { //edgeNode No.3
									if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(1) = 1;
									if (j == 2) { tmp.coeffRef(1) = 1; tmp.coeffRef(2) = 1; }
									if (j == 3) tmp.coeffRef(2) = 1;
								}
								else if (i == 3) { //edgeNode No.4
									if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(1) = 1;
									if (j == 2) { tmp.coeffRef(1) = 1; tmp.coeffRef(0) = -1; }
									if (j == 3) tmp.coeffRef(0) = -1;
								}
							}
							else if (pos.coeffRef(2) == -1) {
								if (i == 0) { //edgeNode No.1
									if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(2) = -1;
									if (j == 2) { tmp.coeffRef(2) = -1; tmp.coeffRef(1) = -1; }
									if (j == 3) tmp.coeffRef(1) = -1;
								}
								else if (i == 1) { //edgeNode No.2
									if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(2) = -1;
									if (j == 2) { tmp.coeffRef(2) = -1; tmp.coeffRef(0) = 1; }
									if (j == 3) tmp.coeffRef(0) = 1;
								}
								else if (i == 2) { //edgeNode No.3
									if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(2) = -1;
									if (j == 2) { tmp.coeffRef(2) = -1; tmp.coeffRef(1) = 1; }
									if (j == 3) tmp.coeffRef(1) = 1;
								}
								else if (i == 3) { //edgeNode No.4
									if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(2) = -1;
									if (j == 2) { tmp.coeffRef(2) = -1; tmp.coeffRef(0) = -1; }
									if (j == 3) tmp.coeffRef(0) = -1;
								}
							}
							else if (pos.coeffRef(2) == 1) {
								if (i == 0) { //edgeNode No.1
									if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(2) = 1;
									if (j == 2) { tmp.coeffRef(2) = 1; tmp.coeffRef(1) = -1; }
									if (j == 3) tmp.coeffRef(1) = -1;
								}
								else if (i == 1) { //edgeNode No.2
									if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(2) = 1;
									if (j == 2) { tmp.coeffRef(2) = 1; tmp.coeffRef(0) = 1; }
									if (j == 3) tmp.coeffRef(0) = 1;
								}
								else if (i == 2) { //edgeNode No.3
									if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(2) = 1;
									if (j == 2) { tmp.coeffRef(2) = 1; tmp.coeffRef(1) = 1; }
									if (j == 3) tmp.coeffRef(1) = 1;
								}
								else if (i == 3) { //edgeNode No.4
									if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
									if (j == 1)tmp.coeffRef(2) = 1;
									if (j == 2) { tmp.coeffRef(2) = 1; tmp.coeffRef(0) = -1; }
									if (j == 3) tmp.coeffRef(0) = -1;
								}
							}
							int neighborVal;
							neighborVal = (tmp.coeff(0) + 1) + 3 * (tmp.coeff(1) + 1) + 9 * (tmp.coeff(2) + 1);
							string neighborIDtmp;

							neighborIDtmp = alreadyFoundNeighborID[neighborVal];
							if (nodeElementsDict.count(neighborIDtmp) == 0) {
								nodeElementsDict[neighborIDtmp] = neighborElements[neighborVal];
								nodeElements.push_back(neighborElements[neighborVal]);
							}
						}
						vector<double> weightList(nodeElements.size());
						vector< Eigen::SparseMatrix<double, Eigen::RowMajor>> vectorList(nodeElements.size());
						//vector< Eigen::MatrixXd> vectorList(nodeElements.size());
						Eigen::Vector3d x0;
						x0.setZero();
						x0 = edgeCenters[i];
						for (int k = 0; k < nodeElements.size(); k++) {
							double tmpdistance;
							vectorList[k] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 1, 1 * numOfCalcElements };
							vectorList[k].reserve(Eigen::VectorXi::Constant(1, 27));
							nodeElements[k]->CalcCenterCoeff(elements, numOfCalcElements, &vectorList[k], 1.0);
							tmpdistance = (nodeElements[k]->centerCoord - edgeCenters[i]).norm();
							vectorList[k].makeCompressed();
							weightList[k] = 1 / tmpdistance;
						}
						double wSum = 0;
						for (int k = 0; k < nodeElements.size(); k++) {
							wSum += weightList[k];
						}
						for (int k = 0; k < nodeElements.size(); k++) {
							Functions::PlusEqual(&edgeVal[i], &vectorList[k], weightList[k] / wSum);
						}
					}
					double sum = 0.0;
					for (int i = 0; i < 4; i++) {
						sum += 1.0/(surfaceCenters[isurf] - edgeCenters[i]).norm();
					}
					for (int i = 0; i < 4; i++) {
						Functions::PlusEqual(resistivitySurfaceCoeff[isurf], &edgeVal[i], 1.0/(surfaceCenters[isurf] - edgeCenters[i]).norm()/ sum);
					}
				}
				else if (layer > neighborElements[ipos]->layer) {
					int j;
					if      (isurf == 0) { j = 1; }
					else if (isurf == 1) { j = 0; }
					else if (isurf == 2) { j = 3; }
					else if (isurf == 3) { j = 2; }
					else if (isurf == 4) { j = 5; }
					else if (isurf == 5) { j = 4; }
					Functions::PlusEqual(resistivitySurfaceCoeff[isurf], neighborElements[ipos]->resistivitySurfaceCoeff[j], 1.0);


				}
				else {
					cout << "something wrong in calc of rho" << endl;
				}


				

				resistivitySurfaceCoeff[isurf]->makeCompressed();
				resistivitySurfaceCoeff[isurf]->data().squeeze();

				diffResistivitySurfaceCoeff[isurf]->makeCompressed();
				diffResistivitySurfaceCoeff[isurf]->data().squeeze();
			}
			(*resistivitySurface)[isurf] = 0.0;
			for (int j = 0; j < resistivitySurfaceCoeff[isurf]->outerSize(); ++j) {
				for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(*resistivitySurfaceCoeff[isurf], j); it; ++it)
				{
					int iRow = it.row();
					int iCol = it.col();
					Element*  element = (*calcElementsVector)[iCol];
					(*resistivitySurface)[isurf] += resistivitySurfaceCoeff[isurf]->coeff(0, iCol).real()*element->resistivity;
				}
			}
		}
		
	}
	isAlreadyCalcResisCoeff = true;
}


void UnstructuredElement::UnstructuredElement::CalcE(Eigen::SparseMatrix<std::complex< double >, Eigen::ColMajor>* Hresult, unordered_map<string, Element*> *elements,int numOfCalcElements,int itr) {
	if (boundary != "NOT_BOUNDARY") {
		return ;
	}
	Eigen::VectorXcd result{ 3 * numOfCalcElements };
	result = Eigen::VectorXcd(Hresult->col(0));

	Eigen::Vector3cd tmpE;
	tmpE.setZero();
	double sumDs = 0;//test
	double sumWeight = 0;
	Eigen::VectorXcd rhs{ 2 * numOfCalcSurfaceForE };
	rhs.setZero();
	Eigen::MatrixXcd A{ 2 * numOfCalcSurfaceForE , 3};
	A.setZero();
	Eigen::MatrixXcd W{ 2 * numOfCalcSurfaceForE, 2 * numOfCalcSurfaceForE };
	W.setZero();
	for (int i = 0; i < numOfCalcSurfaceForE; i++) {
		W.coeffRef(2 * i, 2 * i) = 1;// dSvector[i];
		W.coeffRef(2 * i + 1, 2 * i + 1) = 1;// dSvector[i];
		double dS = dSvector[i];
		double rho = 0.0;
		rho = (*resistivitySurface)[i];
		Eigen::Vector3d pararellUnitVec1;
		Eigen::Vector3d pararellUnitVec2;

		pararellUnitVec1 = surfaceParallelVectors[i][0];
		pararellUnitVec2 = surfaceParallelVectors[i][1];

		for (int j = 0; j < 3; j++) { //nVec_parallel.dot(E)
			A.coeffRef(2 * i, j) = pararellUnitVec1.coeff(j);
			A.coeffRef(2 * i + 1, j) = pararellUnitVec2.coeff(j);
		}
		rhs.coeffRef(2 * i) = rho*pararellUnitVec1.dot(rotHdS[i] * result / dSvector[i]);
		rhs.coeffRef(2 * i + 1) = rho*pararellUnitVec2.dot(rotHdS[i] * result / dSvector[i]);
	}
	tmpE = (A.adjoint() * W * A).inverse() * A.adjoint() * W * rhs;
	E[itr] = tmpE;
	return;

}

void UnstructuredElement::UnstructuredElement::CalcE(Eigen::VectorXcd* Hresult, unordered_map<string, Element*>* elements, int numOfCalcElements, int itr) {
	if (boundary != "NOT_BOUNDARY") {
		return;
	}


	Eigen::Vector3cd tmpE;
	tmpE.setZero();
	double sumDs = 0;//test
	double sumWeight = 0;
	Eigen::VectorXcd rhs{ 2 * numOfCalcSurfaceForE };
	rhs.setZero();
	Eigen::MatrixXcd A{ 2 * numOfCalcSurfaceForE , 3 };
	A.setZero();
	Eigen::MatrixXcd W{ 2 * numOfCalcSurfaceForE, 2 * numOfCalcSurfaceForE };
	W.setZero();
	for (int i = 0; i < numOfCalcSurfaceForE; i++) {
		W.coeffRef(2 * i, 2 * i) = 1;// dSvector[i];
		W.coeffRef(2 * i + 1, 2 * i + 1) = 1;// dSvector[i];
		double dS = dSvector[i];
		double rho = 0.0;
		rho = (*resistivitySurface)[i];
		Eigen::Vector3d pararellUnitVec1;
		Eigen::Vector3d pararellUnitVec2;
		pararellUnitVec1 = surfaceParallelVectors[i][0];
		pararellUnitVec2 = surfaceParallelVectors[i][1];

		for (int j = 0; j < 3; j++) { //nVec_parallel.dot(E)
			A.coeffRef(2 * i, j) = pararellUnitVec1.coeff(j);
			A.coeffRef(2 * i + 1, j) = pararellUnitVec2.coeff(j);
		}
		rhs.coeffRef(2 * i) = rho * pararellUnitVec1.dot(rotHdS[i] * (*Hresult)) / dSvector[i];
		rhs.coeffRef(2 * i + 1) = rho * pararellUnitVec2.dot(rotHdS[i] * (*Hresult)) / dSvector[i];
	}
	tmpE = (A.adjoint() * W * A).inverse() * A.adjoint() * W * rhs;
	E[itr] = tmpE;
	return;
}


void UnstructuredElement::UnstructuredElement::CalcSumNCrossRhoRotHdS(unordered_map<string, Element*> *elements, int numOfCalcElements) {

	if (isAlreadyCalcRotHdS == false) {
		sumNCrossRhoRotHdS = new Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor >{ 3,3 * numOfCalcElements };
		sumNCrossRhoRotHdS->reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
		//sumNCrossRhoRotHdS->uncompress();
		rotHdS.resize(6);
		nCrossRotHdS.resize(6);
		
		if (isAlreadyCalcRotHdS == false) {
			for (int i = 0; i < 6; i++) {
				rotHdS[i] = Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
				rotHdS[i].reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
				rotHdS[i].uncompress();

				nCrossRotHdS[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
				nCrossRotHdS[i].reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
				nCrossRotHdS[i].uncompress();

				


			}
		}
	}



	if (boundary != "NOT_BOUNDARY") {
		return;
	}



	for (int i = 0; i < 6; i++) {
		
		double rho = (*resistivitySurface)[i];
		Eigen::Vector3i pos;
		pos[0] = 0;
		pos[1] = 0;
		pos[2] = 0;
		if (i == 0) pos[0] = -1;
		else if (i == 1) pos[0] = 1;
		else if (i == 2) pos[1] = -1;
		else if (i == 3) pos[1] = 1;
		else if (i == 4) pos[2] = -1;
		else if (i == 5) pos[2] = 1;

		if (isAlreadyCalcRotHdS == false) {
			rotHdS[i] = CalcRotHdS(elements, numOfCalcElements, pos);

			//prune 
			double p = 1e-8;
			for (int j = 0; j < 3; j++) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], j); it; ++it)
				{

					if (pow(pow(rotHdS[i].coeff(j, it.col()).real(), 2.0) + pow(rotHdS[i].coeff(j, it.col()).imag(), 2.0), 0.5) < p) {
						rotHdS[i].coeffRef(j, it.col()) = 0.0;
					}
				}
			}

			nCrossRotHdS[i].setZero();
			for (int j = 0; j < 3; j++) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], (j + 2) % 3); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(j, iCol) += surfaceNormalVectors[i].coeff((j + 1) % 3) * rotHdS[i].coeff((j + 2) % 3, iCol).real();
				}
			}
			for (int j = 0; j < 3; j++) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], (j + 1) % 3); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(j, iCol) -= surfaceNormalVectors[i].coeff((j + 2) % 3) * rotHdS[i].coeff((j + 1) % 3, iCol).real();
				}
			}
			
			nCrossRotHdS[i].prune(1e-20);
			nCrossRotHdS[i].makeCompressed();
			nCrossRotHdS[i].data().squeeze();
			
			rotHdS[i].makeCompressed();
			rotHdS[i].data().squeeze();
		}

	}
	//sumNCrossRhoRotHdS->setZero();
	for (int j = 0; j < 3; ++j) {
		for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*sumNCrossRhoRotHdS, j); it; ++it)
		{
			sumNCrossRhoRotHdS->coeffRef(j,it.col())=0.0;
		}
	}
	
	

	for (int i = 0; i < 6; i++) {
		Functions::PlusEqual(sumNCrossRhoRotHdS, &nCrossRotHdS[i], (*resistivitySurface)[i]);	

	}
	//for (int j = 0; j < 3; ++j) {
	//	for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*sumNCrossRhoRotHdS, j); it; ++it)
	//		cout << sumNCrossRhoRotHdS->coeff(it.row(), it.col()) << endl;
	//	{
	//	}
	//}
	//*sumNCrossRhoRotHdS =Functions::DotConst ((*resistivitySurface)[0],& nCrossRotHdS[0]) 
	//	+ Functions::DotConst((*resistivitySurface)[1] , &nCrossRotHdS[1])
	//	+ Functions::DotConst((*resistivitySurface)[2] , & nCrossRotHdS[2])
	//	+ Functions::DotConst((*resistivitySurface)[3] , & nCrossRotHdS[3])
	//	+ Functions::DotConst((*resistivitySurface)[4] , & nCrossRotHdS[4])
	//	+ Functions::DotConst((*resistivitySurface)[5] , & nCrossRotHdS[5]); //This writing way is For Speed
	isAlreadyCalcRotHdS = true;
	//cout << rotHdS[0].rows() << " " << rotHdS[0].cols() <<boundary<< " rotHds"  << endl;
	//cout << sumNCrossRhoRotHdS.rows() << " " << sumNCrossRhoRotHdS.cols() << " rotHds" << endl;
	
	sumNCrossRhoRotHdS->makeCompressed();


	return;
}

Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> UnstructuredElement::UnstructuredElement::CalcRotHdS(unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::Vector3i pos)
{


	int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);

	if (isParent == true) {
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmpRotHdS{ 3,3 * numOfCalcElements };
		tmpRotHdS.reserve(Eigen::VectorXi::Constant(3, 81));
		if (pos[0] == -1) {
			for (int j = 0; j < 2; j++) {
				string childID = ID + Functions::GetBinaryValue(0, j);
				tmpRotHdS += (*elements)[childID]->CalcRotHdS(elements, numOfCalcElements, pos);

			}
		}
		else if (pos[0] == 1) {
			for (int j = 0; j < 2; j++) {
				string childID = ID + Functions::GetBinaryValue(1, j);
				tmpRotHdS += (*elements)[childID]->CalcRotHdS(elements, numOfCalcElements, pos);

			}
		}
		else if (pos[1] == -1) {
			for (int i = 0; i < 2; i++) {
				string childID = ID + Functions::GetBinaryValue(i, 0);
				tmpRotHdS += (*elements)[childID]->CalcRotHdS(elements, numOfCalcElements, pos);

			}
		}
		else if (pos[1] == 1) {
			for (int i = 0; i < 2; i++) {
				string childID = ID + Functions::GetBinaryValue(i, 1);
				tmpRotHdS += (*elements)[childID]->CalcRotHdS(elements, numOfCalcElements, pos);
			}
		}
		else if (pos[2] == -1) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					tmpRotHdS += (*elements)[childID]->CalcRotHdS(elements, numOfCalcElements, pos);
				}
			}
		}
		else if (pos[2] == 1) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					tmpRotHdS += (*elements)[childID]->CalcRotHdS(elements, numOfCalcElements, pos);
				}
			}
		}
		return tmpRotHdS;
	}
	string neighborID = alreadyFoundNeighborID[ipos];
	Element* neighborElement = neighborElements[ipos];

	if (neighborElement->isParent == true) {
		return  neighborElement->CalcRotHdS(elements, numOfCalcElements, -1 * pos);
	}
	if (isAlreadyCalcRotHdS) {
		if (pos[0] == -1) { return rotHdS[0]; }
		if (pos[0] == +1) { return rotHdS[1]; }
		else if (pos[1] == -1) { return rotHdS[2]; }
		else if (pos[1] == +1) { return rotHdS[3]; }
		else if (pos[2] == -1) { return rotHdS[4]; }
		else if (pos[2] == +1) { return rotHdS[5]; }
	}
	if (neighborElement->isAlreadyCalcRotHdS) {
		if (pos[0] == -1) {
			return neighborElement->rotHdS[1];
		}
		else if (pos[0] == +1) { 
			return neighborElement->rotHdS[0]; 
		}
		else if (pos[1] == -1) { 
			return neighborElement->rotHdS[3]; 
		}
		else if (pos[1] == +1) { 
			return neighborElement->rotHdS[2]; 
		}
		else if (pos[2] == -1) { 
			return neighborElement->rotHdS[5]; 
		}
		else if (pos[2] == +1) { 
			return neighborElement->rotHdS[4]; 
		}
	}

	//============Calc Each Faces======================================
	//vector < Eigen::SparseMatrix<double, Eigen::RowMajor> > dHdl(3);
	////vector <Eigen::MatrixXd> dHdl(3);
	//for (int i = 0; i < 3; i++) {
	//	dHdl[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
	//	dHdl[i].reserve(Eigen::VectorXi::Constant(3, 81));
	//}


	vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> edgeVal(4);
	//vector<Eigen::MatrixXd> edgeVal(4);
	for (int i = 0; i < 4; i++) {
		edgeVal[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3,3 * numOfCalcElements };
		edgeVal[i].reserve(Eigen::VectorXi::Constant(3, 243));
		//edgeVal[i].resize(3, 3 * numOfCalcElements);
	}
	for (int i = 0; i < 4; i++) {
		edgeVal[i].setZero();
	}

	int surfaceNum;
	if (pos[0] == -1) {
		surfaceNum = 0;
	}
	else if (pos[0] == 1) {
		surfaceNum = 1;
	}
	else if (pos[1] == -1) {
		surfaceNum = 2;
	}
	else if (pos[1] == 1) {
		surfaceNum = 3;
	}
	else if (pos[2] == -1) {
		surfaceNum = 4;
	}
	else {
		surfaceNum = 5;
	}


	
	vector<Eigen::Vector3d> edgeCenters(4);
	for (int i = 0; i < 4; i++) { //Edge loop
		Eigen::Vector3d edgeCenter;
		bool possibilityTwoPoints = false;
		if (pos[0] == -1) {
			if (i == 0) { edgeCenter = (nodes[0]->x + nodes[3]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 1) { edgeCenter = (nodes[3]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[4]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 3) { edgeCenter = (nodes[0]->x + nodes[4]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[0] == 1) {
			if (i == 0) { edgeCenter = (nodes[1]->x + nodes[2]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 1) { edgeCenter = (nodes[2]->x + nodes[6]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[6]->x + nodes[5]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 3) { edgeCenter = (nodes[5]->x + nodes[1]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[1] == -1) {
			if (i == 0) { edgeCenter = (nodes[0]->x + nodes[1]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 1) { edgeCenter = (nodes[1]->x + nodes[5]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[5]->x + nodes[4]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 3) { edgeCenter = (nodes[4]->x + nodes[0]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[1] == 1) {
			if (i == 0) { edgeCenter = (nodes[3]->x + nodes[2]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 1) { edgeCenter = (nodes[2]->x + nodes[6]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[6]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 3) { edgeCenter = (nodes[7]->x + nodes[3]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[2] == -1) {
			if (i == 0) { edgeCenter = (nodes[0]->x + nodes[1]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 1) { edgeCenter = (nodes[1]->x + nodes[2]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[2]->x + nodes[3]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 3) { edgeCenter = (nodes[3]->x + nodes[0]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[2] == 1) {
			if (i == 0) { edgeCenter = (nodes[4]->x + nodes[5]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 1) { edgeCenter = (nodes[5]->x + nodes[6]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[6]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 3) { edgeCenter = (nodes[7]->x + nodes[4]->x) / 2.0; possibilityTwoPoints = false; }
		}
		edgeCenters[i] = edgeCenter;

		unordered_map<string, Element*>nodeElementsDict;
		nodeElementsDict.reserve(8);
		vector<Element*> nodeElements;
		Eigen::Vector3i tmp; 
		
		for (int j = 0; j < 4; j++) {
			tmp.setZero();
			if (pos.coeffRef(0) == -1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(0) = -1;
					if (j == 2) { tmp.coeffRef(0) = -1; tmp.coeffRef(2) = -1; }
					if (j == 3) tmp.coeffRef(2) = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(0) = -1;
					if (j == 2) { tmp.coeffRef(0) = -1; tmp.coeffRef(1) = +1; }
					if (j == 3) tmp.coeffRef(1) = +1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(0) = -1;
					if (j == 2) { tmp.coeffRef(0) = -1; tmp.coeffRef(2) = +1; }
					if (j == 3) tmp.coeffRef(2) = +1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(0) = -1;
					if (j == 2) { tmp.coeffRef(0) = -1; tmp.coeffRef(1) = -1; }
					if (j == 3) tmp.coeffRef(1) = -1;
				}
			}
			else if (pos.coeffRef(0) == 1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(0) = 1;
					if (j == 2) { tmp.coeffRef(0) = 1; tmp.coeffRef(2) = -1; }
					if (j == 3) tmp.coeffRef(2) = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(0) = 1;
					if (j == 2) { tmp.coeffRef(0) = 1; tmp.coeffRef(1) = +1; }
					if (j == 3) tmp.coeffRef(1) = +1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(0) = 1;
					if (j == 2) { tmp.coeffRef(0) = 1; tmp.coeffRef(2) = +1; }
					if (j == 3) tmp.coeffRef(2) = +1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp.coeffRef(0) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(0) = 1;
					if (j == 2) { tmp.coeffRef(0) = 1; tmp.coeffRef(1) = -1; }
					if (j == 3) tmp.coeffRef(1) = -1;
				}
			}
			else if (pos.coeffRef(1) == -1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(1) = -1;
					if (j == 2) { tmp.coeffRef(1) = -1; tmp.coeffRef(2) = -1; }
					if (j == 3) tmp.coeffRef(2) = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(1) = -1;
					if (j == 2) { tmp.coeffRef(1) = -1; tmp.coeffRef(0) = 1; }
					if (j == 3) tmp.coeffRef(0) = 1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(1) = -1;
					if (j == 2) { tmp.coeffRef(1) = -1; tmp.coeffRef(2) = 1; }
					if (j == 3) tmp.coeffRef(2) = 1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(1) = -1;
					if (j == 2) { tmp.coeffRef(1) = -1; tmp.coeffRef(0) = -1; }
					if (j == 3) tmp.coeffRef(0) = -1;
				}
			}
			else if (pos.coeffRef(1) == 1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(1) = 1;
					if (j == 2) { tmp.coeffRef(1) = 1; tmp.coeffRef(2) = -1; }
					if (j == 3) tmp.coeffRef(2) = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(1) = 1;
					if (j == 2) { tmp.coeffRef(1) = 1; tmp.coeffRef(0) = 1; }
					if (j == 3) tmp.coeffRef(0) = 1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(1) = 1;
					if (j == 2) { tmp.coeffRef(1) = 1; tmp.coeffRef(2) = 1; }
					if (j == 3) tmp.coeffRef(2) = 1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp.coeffRef(1) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(1) = 1;
					if (j == 2) { tmp.coeffRef(1) = 1; tmp.coeffRef(0) = -1; }
					if (j == 3) tmp.coeffRef(0) = -1;
				}
			}
			else if (pos.coeffRef(2) == -1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(2) = -1;
					if (j == 2) { tmp.coeffRef(2) = -1; tmp.coeffRef(1) = -1; }
					if (j == 3) tmp.coeffRef(1) = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(2) = -1;
					if (j == 2) { tmp.coeffRef(2) = -1; tmp.coeffRef(0) = 1; }
					if (j == 3) tmp.coeffRef(0) = 1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(2) = -1;
					if (j == 2) { tmp.coeffRef(2) = -1; tmp.coeffRef(1) = 1; }
					if (j == 3) tmp.coeffRef(1) = 1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(2) = -1;
					if (j == 2) { tmp.coeffRef(2) = -1; tmp.coeffRef(0) = -1; }
					if (j == 3) tmp.coeffRef(0) = -1;
				}
			}
			else if (pos.coeffRef(2) == 1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(2) = 1;
					if (j == 2) { tmp.coeffRef(2) = 1; tmp.coeffRef(1) = -1; }
					if (j == 3) tmp.coeffRef(1) = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(2) = 1;
					if (j == 2) { tmp.coeffRef(2) = 1; tmp.coeffRef(0) = 1; }
					if (j == 3) tmp.coeffRef(0) = 1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(2) = 1;
					if (j == 2) { tmp.coeffRef(2) = 1; tmp.coeffRef(1) = 1; }
					if (j == 3) tmp.coeffRef(1) = 1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp.coeffRef(2) = 0; //nothing to do
					if (j == 1)tmp.coeffRef(2) = 1;
					if (j == 2) { tmp.coeffRef(2) = 1; tmp.coeffRef(0) = -1; }
					if (j == 3) tmp.coeffRef(0) = -1;
				}
			}
			int neighborVal;
			neighborVal = (tmp.coeff(0) + 1) + 3 * (tmp.coeff(1) + 1) + 9 * (tmp.coeff(2) + 1);
			string neighborIDtmp;
			
			neighborIDtmp = alreadyFoundNeighborID[neighborVal];
			if (nodeElementsDict.count(neighborIDtmp) == 0) {
				nodeElementsDict[neighborIDtmp] = neighborElements[neighborVal];
				nodeElements.push_back(neighborElements[neighborVal]);
			}
		}
		vector<double> weightList(nodeElements.size());
		vector< Eigen::SparseMatrix<double, Eigen::RowMajor>> vectorList(nodeElements.size());
		//vector< Eigen::MatrixXd> vectorList(nodeElements.size());
		Eigen::Vector3d x0;
		x0.setZero();
		x0 = edgeCenters[i];
		for (int k = 0; k < nodeElements.size(); k++) {
			double tmpdistance;
			vectorList[k] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
			vectorList[k].reserve(Eigen::VectorXi::Constant(3, 81));
			if (nodeElements[k]->isParent && possibilityTwoPoints) {
				static_cast<UnstructuredElement*>(nodeElements[k])->CalcNearestNeighborVectorEdge(x0, elements, numOfCalcElements, &vectorList[k], &tmpdistance, 2);
			}
			else {
				static_cast<UnstructuredElement*>(nodeElements[k])->CalcNearestNeighborVectorEdge(x0, elements, numOfCalcElements, &vectorList[k], &tmpdistance, 1);
			}
			vectorList[k].makeCompressed();
			weightList[k] = 1 / tmpdistance;
		}
		double wSum = 0;
		for (int k = 0; k < nodeElements.size(); k++) {
			wSum += weightList[k];
		}
		for (int k = 0; k < nodeElements.size(); k++) {
			Functions::PlusEqual(&edgeVal[i], &vectorList[k], weightList[k] / wSum);
		}
	}

	//compose dH/de0, dH/de1, dH/de2
	

	vector < Eigen::SparseMatrix<double, Eigen::RowMajor>>centerVal(2);
	//vector < Eigen::MatrixXd>centerVal(2);
	for (int i = 0; i < 2; i++) {
		centerVal[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
		centerVal[i].reserve(Eigen::VectorXi::Constant(3, 81));
	}
	centerVal[0].coeffRef(0, 3 * calcID) = 1.0;
	centerVal[0].coeffRef(1, 3 * calcID + 1) = 1.0;
	centerVal[0].coeffRef(2, 3 * calcID + 2) = 1.0;
	//neighborID = Functions::GetNeighborElement(elements, this, pos);
	Eigen::Vector3d x0;
	ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
	neighborID = alreadyFoundNeighborID[ipos];
	if (layer > neighborElements[ipos]->layer) {

		x0 = surfaceCenters[surfaceNum] + (surfaceCenters[surfaceNum] - centerCoord);
		CalcInterpolatedVectorInElement(pos, x0, elements, numOfCalcElements, centerVal[1]);
	}
	else {
		x0 = neighborElements[ipos]->centerCoord;
		centerVal[1].coeffRef(0, 3 * neighborElements[ipos]->calcID) = 1.0;
		centerVal[1].coeffRef(1, 3 * neighborElements[ipos]->calcID + 1) = 1.0;
		centerVal[1].coeffRef(2, 3 * neighborElements[ipos]->calcID + 2) = 1.0;
	}
	vector < Eigen::SparseMatrix<double, Eigen::RowMajor> > dHdl(3);
	//vector <Eigen::MatrixXd> dHdl(3);
	for (int i = 0; i < 3; i++) {
		dHdl[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
		dHdl[i].reserve(Eigen::VectorXi::Constant(3, 81));
	}

	dHdl[0] = (centerVal[1] - centerVal[0]) / (x0 - centerCoord).norm();
	dHdl[1] = (edgeVal[1] - edgeVal[3]) / (edgeCenters[1] - edgeCenters[3]).norm();
	dHdl[2] = (edgeVal[2] - edgeVal[0]) / (edgeCenters[2] - edgeCenters[0]).norm();


	//modify dHdl to orthogonal coordinate system
	Eigen::Matrix3d coeffMat;
	coeffMat.setZero();

	vector<Eigen::Vector3d>e(3);
	e[0].setZero();
	e[0].coeffRef(0) = 1.0;
	e[1].setZero();
	e[1].coeffRef(1) = 1.0;
	e[2].setZero();
	e[2].coeffRef(2) = 1.0;

	vector<Eigen::Vector3d>e_old(3);
	e_old[0] = (x0 - centerCoord) / (x0 - centerCoord).norm();
	e_old[1] = (edgeCenters[1] - edgeCenters[3]) / (edgeCenters[1] - edgeCenters[3]).norm();
	e_old[2] = (edgeCenters[2] - edgeCenters[0]) / (edgeCenters[2] - edgeCenters[0]).norm();

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			coeffMat.coeffRef(i, j) = e[j].dot(e_old[i]);
		}
	}
	Eigen::Matrix3d invMat;
	invMat = Functions::inv3(coeffMat);
	//Convert dHdl to dHdx
	vector < Eigen::SparseMatrix<double, Eigen::RowMajor> > dHdx(3);
	//vector <Eigen::MatrixXd> dHdl(3);
	for (int i = 0; i < 3; i++) {
		dHdx[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
		dHdx[i].reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
	}
	for (int i = 0; i < 3; i++) {
		Functions::PlusEqual(&dHdx[0], &dHdl[i], invMat.coeff(0, i));
		Functions::PlusEqual(&dHdx[1], &dHdl[i], invMat.coeff(1, i));
		Functions::PlusEqual(&dHdx[2], &dHdl[i], invMat.coeff(2, i));
	}

	//}	//---------------calc rotH--------------------------
	Eigen::SparseMatrix<double, Eigen::RowMajor> rotH{ 3, 3 * numOfCalcElements };

	rotH.reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables)); //3*3*3*3*3
	rotH.row(0) = dHdx[1].row(2) - dHdx[2].row(1);
	rotH.row(1) = dHdx[2].row(0) - dHdx[0].row(2);
	rotH.row(2) = dHdx[0].row(1) - dHdx[1].row(0);
	Functions::DotConstSelf(dSvector[surfaceNum], &rotH);

	Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> rotHSparse{ 3, 3 * numOfCalcElements };
	rotHSparse.reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
	for (int j = 0; j < rotH.outerSize(); ++j) {
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(rotH, j); it; ++it)
		{
			int iCol = it.col();
			int iRow = it.row();
			rotHSparse.coeffRef(iRow, iCol) = std::complex<double>(rotH.coeff(iRow, iCol), 0.0);
			
		}
	}
	rotHSparse.makeCompressed();
	rotHSparse.data().squeeze();

	return rotHSparse;
}


void UnstructuredElement::UnstructuredElement::CalcNearestNeighborVectorEdge(Eigen::Vector3d x0, unordered_map<string, Element*> *elements, int numChildElements, Eigen::SparseMatrix<double, Eigen::RowMajor>* row, double* distance,int numOfPoints) {
	//This Function is able to be used ONLY ONE LAYER DIFFERENCE!!
	if (isParent == false) {
		
		row->reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
		row->coeffRef(0, 3 * calcID) = 1.0;
		row->coeffRef(1, 3 * calcID + 1) = 1.0;
		row->coeffRef(2, 3 * calcID + 2) = 1.0;
		*distance=(centerCoord-x0).norm();
		return;
	}
	else {
		vector<string>IDVec;
		IDVec.resize( numOfPoints );
		vector<bool>alreadyUsed;
		alreadyUsed.resize( 4 );
		for (int i = 0; i < 4; i++) {
			alreadyUsed[i] = false;
		}
		for (int i = 0; i < numOfPoints; i++) {
			double minDistance = 1e30;
			int pickID = -1;
			string pickChildID;
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					double tmpDistance = ((*elements)[childID]->centerCoord - x0).norm();
					if (tmpDistance < minDistance && alreadyUsed[i + 2 * j] == false) {
						minDistance = tmpDistance;
						pickID = i + 2 * j;
						pickChildID = childID;
					}
				}
			}
			alreadyUsed[pickID] = true;
			IDVec[i] = pickChildID;

		}
		
		Eigen::Vector3d averageCenterPoint;
		averageCenterPoint.setZero();
		for (int i = 0; i < IDVec.size(); i++) {
			Eigen::SparseMatrix<double, Eigen::RowMajor>rowtmp{ 3,3 * numChildElements };
			rowtmp.reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
			rowtmp.coeffRef(0, 3 * (*elements)[IDVec[i]]->calcID) = 1.0;
			rowtmp.coeffRef(1, 3 * (*elements)[IDVec[i]]->calcID + 1) = 1.0;
			rowtmp.coeffRef(2, 3 * (*elements)[IDVec[i]]->calcID + 2) = 1.0;
			averageCenterPoint += (*elements)[IDVec[i]]->centerCoord;
			*row += rowtmp;
		}
		*row = *row / IDVec.size();
		averageCenterPoint= averageCenterPoint/ IDVec.size();
		*distance = (averageCenterPoint - x0).norm();
		return;
	}
	
}


Eigen::Vector3cd UnstructuredElement::UnstructuredElement::CalcDEDH(int derID, int numOfCalcElements, unordered_map<string, Element*>* elements) {
	if (boundary != "NOT_BOUNDARY") {
		Eigen::Vector3cd dEdH;
		dEdH.setZero();
		return dEdH;
	}
	Eigen::VectorXcd dCoeffdH{ 3 * numOfCalcElements };
	dCoeffdH.setZero();
	dCoeffdH.coeffRef(derID) = 1.0;

	Eigen::Vector3cd dEdH;
	dEdH.setZero();


	double sumDs = 0;//test
	double sumWeight = 0;
	Eigen::VectorXcd rhs{ 2 * numOfCalcSurfaceForE };
	rhs.setZero();
	Eigen::MatrixXcd A{ 2 * numOfCalcSurfaceForE , 3 };
	A.setZero();
	Eigen::MatrixXcd At{ 3, 2 * numOfCalcSurfaceForE };
	A.setZero();
	Eigen::MatrixXcd W{ 2 * numOfCalcSurfaceForE, 2 * numOfCalcSurfaceForE };
	W.setZero();
	for (int i = 0; i < numOfCalcSurfaceForE; i++) {
		W.coeffRef(2 * i, 2 * i) = 1;// dSvector[i];
		W.coeffRef(2 * i + 1, 2 * i + 1) = 1;// dSvector[i];
		double dS = dSvector[i];
		double rho = 0.0;
		rho = (*resistivitySurface)[i];
		Eigen::Vector3d pararellUnitVec1;
		pararellUnitVec1.setZero();
		pararellUnitVec1.coeffRef(2) = 1.0;
		Eigen::Vector3d pararellUnitVec2;
		pararellUnitVec2.setZero();

		pararellUnitVec1 = surfaceParallelVectors[i][0];
		pararellUnitVec2 = surfaceParallelVectors[i][1];

		for (int j = 0; j < 3; j++) { //nVec_parallel.dot(E)
			A.coeffRef(2 * i, j) = pararellUnitVec1.coeff(j);
			A.coeffRef(2 * i + 1, j) = pararellUnitVec2.coeff(j);
		}
		rhs.coeffRef(2 * i) = rho * pararellUnitVec1.dot(rotHdS[i] * dCoeffdH / dSvector[i]);
		rhs.coeffRef(2 * i + 1) = rho * pararellUnitVec2.dot(rotHdS[i] * dCoeffdH / dSvector[i]);
	}
	dEdH = (A.adjoint() * W * A).inverse() * A.adjoint() * W * rhs;

	return dEdH;
}



ub::matrix<kv::autodif<kv::complex<double>>> UnstructuredElement::UnstructuredElement::CalcDEDH(ub::vector<kv::autodif<kv::complex<double>>>* HresultTwoItr, std::vector<int>nonZeroRowIndices, unordered_map<string, Element*>* elements, int numOfCalcElements) {
	if (boundary != "NOT_BOUNDARY") {
		ub::matrix< kv::autodif<kv::complex<double>>> tmpE(3, 2);
		return tmpE;
	}

	ub::matrix< kv::autodif<kv::complex<double>>> tmpE(3, 2);
	tmpE(0, 0) = 0; tmpE(1, 0) = 0; tmpE(2, 0) = 0;
	tmpE(0, 1) = 0; tmpE(1, 1) = 0; tmpE(2, 1) = 0;
	double sumDs = 0;//test
	double sumWeight = 0;

	//}
	for (int itr = 0; itr < 2; itr++) {
		
		
		ub::vector<kv::autodif<kv::complex<double>>> rhs( 2 * numOfCalcSurfaceForE );
		Eigen::MatrixXd A{ 2 * numOfCalcSurfaceForE , 3 };
		Eigen::MatrixXd W{ 2 * numOfCalcSurfaceForE, 2 * numOfCalcSurfaceForE };
		W.setZero();
		for (int i = 0; i < numOfCalcSurfaceForE; i++) {
			W.coeffRef(2 * i, 2 * i) = 1;// dSvector[i];
			W.coeffRef(2 * i + 1, 2 * i + 1) = 1;// dSvector[i];
			double dS = dSvector[i];
			double rho = (*resistivitySurface)[i];
			ub::vector<kv::autodif<kv::complex<double>>> tmp(3);
			tmp(0) = 0; tmp(1) = 0; tmp(2) = 0;
			for (int j = 0; j < rotHdS[i].outerSize(); ++j) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], j); it; ++it)
				{
					int iRow = it.row();
					int iCol = it.col();
					int index = -1;
					for (int k = 0; k < nonZeroRowIndices.size(); k++) {
						if (nonZeroRowIndices[k] == itr * 3 * numOfCalcElements + iCol) {
							index = k;
							break;
						}
					}
					if (index < 0) {
						cout << "Wrong In CalcEForAutoDiff" << endl;
						exit(-1);
					}

					tmp(iRow) += 1.0 / dS * rho * kv::complex<double>(rotHdS[i].coeff(iRow, iCol).real(), rotHdS[i].coeff(iRow, iCol).imag()) * (*HresultTwoItr)(index);
					//std::cout << "(" << it.row() << ","; // row index
					//std::cout << it.col() << ")\t"; // col index (here it is equal to k)
				}
			}
			Eigen::Vector3d pararellUnitVec1;

			Eigen::Vector3d pararellUnitVec2;


			pararellUnitVec1 = surfaceParallelVectors[i][0];
			pararellUnitVec2 = surfaceParallelVectors[i][1];

			for (int j = 0; j < 3; j++) { //nVec_parallel.dot(E)
				A.coeffRef(2 * i, j) = pararellUnitVec1.coeff(j);
				A.coeffRef(2 * i + 1, j) = pararellUnitVec2.coeff(j);
			}
			rhs(2 * i) =  (tmp(0) * pararellUnitVec1.coeff(0) + tmp(1) * pararellUnitVec1.coeff(1) + tmp(2) * pararellUnitVec1.coeff(2));
			rhs(2 * i + 1) =  (tmp(0) * pararellUnitVec2.coeff(0) + tmp(1) * pararellUnitVec2.coeff(1) + tmp(2) * pararellUnitVec2.coeff(2));

		}
		ub::vector<kv::autodif<kv::complex<double>>> tmpEOneItr(3);
		tmpEOneItr = Functions::SolveLeastSquare(A, rhs,W);
		for (int i = 0; i < 3; i++) {
			tmpE(i, itr) = tmpEOneItr(i);
		}
	}
	return tmpE;
}


ub::matrix<kv::autodif<kv::complex<double>>> UnstructuredElement::UnstructuredElement::CalcDEDRho(ub::vector<kv::autodif<kv::complex<double>>>* rhoVec, ub::vector<kv::complex<double>>* HresultTwoItr, std::vector<int>nonZeroRowIndices, int numOfCalcElements) {
	if (boundary != "NOT_BOUNDARY") {
		ub::matrix< kv::autodif<kv::complex<double>>> tmpE(3, 2);
		return tmpE;
	}
	ub::matrix< kv::autodif<kv::complex<double>>> tmpE(3, 2);

	for (int itr = 0; itr < 2; itr++) {


		ub::vector<kv::autodif<kv::complex<double>>> rhs( 2 * numOfCalcSurfaceForE );
		Eigen::MatrixXd A{ 2 * numOfCalcSurfaceForE , 3 };
		Eigen::MatrixXd W{ 2 * numOfCalcSurfaceForE, 2 * numOfCalcSurfaceForE };
		W.setZero();
		for (int i = 0; i < numOfCalcSurfaceForE; i++) {
			W.coeffRef(2 * i, 2 * i) = 1;// dSvector[i];
			W.coeffRef(2 * i + 1, 2 * i + 1) = 1;// dSvector[i];
			double dS = dSvector[i];

			kv::autodif<kv::complex<double>>rho;
			rho = 0.0;
			for (int j = 0; j < resistivitySurfaceCoeff[i]->outerSize(); ++j) {
				for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(*resistivitySurfaceCoeff[i], j); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					for (int k = 0; k < nonZeroRowIndices.size(); k++) {
						if (nonZeroRowIndices[k] == iCol) {
							rho += resistivitySurfaceCoeff[i]->coeff(0, iCol).real() * (*rhoVec)(k);
						}
					}
				}
			}

			ub::vector<kv::autodif<kv::complex<double>>> tmp(3);
			tmp(0) = 0; tmp(1) = 0; tmp(2) = 0;
			for (int j = 0; j < rotHdS[i].outerSize(); ++j) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], j); it; ++it)
				{
					int iRow = it.row();
					int iCol = it.col();
					/*int index = -1;
					for (int k = 0; k < nonZeroRowIndices.size(); k++) {
						if (nonZeroRowIndices[k] == itr * 3 * numOfCalcElements + iCol) {
							index = k;
							break;
						}
					}
					if (index < 0) {
						cout << "Wrong In CalcEForAutoDiff" << endl;
						exit(-1);
					}*/

					tmp(iRow) += 1.0 / dS * rho * kv::complex<double>(rotHdS[i].coeff(iRow, iCol).real(), rotHdS[i].coeff(iRow, iCol).imag())  *(*HresultTwoItr)(itr * 3 * numOfCalcElements + iCol);
					//std::cout << "(" << it.row() << ","; // row index
					//std::cout << it.col() << ")\t"; // col index (here it is equal to k)
				}
			}
			Eigen::Vector3d pararellUnitVec1;
			pararellUnitVec1.setZero();
			pararellUnitVec1.coeffRef(2) = 1.0;
			Eigen::Vector3d pararellUnitVec2;
			pararellUnitVec2.setZero();

			pararellUnitVec1 = surfaceParallelVectors[i][0];
			pararellUnitVec2 = surfaceParallelVectors[i][1];

			for (int j = 0; j < 3; j++) { //nVec_parallel.dot(E)
				A.coeffRef(2 * i, j) = pararellUnitVec1.coeff(j);
				A.coeffRef(2 * i + 1, j) = pararellUnitVec2.coeff(j);
			}
			rhs(2 * i) =  (tmp(0) * pararellUnitVec1.coeff(0) + tmp(1) * pararellUnitVec1.coeff(1) + tmp(2) * pararellUnitVec1.coeff(2));
			rhs(2 * i + 1) =  (tmp(0) * pararellUnitVec2.coeff(0) + tmp(1) * pararellUnitVec2.coeff(1) + tmp(2) * pararellUnitVec2.coeff(2));

		}
		ub::vector<kv::autodif<kv::complex<double>>> tmpEOneItr(3);
		tmpEOneItr = Functions::SolveLeastSquare(A, rhs,W);
		for (int i = 0; i < 3; i++) {
			tmpE(i, itr) = tmpEOneItr(i);
		}
	}
	return tmpE;
}


void UnstructuredElement::UnstructuredElement::CalcDivHdS(Eigen::Vector3i pos, unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>& divHdS)
{
	int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);

	/*if (isParent == true) {
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmpDivHdS{ 3,3 * numOfCalcElements };
		tmpDivHdS.reserve(Eigen::VectorXi::Constant(3, 81));
		if (pos[0] == -1) {
			for (int j = 0; j < 2; j++) {
				string childID = ID + Functions::GetBinaryValue(0, j);
				(*elements)[childID]->CalcNDotGradOperatorDsAndGradientOperator(pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);

			}
		}
		else if (pos[0] == 1) {
			for (int j = 0; j < 2; j++) {
				string childID = ID + Functions::GetBinaryValue(1, j);
				(*elements)[childID]->CalcNDotGradOperatorDsAndGradientOperator(pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);

			}
		}
		else if (pos[1] == -1) {
			for (int i = 0; i < 2; i++) {
				string childID = ID + Functions::GetBinaryValue(i, 0);
				(*elements)[childID]->CalcNDotGradOperatorDsAndGradientOperator(pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);

			}
		}
		else if (pos[1] == 1) {
			for (int i = 0; i < 2; i++) {
				string childID = ID + Functions::GetBinaryValue(i, 1);
				(*elements)[childID]->CalcNDotGradOperatorDsAndGradientOperator(pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);
			}
		}
		else if (pos[2] == -1) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					(*elements)[childID]->CalcNDotGradOperatorDsAndGradientOperator(pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);
				}
			}
		}
		else if (pos[2] == 1) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					(*elements)[childID]->CalcNDotGradOperatorDsAndGradientOperator(pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);
				}
			}
		}
		return;
	}
	*/
	string neighborID = alreadyFoundNeighborID[ipos];
	Element* neighborElement = neighborElements[ipos];

	//if (neighborElement->isParent == true) {
	//	neighborElement->CalcNDotGradOperatorDsAndGradientOperator(-1 * pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);
	//	return;
	//}




	//============Calc Each Faces======================================
	//vector < Eigen::SparseMatrix<double, Eigen::RowMajor> > dHdl(3);
	////vector <Eigen::MatrixXd> dHdl(3);
	//for (int i = 0; i < 3; i++) {
	//	dHdl[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
	//	dHdl[i].reserve(Eigen::VectorXi::Constant(3, 81));

	//}


	vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> edgeVal(4);
	//vector<Eigen::MatrixXd> edgeVal(4);
	for (int i = 0; i < 4; i++) {
		edgeVal[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3,3 * numOfCalcElements };
		edgeVal[i].reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
		//edgeVal[i].resize(3, 3 * numOfCalcElements);
	}


	int surfaceNum;
	if (pos[0] == -1) {
		surfaceNum = 0;
	}
	else if (pos[0] == 1) {
		surfaceNum = 1;
	}
	else if (pos[1] == -1) {
		surfaceNum = 2;
	}
	else if (pos[1] == 1) {
		surfaceNum = 3;
	}
	else if (pos[2] == -1) {
		surfaceNum = 4;
	}
	else if (pos[2] == 1) {
		surfaceNum = 5;
	}



	vector<Eigen::Vector3d> edgeCenters(4);
	for (int i = 0; i < 4; i++) { //Edge loop
		Eigen::Vector3d edgeCenter;
		bool possibilityTwoPoints = false;
		if (pos[0] == -1) {
			if (i == 0) { edgeCenter = (nodes[0]->x + nodes[3]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 1) { edgeCenter = (nodes[3]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[4]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 3) { edgeCenter = (nodes[0]->x + nodes[4]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[0] == 1) {
			if (i == 0) { edgeCenter = (nodes[1]->x + nodes[2]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 1) { edgeCenter = (nodes[2]->x + nodes[6]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[6]->x + nodes[5]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 3) { edgeCenter = (nodes[5]->x + nodes[1]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[1] == -1) {
			if (i == 0) { edgeCenter = (nodes[0]->x + nodes[1]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 1) { edgeCenter = (nodes[1]->x + nodes[5]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[5]->x + nodes[4]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 3) { edgeCenter = (nodes[4]->x + nodes[0]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[1] == 1) {
			if (i == 0) { edgeCenter = (nodes[3]->x + nodes[2]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 1) { edgeCenter = (nodes[2]->x + nodes[6]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[6]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 3) { edgeCenter = (nodes[7]->x + nodes[3]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[2] == -1) {
			if (i == 0) { edgeCenter = (nodes[0]->x + nodes[1]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 1) { edgeCenter = (nodes[1]->x + nodes[2]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[2]->x + nodes[3]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 3) { edgeCenter = (nodes[3]->x + nodes[0]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[2] == 1) {
			if (i == 0) { edgeCenter = (nodes[4]->x + nodes[5]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 1) { edgeCenter = (nodes[5]->x + nodes[6]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[6]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 3) { edgeCenter = (nodes[7]->x + nodes[4]->x) / 2.0; possibilityTwoPoints = false; }
		}
		edgeCenters[i] = edgeCenter;

		unordered_map<string, Element*>nodeElementsDict;
		nodeElementsDict.reserve(8);
		vector<Element*> nodeElements;

		for (int j = 0; j < 4; j++) {
			Eigen::Vector3i tmp;
			tmp.setZero();
			if (pos[0] == -1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = -1;
					if (j == 2) { tmp[0] = -1; tmp[2] = -1; }
					if (j == 3) tmp[2] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = -1;
					if (j == 2) { tmp[0] = -1; tmp[1] = +1; }
					if (j == 3) tmp[1] = +1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = -1;
					if (j == 2) { tmp[0] = -1; tmp[2] = +1; }
					if (j == 3) tmp[2] = +1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = -1;
					if (j == 2) { tmp[0] = -1; tmp[1] = -1; }
					if (j == 3) tmp[1] = -1;
				}
			}
			else if (pos[0] == 1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = 1;
					if (j == 2) { tmp[0] = 1; tmp[2] = -1; }
					if (j == 3) tmp[2] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = 1;
					if (j == 2) { tmp[0] = 1; tmp[1] = +1; }
					if (j == 3) tmp[1] = +1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = 1;
					if (j == 2) { tmp[0] = 1; tmp[2] = +1; }
					if (j == 3) tmp[2] = +1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = 1;
					if (j == 2) { tmp[0] = 1; tmp[1] = -1; }
					if (j == 3) tmp[1] = -1;
				}
			}
			else if (pos[1] == -1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = -1;
					if (j == 2) { tmp[1] = -1; tmp[2] = -1; }
					if (j == 3) tmp[2] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = -1;
					if (j == 2) { tmp[1] = -1; tmp[0] = 1; }
					if (j == 3) tmp[0] = 1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = -1;
					if (j == 2) { tmp[1] = -1; tmp[2] = 1; }
					if (j == 3) tmp[2] = 1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = -1;
					if (j == 2) { tmp[1] = -1; tmp[0] = -1; }
					if (j == 3) tmp[0] = -1;
				}
			}
			else if (pos[1] == 1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = 1;
					if (j == 2) { tmp[1] = 1; tmp[2] = -1; }
					if (j == 3) tmp[2] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = 1;
					if (j == 2) { tmp[1] = 1; tmp[0] = 1; }
					if (j == 3) tmp[0] = 1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = 1;
					if (j == 2) { tmp[1] = 1; tmp[2] = 1; }
					if (j == 3) tmp[2] = 1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = 1;
					if (j == 2) { tmp[1] = 1; tmp[0] = -1; }
					if (j == 3) tmp[0] = -1;
				}
			}
			else if (pos[2] == -1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = -1;
					if (j == 2) { tmp[2] = -1; tmp[1] = -1; }
					if (j == 3) tmp[1] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = -1;
					if (j == 2) { tmp[2] = -1; tmp[0] = 1; }
					if (j == 3) tmp[0] = 1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = -1;
					if (j == 2) { tmp[2] = -1; tmp[1] = 1; }
					if (j == 3) tmp[1] = 1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = -1;
					if (j == 2) { tmp[2] = -1; tmp[0] = -1; }
					if (j == 3) tmp[0] = -1;
				}
			}
			else if (pos[2] == 1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = 1;
					if (j == 2) { tmp[2] = 1; tmp[1] = -1; }
					if (j == 3) tmp[1] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = 1;
					if (j == 2) { tmp[2] = 1; tmp[0] = 1; }
					if (j == 3) tmp[0] = 1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = 1;
					if (j == 2) { tmp[2] = 1; tmp[1] = 1; }
					if (j == 3) tmp[1] = 1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = 1;
					if (j == 2) { tmp[2] = 1; tmp[0] = -1; }
					if (j == 3) tmp[0] = -1;
				}
			}
			int neighborVal;
			neighborVal = (tmp.coeff(0) + 1) + 3 * (tmp.coeff(1) + 1) + 9 * (tmp.coeff(2) + 1);
			string neighborIDtmp;
			neighborIDtmp = alreadyFoundNeighborID[neighborVal];
			if (nodeElementsDict.count(neighborIDtmp) == 0) {
				nodeElementsDict[neighborIDtmp] = neighborElements[neighborVal];
				nodeElements.push_back(neighborElements[neighborVal]);
			}
		}
		vector<double> weightList(nodeElements.size());
		vector< Eigen::SparseMatrix<double, Eigen::RowMajor>> vectorList(nodeElements.size());
		//vector< Eigen::MatrixXd> vectorList(nodeElements.size());
		Eigen::Vector3d x0;
		x0.setZero();
		x0 = edgeCenters[i];
		for (int k = 0; k < nodeElements.size(); k++) {
			double tmpdistance;
			vectorList[k].resize(3, 3 * numOfCalcElements);
			vectorList[k].reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
			/*if (nodeElements[k]->isParent && possibilityTwoPoints) {
				static_cast<UnstructuredElement*>(nodeElements[k])->CalcNearestNeighborVectorEdge(x0, elements, numOfCalcElements, &vectorList[k], &tmpdistance, 2);
			}
			else {
				static_cast<UnstructuredElement*>(nodeElements[k])->CalcNearestNeighborVectorEdge(x0, elements, numOfCalcElements, &vectorList[k], &tmpdistance, 1);
			}*/
			nodeElements[k]->CalcCenterVector(elements, numOfCalcElements, vectorList[k], 1.0);
			//weightList[k] = 1 / tmpdistance;
			weightList[k] = 1 / (x0 - nodeElements[k]->centerCoord).norm();
		}
		double wSum = 0;
		for (int k = 0; k < nodeElements.size(); k++) {
			wSum += weightList[k];
		}
		for (int k = 0; k < nodeElements.size(); k++) {
			Functions::PlusEqual(&edgeVal[i], &vectorList[k], weightList[k] / wSum);
		}
	}

	//compose dH/de0, dH/de1, dH/de2

	vector < Eigen::SparseMatrix<double, Eigen::RowMajor>>centerVal(2);
	//vector < Eigen::MatrixXd>centerVal(2);
	for (int i = 0; i < 2; i++) {
		centerVal[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
		centerVal[i].reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));

	}
	centerVal[0].coeffRef(0, 3 * calcID) = 1.0;
	centerVal[0].coeffRef(1, 3 * calcID + 1) = 1.0;
	centerVal[0].coeffRef(2, 3 * calcID + 2) = 1.0;
	//neighborID = Functions::GetNeighborElement(elements, this, pos);
	Eigen::Vector3d x0;
	ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
	neighborID = alreadyFoundNeighborID[ipos];
	if (layer > neighborElements[ipos]->layer) {

		x0 = surfaceCenters[surfaceNum] + (surfaceCenters[surfaceNum] - centerCoord);
		CalcInterpolatedVectorInElement(pos, x0, elements, numOfCalcElements, centerVal[1]);
	}
	else {
		x0 = neighborElements[ipos]->centerCoord;
		neighborElements[ipos]->CalcCenterVector(elements, numOfCalcElements, centerVal[1], 1.0);
		//centerVal[1].coeffRef(0, 3 * neighborElements[ipos]->calcID) = 1.0;
		//centerVal[1].coeffRef(1, 3 * neighborElements[ipos]->calcID + 1) = 1.0;
		//centerVal[1].coeffRef(2, 3 * neighborElements[ipos]->calcID + 2) = 1.0;
	}

	vector < Eigen::SparseMatrix<double, Eigen::RowMajor> > dHdl(3);
	//vector <Eigen::MatrixXd> dHdl(3);
	for (int i = 0; i < 3; i++) {
		dHdl[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
		dHdl[i].reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
	}

	dHdl[0] = (centerVal[1] - centerVal[0]) / (x0 - centerCoord).norm();
	dHdl[1] = (edgeVal[1] - edgeVal[3]) / (edgeCenters[1] - edgeCenters[3]).norm();
	dHdl[2] = (edgeVal[2] - edgeVal[0]) / (edgeCenters[2] - edgeCenters[0]).norm();


	//modify dHdl to orthogonal coordinate system
	Eigen::Matrix3d coeffMat;
	coeffMat.setZero();

	vector<Eigen::Vector3d>e(3);
	e[0].setZero();
	e[0].coeffRef(0) = 1.0;
	e[1].setZero();
	e[1].coeffRef(1) = 1.0;
	e[2].setZero();
	e[2].coeffRef(2) = 1.0;

	vector<Eigen::Vector3d>e_old(3);
	e_old[0] = (x0 - centerCoord) / (x0 - centerCoord).norm();
	e_old[1] = (edgeCenters[1] - edgeCenters[3]) / (edgeCenters[1] - edgeCenters[3]).norm();
	e_old[2] = (edgeCenters[2] - edgeCenters[0]) / (edgeCenters[2] - edgeCenters[0]).norm();

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			coeffMat.coeffRef(i, j) = e[j].dot(e_old[i]);
		}
	}
	Eigen::Matrix3d invMat;
	invMat = Functions::inv3(coeffMat);
	//Convert dHdl to dHdx
	vector < Eigen::SparseMatrix<double, Eigen::RowMajor> > dHdx(3);
	//vector <Eigen::MatrixXd> dHdl(3);
	for (int i = 0; i < 3; i++) {
		dHdx[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
		dHdx[i].reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
	}
	for (int i = 0; i < 3; i++) {
		Functions::PlusEqual(&dHdx[0], &dHdl[i], invMat.coeff(0, i));
		Functions::PlusEqual(&dHdx[1], &dHdl[i], invMat.coeff(1, i));
		Functions::PlusEqual(&dHdx[2], &dHdl[i], invMat.coeff(2, i));
	}
	
	for (int i = 0; i < 3; i++) {
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(dHdx[i], i); it; ++it) {
			int col = it.col();
			divHdS.coeffRef(0, col) += dHdx[i].coeff(i, col)*dSvector[surfaceNum];
	
		}
	}
	return;
}
void UnstructuredElement::UnstructuredElement::CalcSumNDivHdS( unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* sumNDivHdS) {
	Eigen::SparseMatrix<double, Eigen::RowMajor> sumNDivHdStmp{ 3,3 * numOfCalcElements }; //need to prune
	sumNDivHdStmp.reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
	for (int i = 0; i < 6; i++) {
		Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> divHdS{ 1,3 * numOfCalcElements };
		divHdS.reserve(Eigen::VectorXi::Constant(1, numOfRelatedCalcVariables));
		Eigen::Vector3i pos;
		pos.setZero();
		int sumComp;
		if (i == 0) {pos[0] = -1; sumComp = 0;}
		else if (i == 1) { pos[0] = 1; sumComp = 0;	}
		else if (i == 2) { pos[1] = -1; sumComp = 1;}
		else if (i == 3) { pos[1] = 1; sumComp = 1;}
		else if (i == 4) { pos[2] = -1;  sumComp = 2;}
		else if (i == 5) { pos[2] = 1;  sumComp = 2;}
		CalcDivHdS(pos, elements, numOfCalcElements, divHdS);

		for (int j = 0; j < 3; j++) {
			for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(divHdS, 0); it; ++it) {
				sumNDivHdStmp.coeffRef(j, it.col()) += divHdS.coeff(0, it.col()).real() * surfaceNormalVectors[i].coeff(j);
			}
		}
		
		/*for (int j = 0; j < 3; j++) {
			for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(divHdS, 0); it; ++it) {
				sumNDivHdStmp.coeffRef(j, it.col()) += divHdS.coeff(0, it.col()).real() * surfaceNormalVectors[i].coeff(j);
			}
		}*/
	}

	sumNDivHdStmp.prune(1e-20);

	for (int i = 0; i < 3; i++) {
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sumNDivHdStmp, i); it; ++it) {
			sumNDivHdS->coeffRef(i, it.col()) = sumNDivHdStmp.coeff(i, it.col());
		}
	}
	sumNDivHdS->makeCompressed();

}

void UnstructuredElement::UnstructuredElement::CalcOperatorSurface
(Eigen::Vector3i pos, unordered_map<string, Element*>* elements, int numOfCalcElements,
	vector<Eigen::Triplet<double>>& divGradDsOperator, int& locDivGrad,
	vector<Eigen::Triplet<double>>& divOperator, int& locDiv, vector<Eigen::Triplet<double>>& gradDsOperator, int& locGrad)
{
	int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);

	/*if (isParent == true) {
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmpDivHdS{ 3,3 * numOfCalcElements };
		tmpDivHdS.reserve(Eigen::VectorXi::Constant(3, 81));
		if (pos[0] == -1) {
			for (int j = 0; j < 2; j++) {
				string childID = ID + Functions::GetBinaryValue(0, j);
				(*elements)[childID]->CalcNDotGradOperatorDsAndGradientOperator(pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);

			}
		}
		else if (pos[0] == 1) {
			for (int j = 0; j < 2; j++) {
				string childID = ID + Functions::GetBinaryValue(1, j);
				(*elements)[childID]->CalcNDotGradOperatorDsAndGradientOperator(pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);

			}
		}
		else if (pos[1] == -1) {
			for (int i = 0; i < 2; i++) {
				string childID = ID + Functions::GetBinaryValue(i, 0);
				(*elements)[childID]->CalcNDotGradOperatorDsAndGradientOperator(pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);

			}
		}
		else if (pos[1] == 1) {
			for (int i = 0; i < 2; i++) {
				string childID = ID + Functions::GetBinaryValue(i, 1);
				(*elements)[childID]->CalcNDotGradOperatorDsAndGradientOperator(pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);
			}
		}
		else if (pos[2] == -1) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					(*elements)[childID]->CalcNDotGradOperatorDsAndGradientOperator(pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);
				}
			}
		}
		else if (pos[2] == 1) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					(*elements)[childID]->CalcNDotGradOperatorDsAndGradientOperator(pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);
				}
			}
		}
		return;
	}
	*/
	string neighborID = alreadyFoundNeighborID[ipos];
	Element* neighborElement = neighborElements[ipos];

	//if (neighborElement->isParent == true) {
	//	neighborElement->CalcNDotGradOperatorDsAndGradientOperator(-1 * pos, elements, numOfCalcElements, nDotGradOperatorDs, gradientOperatorDs);
	//	return;
	//}




	//============Calc Each Faces======================================
	//vector < Eigen::SparseMatrix<double, Eigen::RowMajor> > dHdl(3);
	////vector <Eigen::MatrixXd> dHdl(3);
	//for (int i = 0; i < 3; i++) {
	//	dHdl[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
	//	dHdl[i].reserve(Eigen::VectorXi::Constant(3, 81));

	//}


	vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> edgeVal(4);
	//vector<Eigen::MatrixXd> edgeVal(4);
	for (int i = 0; i < 4; i++) {
		edgeVal[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3,3 * numOfCalcElements };
		edgeVal[i].reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
		//edgeVal[i].resize(3, 3 * numOfCalcElements);
	}
	

	int surfaceNum;
	if (pos[0] == -1) {
		surfaceNum = 0;
	}
	else if (pos[0] == 1) {
		surfaceNum = 1;
	}
	else if (pos[1] == -1) {
		surfaceNum = 2;
	}
	else if (pos[1] == 1) {
		surfaceNum = 3;
	}
	else if (pos[2] == -1) {
		surfaceNum = 4;
	}
	else if (pos[2] == 1) {
		surfaceNum = 5;
	}



	vector<Eigen::Vector3d> edgeCenters(4);
	for (int i = 0; i < 4; i++) { //Edge loop
		Eigen::Vector3d edgeCenter;
		bool possibilityTwoPoints = false;
		if (pos[0] == -1) {
			if (i == 0) { edgeCenter = (nodes[0]->x + nodes[3]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 1) { edgeCenter = (nodes[3]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[4]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 3) { edgeCenter = (nodes[0]->x + nodes[4]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[0] == 1) {
			if (i == 0) { edgeCenter = (nodes[1]->x + nodes[2]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 1) { edgeCenter = (nodes[2]->x + nodes[6]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[6]->x + nodes[5]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 3) { edgeCenter = (nodes[5]->x + nodes[1]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[1] == -1) {
			if (i == 0) { edgeCenter = (nodes[0]->x + nodes[1]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 1) { edgeCenter = (nodes[1]->x + nodes[5]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[5]->x + nodes[4]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 3) { edgeCenter = (nodes[4]->x + nodes[0]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[1] == 1) {
			if (i == 0) { edgeCenter = (nodes[3]->x + nodes[2]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 1) { edgeCenter = (nodes[2]->x + nodes[6]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[6]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = true; }
			else if (i == 3) { edgeCenter = (nodes[7]->x + nodes[3]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[2] == -1) {
			if (i == 0) { edgeCenter = (nodes[0]->x + nodes[1]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 1) { edgeCenter = (nodes[1]->x + nodes[2]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[2]->x + nodes[3]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 3) { edgeCenter = (nodes[3]->x + nodes[0]->x) / 2.0; possibilityTwoPoints = false; }
		}
		else if (pos[2] == 1) {
			if (i == 0) { edgeCenter = (nodes[4]->x + nodes[5]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 1) { edgeCenter = (nodes[5]->x + nodes[6]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 2) { edgeCenter = (nodes[6]->x + nodes[7]->x) / 2.0; possibilityTwoPoints = false; }
			else if (i == 3) { edgeCenter = (nodes[7]->x + nodes[4]->x) / 2.0; possibilityTwoPoints = false; }
		}
		edgeCenters[i] = edgeCenter;

		unordered_map<string, Element*>nodeElementsDict;
		nodeElementsDict.reserve(8);
		vector<Element*> nodeElements;

		for (int j = 0; j < 4; j++) {
			Eigen::Vector3i tmp;
			tmp.setZero();
			if (pos[0] == -1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = -1;
					if (j == 2) { tmp[0] = -1; tmp[2] = -1; }
					if (j == 3) tmp[2] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = -1;
					if (j == 2) { tmp[0] = -1; tmp[1] = +1; }
					if (j == 3) tmp[1] = +1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = -1;
					if (j == 2) { tmp[0] = -1; tmp[2] = +1; }
					if (j == 3) tmp[2] = +1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = -1;
					if (j == 2) { tmp[0] = -1; tmp[1] = -1; }
					if (j == 3) tmp[1] = -1;
				}
			}
			else if (pos[0] == 1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = 1;
					if (j == 2) { tmp[0] = 1; tmp[2] = -1; }
					if (j == 3) tmp[2] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = 1;
					if (j == 2) { tmp[0] = 1; tmp[1] = +1; }
					if (j == 3) tmp[1] = +1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = 1;
					if (j == 2) { tmp[0] = 1; tmp[2] = +1; }
					if (j == 3) tmp[2] = +1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = 1;
					if (j == 2) { tmp[0] = 1; tmp[1] = -1; }
					if (j == 3) tmp[1] = -1;
				}
			}
			else if (pos[1] == -1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = -1;
					if (j == 2) { tmp[1] = -1; tmp[2] = -1; }
					if (j == 3) tmp[2] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = -1;
					if (j == 2) { tmp[1] = -1; tmp[0] = 1; }
					if (j == 3) tmp[0] = 1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = -1;
					if (j == 2) { tmp[1] = -1; tmp[2] = 1; }
					if (j == 3) tmp[2] = 1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = -1;
					if (j == 2) { tmp[1] = -1; tmp[0] = -1; }
					if (j == 3) tmp[0] = -1;
				}
			}
			else if (pos[1] == 1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = 1;
					if (j == 2) { tmp[1] = 1; tmp[2] = -1; }
					if (j == 3) tmp[2] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = 1;
					if (j == 2) { tmp[1] = 1; tmp[0] = 1; }
					if (j == 3) tmp[0] = 1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = 1;
					if (j == 2) { tmp[1] = 1; tmp[2] = 1; }
					if (j == 3) tmp[2] = 1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[1] = 0; //nothing to do
					if (j == 1)tmp[1] = 1;
					if (j == 2) { tmp[1] = 1; tmp[0] = -1; }
					if (j == 3) tmp[0] = -1;
				}
			}
			else if (pos[2] == -1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = -1;
					if (j == 2) { tmp[2] = -1; tmp[1] = -1; }
					if (j == 3) tmp[1] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = -1;
					if (j == 2) { tmp[2] = -1; tmp[0] = 1; }
					if (j == 3) tmp[0] = 1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = -1;
					if (j == 2) { tmp[2] = -1; tmp[1] = 1; }
					if (j == 3) tmp[1] = 1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = -1;
					if (j == 2) { tmp[2] = -1; tmp[0] = -1; }
					if (j == 3) tmp[0] = -1;
				}
			}
			else if (pos[2] == 1) {
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = 1;
					if (j == 2) { tmp[2] = 1; tmp[1] = -1; }
					if (j == 3) tmp[1] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = 1;
					if (j == 2) { tmp[2] = 1; tmp[0] = 1; }
					if (j == 3) tmp[0] = 1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = 1;
					if (j == 2) { tmp[2] = 1; tmp[1] = 1; }
					if (j == 3) tmp[1] = 1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[2] = 0; //nothing to do
					if (j == 1)tmp[2] = 1;
					if (j == 2) { tmp[2] = 1; tmp[0] = -1; }
					if (j == 3) tmp[0] = -1;
				}
			}
			int neighborVal;
			neighborVal = (tmp.coeff(0) + 1) + 3 * (tmp.coeff(1) + 1) + 9 * (tmp.coeff(2) + 1);
			string neighborIDtmp;
			neighborIDtmp = alreadyFoundNeighborID[neighborVal];
			if (nodeElementsDict.count(neighborIDtmp) == 0) {
				nodeElementsDict[neighborIDtmp] = neighborElements[neighborVal];
				nodeElements.push_back(neighborElements[neighborVal]);
			}
		}
		vector<double> weightList(nodeElements.size());
		vector< Eigen::SparseMatrix<double, Eigen::RowMajor>> vectorList(nodeElements.size());
		//vector< Eigen::MatrixXd> vectorList(nodeElements.size());
		Eigen::Vector3d x0;
		x0.setZero();
		x0 = edgeCenters[i];
		for (int k = 0; k < nodeElements.size(); k++) {
			double tmpdistance;
			vectorList[k].resize(3, 3 * numOfCalcElements);
			vectorList[k].reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
			/*if (nodeElements[k]->isParent && possibilityTwoPoints) {
				static_cast<UnstructuredElement*>(nodeElements[k])->CalcNearestNeighborVectorEdge(x0, elements, numOfCalcElements, &vectorList[k], &tmpdistance, 2);
			}
			else {
				static_cast<UnstructuredElement*>(nodeElements[k])->CalcNearestNeighborVectorEdge(x0, elements, numOfCalcElements, &vectorList[k], &tmpdistance, 1);
			}*/
			nodeElements[k]->CalcCenterVector(elements, numOfCalcElements, vectorList[k],1.0);
			//weightList[k] = 1 / tmpdistance;
			weightList[k] = 1 / (x0 - nodeElements[k]->centerCoord).norm();
		}
		double wSum = 0;
		for (int k = 0; k < nodeElements.size(); k++) {
			wSum += weightList[k];
		}
		for (int k = 0; k < nodeElements.size(); k++) {
			Functions::PlusEqual(&edgeVal[i], &vectorList[k], weightList[k] / wSum);
		}
	}

	//compose dH/de0, dH/de1, dH/de2

	vector < Eigen::SparseMatrix<double, Eigen::RowMajor>>centerVal(2);
	//vector < Eigen::MatrixXd>centerVal(2);
	for (int i = 0; i < 2; i++) {
		centerVal[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
		centerVal[i].reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));

	}
	centerVal[0].coeffRef(0, 3 * calcID) = 1.0;
	centerVal[0].coeffRef(1, 3 * calcID + 1) = 1.0;
	centerVal[0].coeffRef(2, 3 * calcID + 2) = 1.0;
	//neighborID = Functions::GetNeighborElement(elements, this, pos);
	Eigen::Vector3d x0;
	ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
	neighborID = alreadyFoundNeighborID[ipos];
	if (layer > neighborElements[ipos]->layer) {

		x0 = surfaceCenters[surfaceNum] + (surfaceCenters[surfaceNum] - centerCoord);
		CalcInterpolatedVectorInElement(pos, x0, elements, numOfCalcElements,centerVal[1] );
	}
	else {
		x0 = neighborElements[ipos]->centerCoord;
		neighborElements[ipos]->CalcCenterVector(elements, numOfCalcElements, centerVal[1],1.0);
		//centerVal[1].coeffRef(0, 3 * neighborElements[ipos]->calcID) = 1.0;
		//centerVal[1].coeffRef(1, 3 * neighborElements[ipos]->calcID + 1) = 1.0;
		//centerVal[1].coeffRef(2, 3 * neighborElements[ipos]->calcID + 2) = 1.0;
	}

	vector < Eigen::SparseMatrix<double, Eigen::RowMajor> > dHdl(3);
	//vector <Eigen::MatrixXd> dHdl(3);
	for (int i = 0; i < 3; i++) {
		dHdl[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
		dHdl[i].reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
	}

	dHdl[0] = (centerVal[1] - centerVal[0]) / (x0 - centerCoord).norm();
	dHdl[1] = (edgeVal[1] - edgeVal[3]) / (edgeCenters[1] - edgeCenters[3]).norm();
	dHdl[2] = (edgeVal[2] - edgeVal[0]) / (edgeCenters[2] - edgeCenters[0]).norm();


	//modify dHdl to orthogonal coordinate system
	Eigen::Matrix3d coeffMat;
	coeffMat.setZero();

	vector<Eigen::Vector3d>e(3);
	e[0].setZero();
	e[0].coeffRef(0) = 1.0;
	e[1].setZero();
	e[1].coeffRef(1) = 1.0;
	e[2].setZero();
	e[2].coeffRef(2) = 1.0;

	vector<Eigen::Vector3d>e_old(3);
	e_old[0] = (x0 - centerCoord) / (x0 - centerCoord).norm();
	e_old[1] = (edgeCenters[1] - edgeCenters[3]) / (edgeCenters[1] - edgeCenters[3]).norm();
	e_old[2] = (edgeCenters[2] - edgeCenters[0]) / (edgeCenters[2] - edgeCenters[0]).norm();

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			coeffMat.coeffRef(i, j) = e[j].dot(e_old[i]);
		}
	}
	Eigen::Matrix3d invMat;
	invMat = Functions::inv3(coeffMat);
	//Convert dHdl to dHdx
	vector < Eigen::SparseMatrix<double, Eigen::RowMajor> > dHdx(3);
	//vector <Eigen::MatrixXd> dHdl(3);
	for (int i = 0; i < 3; i++) {
		dHdx[i] = Eigen::SparseMatrix<double, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
		dHdx[i].reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
	}
	for (int i = 0; i < 3; i++) {
		Functions::PlusEqual(&dHdx[0], &dHdl[i], invMat.coeff(0, i));
		Functions::PlusEqual(&dHdx[1], &dHdl[i], invMat.coeff(1, i));
		Functions::PlusEqual(&dHdx[2], &dHdl[i], invMat.coeff(2, i));
	}

	//dHdxはベクトルの三方向微分だが、スカラー量の3方向微分が必要なので、dHdxから必要な情報のみ取り出す。
	//プログラムの使いまわしの観点からdPhidx専用の書き方はしない(3*numOfCalcElementsをnumOfCalcElementsに関連する箇所すべて書き直す必要があり面倒）
	
	int isurf;
	if (pos[0] == -1) { isurf = 0; }
	else if (pos[0] == 1) { isurf = 1; }
	else if (pos[1] == -1) { isurf = 2; }
	else if (pos[1] == 1) { isurf = 3; }
	else if (pos[2] == -1) { isurf = 4; }
	else if (pos[2] == 1) { isurf = 5; }

	/*double resis = ((*resistivitySurface)[isurf]);
	if (isAirGroundBoundaryCell && (neighborElements[ipos]->property->type == Property::Property::AIR && !neighborElements[ipos]->isAirGroundBoundaryCell)) {
		resis = resistivity;
	}
	else if ((property->type == Property::Property::AIR && !isAirGroundBoundaryCell) && neighborElements[ipos]->isAirGroundBoundaryCell) {
		resis = resistivity;
	}*/
	//if (calcGradDivOperationElement) {
		for (int i = 0; i < 3; i++) {
			int j = 0;
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(dHdx[i], 0); it; ++it)
			{
				if (it.col() % 3 != 0) {
					cout << "Something is Wrong in CalcGradOperator" << endl;
					exit(1);
				}
				Eigen::Triplet<double> val(calcID, int(it.col() / 3), dHdx[i].coeff(0, it.col()) * surfaceNormalVectors[isurf].coeff(i) * dSvector[isurf]);
				divGradDsOperator[locDivGrad] = val;
				locDivGrad++;
			}
		}
	//}
	//else {
		//Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> HonSurface{ 3,3 * numOfCalcElements };
		//HonSurface.reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
		//for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*resistivitySurfaceCoeff[isurf], 0); it; ++it) { //reuse
		//	HonSurface.coeffRef(0, 3 * it.col()) = resistivitySurfaceCoeff[isurf]->coeff(0, it.col());
		//	HonSurface.coeffRef(1, 3 * it.col() + 1) = resistivitySurfaceCoeff[isurf]->coeff(0, it.col());
		//	HonSurface.coeffRef(2, 3 * it.col() + 2) = resistivitySurfaceCoeff[isurf]->coeff(0, it.col());

		//	for (int j = 0; j < 3; j++) {
		//		if (surfaceNormalVectors[isurf].coeff(j) != 0.0) {
		//			Eigen::Triplet<double> val(3 * calcID + j, it.col(), resistivitySurfaceCoeff[isurf]->coeff(0, it.col()).real() * surfaceNormalVectors[isurf].coeff(j) * dSvector[isurf]);
	
		//			gradDsOperator[locGrad] = val;
		//			locGrad++;
		//		}
		//	}

		//}

		//for (int i = 0; i < 3; i++) {
		//	for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(HonSurface, i); it; ++it) { //reuse
		//		if (surfaceNormalVectors[isurf].coeff(i) != 0.0) {
		//			Eigen::Triplet<double> val(calcID, it.col(),
		//				HonSurface.coeff(i, it.col()).real() * surfaceNormalVectors[isurf].coeff(i) * dSvector[isurf]);
		//			divOperator[locDiv] = val;
		//			locDiv++;
		//		}
		//	}
		//}
	//}
	return;

}
void  UnstructuredElement::UnstructuredElement::CalcOperator(unordered_map<string, Element*>* elements, int numOfCalcElements,
	vector<Eigen::Triplet<double>>& divGradDsOperator, int& locDivGrad,
	vector<Eigen::Triplet<double>>& divOperator, int& locDiv, vector<Eigen::Triplet<double>>& gradDsOperator, int& locgrad)
{
	for (int i = 0; i < 6; i++) {
		Eigen::Vector3i pos;
		pos[0] = 0;
		pos[1] = 0;
		pos[2] = 0;
		if (i == 0) pos[0] = -1;
		else if (i == 1) pos[0] = 1;
		else if (i == 2) pos[1] = -1;
		else if (i == 3) pos[1] = 1;
		else if (i == 4) pos[2] = -1;
		else if (i == 5) pos[2] = 1;
		CalcOperatorSurface(pos, elements, numOfCalcElements, divGradDsOperator, locDivGrad,
			divOperator, locDiv, gradDsOperator, locgrad); 
	}


}
void UnstructuredElement::UnstructuredElement::CalcNDotHdSOperator(Eigen::Vector3i pos, int numOfCalcElements, Eigen::SparseMatrix<double, Eigen::RowMajor>& sumNDotHdsOperator) {
	int isurf;
	if (pos[0] == -1) { isurf = 0; }
	else if (pos[0] == 1) { isurf = 1; }
	else if (pos[1] == -1) { isurf = 2; }
	else if (pos[1] == 1) { isurf = 3; }
	else if (pos[2] == -1) { isurf = 4; }
	else if (pos[2] == 1) { isurf = 5; }
	Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> HonSurface{ 3,3 * numOfCalcElements };
	HonSurface.reserve(Eigen::VectorXi::Constant(3, numOfRelatedCalcVariables));
	for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*resistivitySurfaceCoeff[isurf], 0); it; ++it) { //reuse
		HonSurface.coeffRef(0, 3 * it.col()) = resistivitySurfaceCoeff[isurf]->coeff(0, it.col());
		HonSurface.coeffRef(1, 3 * it.col() + 1) = resistivitySurfaceCoeff[isurf]->coeff(0, it.col());
		HonSurface.coeffRef(2, 3 * it.col() + 2) = resistivitySurfaceCoeff[isurf]->coeff(0, it.col());
	}

	for (int i = 0; i < 3; i++) {
		for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(HonSurface, i); it; ++it) { //reuse
			sumNDotHdsOperator.coeffRef(0, it.col()) += HonSurface.coeff(i, it.col()).real() * surfaceNormalVectors[isurf].coeff(i) *dSvector[isurf];
		}
	}
}
void UnstructuredElement::UnstructuredElement::CalcSumNDotHdSOperator(int numOfCalcElements, Eigen::SparseMatrix<double, Eigen::RowMajor>& sumNDotHdsOperator)
{
	for (int i = 0; i < 6; i++) {
		Eigen::Vector3i pos;
		pos[0] = 0;
		pos[1] = 0;
		pos[2] = 0;
		if (i == 0) pos[0] = -1;
		else if (i == 1) pos[0] = 1;
		else if (i == 2) pos[1] = -1;
		else if (i == 3) pos[1] = 1;
		else if (i == 4) pos[2] = -1;
		else if (i == 5) pos[2] = 1;
		CalcNDotHdSOperator(pos, numOfCalcElements, sumNDotHdsOperator);
	}
	sumNDotHdsOperator = sumNDotHdsOperator / dv;
}

void UnstructuredElement::UnstructuredElement::CalcInterpolatedVectorInElement(Eigen::Vector3i val, Eigen::Vector3d x0, unordered_map < string, Element* >* elements, int numChildElements, Eigen::SparseMatrix<double, Eigen::RowMajor>& row) {
	//val方向の隣接セル内の値を補間する。


	//vector< Eigen::MatrixXd> relatedElementsRow;
	//vector<Element*> relatedElementsVector;
	vector < Eigen::Vector3d> relatedElementsCenterCoord;
	Eigen::Vector3i pos;
	pos.setZero();
	unordered_map<string, Element*> nodeElementsDict;
	vector<Element*> nodeElementsVector;
	nodeElementsDict.reserve(100); //100は適当
	string neighborIDVirtual = Functions::GetVirturalNeighborElement(elements, ID, layer, val, nx, ny, nz);
	for (int i = 0; i < 6; i++) {
		pos.setZero();
		if (i == 0)     pos[0] = -1;
		else if (i == 1)pos[0] = 1;
		else if (i == 2)pos[1] = -1;
		else if (i == 3)pos[1] = 1;
		else if (i == 4)pos[2] = -1;
		else if (i == 5)pos[2] = 1;
		string neighborID = Functions::GetNeighborElement(elements, neighborIDVirtual, layer, pos, nx, ny, nz);
		//if (neighborIDVirtual.length() != neighborID.length()) {
		//	cout << ID << " " << neighborIDVirtual << " " << neighborID << endl;
		//	cout << val[0] << " " << val[1] << " " << val[2] << endl;
		//	cout << pos[0] << " " << pos[1] << " " << pos[2] << endl;
		//}
		if (neighborID == "-X_BOUNDARY" || neighborID == "+X_BOUNDARY" ||
			neighborID == "-Y_BOUNDARY" || neighborID == "+Y_BOUNDARY" ||
			neighborID == "-Z_BOUNDARY" || neighborID == "+Z_BOUNDARY") {
			std::cout << "Splitting Elements Next To Boundaries is Not Allowed!!!" << std::endl;
			exit(1);
		}
		if (nodeElementsDict.count(neighborID) == 0) {
			Element* element = (*elements)[neighborID];
			nodeElementsDict[neighborID] = element;
			nodeElementsVector.push_back(element);
			//cout <<boundary<< neighborID<<ID << endl;
			//cout << "pos" << pos[0]<< pos[1]<< pos[2]  << endl;
			//cout <<"val"<< val[0] << val[1] << val[2] << endl;
			relatedElementsCenterCoord.push_back(element->centerCoord);
		}
	}

	vector<double> w = Functions::CalcWeight(relatedElementsCenterCoord, x0);

	for (int i = 0; i < w.size(); i++) {
		Element* element = nodeElementsVector[i];
		element->CalcCenterVector(elements, numChildElements, row,w[i]);
	}
}
