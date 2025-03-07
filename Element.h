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
#include "Node.h"
namespace ub = boost::numeric::ublas;
using namespace std;

namespace Element {
	class Element
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			std::string ID = "-1";
		std::vector<Node::Node*> nodes; //order is (i,j,k),(i+1,j,k),(i+1,j+1,k),(i,j+1,k),(i,j,k+1),(i+1,j,k+1),(i+1,j+1,k+1),(i,j+1,k+1)
		bool isAirGroundBoundaryCell;
		bool isSecondCellOfAirGroundBoundary = false;
		bool isThirdCellOfAirGroundBoundary = false;
		bool isInvertedElement = false;
		int propID = -1;
		int outputSensitivityID = -1;
		Eigen::Vector3d rootCoord = Eigen::Vector3d::Zero();
		Eigen::Vector3d centerCoord = Eigen::Vector3d::Zero();
		double dx;
		double dy;
		double dz;
		double dv;
		int IDX = -1;
		int IDY = -1;
		int IDZ = -1;

		int nx = -1;
		int ny = -1;
		int nz = -1;

		double rhoXY = 0;
		double rhoYX = 0;
		double phiXY = 0;
		double phiYX = 0;
		int calcID; //全体の係数行列の行数に使う
		bool isParent;
		bool isObservationElement = false;
		ObsData::ObsData* impedanceObsData;
		ObsData::ObsData* tipperObsData;

		std::vector<Eigen::Vector3cd, Eigen::aligned_allocator<Eigen::Vector3cd>> E;
		std::vector<Eigen::Vector3cd, Eigen::aligned_allocator<Eigen::Vector3cd>> H;
		std::vector<Eigen::Matrix2cd, Eigen::aligned_allocator<Eigen::Matrix2cd>> Z;
		std::vector<Eigen::Vector2cd, Eigen::aligned_allocator<Eigen::Vector2cd>> T;
		std::vector<Eigen::Matrix2cd, Eigen::aligned_allocator<Eigen::Matrix2cd>> Zpre;
		std::vector<Eigen::Matrix2cd, Eigen::aligned_allocator<Eigen::Matrix2cd>> Zpos;
		Property::Property* property;
		double resistivity;
		double initialResistivity;
		std::vector<double>* resistivitySurface;
		bool isAlreadyCalcResisCoeff = false;
		std::vector<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>*> resistivitySurfaceCoeff;
		std::vector<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>*> diffResistivitySurfaceCoeff;
		Eigen::SparseMatrix<double,Eigen::RowMajor> MPCResistivityCoeff;
		int layer;
		bool isAlreadyCalcRotHdS = false;
		std::string boundary = "NOT_BOUNDARY";
		vector<string> alreadyFoundNeighborID{ 27,"NOT_FOUND" };
		vector<Element*> neighborElements{ 27 };
		Eigen::SparseMatrix<double, Eigen::RowMajor> dKDRho;
		map<string, Element*> relatedNeighborCalcElementsMap;
		vector<Element*> relatedNeighborCalcElementsVector;
		int invertedRhoElementsID;
		vector<ub::matrix < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>>dZdH;
		vector<ub::matrix < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>>dZdRho;

		vector<ub::vector < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>>dTdH;

		bool isInversionImpedance = false;
		bool isInversionTipper = false;
		Element* masterResistivityElement = nullptr;

		Eigen::Vector3cd lambda1 = Eigen::Vector3cd::Zero();
		Eigen::Vector3cd lambda2 = Eigen::Vector3cd::Zero();

		std::vector<int>nonZeroRowIndices;
		int maxResisIndexInSameRowsOfMatrix = -1;
		double maxResistivityInSameRowsOfMatrix = 0;

		double debug;

		double roughenMatrixUnit = 1.0;

		int impedanceObsID = -1;
		int tipperObsID = -1;

		Eigen::Matrix2d distortionMatrix;

		ub::matrix < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> dZdHOneOmega;
		ub::matrix < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> dZdRhoOneOmega;
		ub::vector < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> dTdHOneOmega;

		LocationData::LocationData* locationData;

		std::vector<double> dSvector; //order is -X,+X,-Y,+Y,-Z,+Z
		std::vector<Eigen::Vector3d> surfaceCenters;
		std::vector<Eigen::Vector3d> surfaceNormalVectors;
		std::vector<std::vector<Eigen::Vector3d>> surfaceParallelVectors;

		int numOfRelatedCalcVariables = 0;
		int numOfRelatedCalcVariablesAdjoint = 0;

		double factorForMatrixScaling = 1.0;

		int numOfCalcSurfaceForE = 6;

		bool calcGradDivOperationElement = false;

		double factorForDivCorr;

		void SearchChildrenElements(unordered_map<string, Element*>* elements, map<string, Element*>* elementsMap);
		void SearchRelatedCalcElements(unordered_map<string, Element*>* elements);
		void ClearHAndE();
		void ClearZ();
		void InitializeHAndEAndZ(int nOmega);
		void SetH(Eigen::VectorXcd& Hresult, int iOmega, int numOfCalcElements);
		void SetNeighborElements(unordered_map<string, Element*>* elements);
		virtual void CalcSurfaceResistivity(unordered_map<string, Element*>* elements, vector<Element*>* calcElementsVector, int numOfCalcElements);
		void CalcCenterCoeff(unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* resisCoeff, double coeff);
		void CalcCenterCoeff(unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::SparseMatrix<double, Eigen::RowMajor>* resisCoeff, double coeff);
		void CalcSurfaceRelatedResistivityCoeff(unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* resisCoeff, double coeff, Eigen::Vector3i pos);
		template<typename Scalar>
		void CalcInterpolationInElementCoeff(Eigen::Vector3i val, Eigen::Vector3d x0, unordered_map < string, Element* >* elements, int numChildElements, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>* resisCoeff);



		vector<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>>> rhoRotHdS;
		vector<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>>> rotHdS;
		vector<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>>> divHdS;
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* sumNCrossRhoRotHdS;
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* sumNCrossRhoRotHdSPerResistivityPerDxDyDz;
		vector<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<double, Eigen::RowMajor>>> nCrossRotHdS;
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> GetSumNCrossRhoRotHdS();
		virtual Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> CalcRotHdS(unordered_map < string, Element* >* elements, const int numOfCalcElements, Eigen::Vector3i pos); //must be calculated from elements of deeper layers.
		void SetTransitionZone(unordered_map < string, Element* >* elements,int numOfCalcElements);
		bool ResetTransitionZone(unordered_map < string, Element* >* elements, int numOfCalcElements, std::vector<Property::Property*>* propertiesVector);
		Eigen::SparseMatrix<double, Eigen::RowMajor> CalcInterpolatedVectorInElement(Eigen::Vector3i val, Eigen::Vector3d x0, unordered_map<string, Element*>* elements, int numChildElements);
		double CalcInterpolatedRhoInElement(Eigen::Vector3i val, Eigen::Vector3d x0, unordered_map < string, Element* >* elements, int numChildElements);

		double CalcCenterResistivity(unordered_map<string, Element*>* elements, int numChildElements);
		void CalcCenterVector(unordered_map<string, Element*>* elements, int numChildElements, Eigen::SparseMatrix<double, Eigen::RowMajor>& rowVec, double coeff=1.0);
		virtual void CalcNearestNeighborVectorEdge(Eigen::Vector3d x0, unordered_map<string, Element*>* elements, int numChildElements, Eigen::SparseMatrix<double, Eigen::RowMajor>* row, double* distance);
		std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>, double, string> CalcNearestNeighborVectorNode(Eigen::Vector3d x0, unordered_map<string, Element*>* elements, int numChildElements);
		virtual void  CalcSumNCrossRhoRotHdS(unordered_map<string, Element*>* elements, int numOfCalcElements);

		virtual void CalcE(Eigen::SparseMatrix<std::complex< double >, Eigen::ColMajor>* Hresult, unordered_map<string, Element*>* elements, int numOfCalcElements, int itr);
		virtual void CalcE(Eigen::VectorXcd* Hresult, unordered_map<string, Element*>* elements, int numOfCalcElements, int itr);
		virtual ub::matrix<kv::autodif<kv::complex<double>>> CalcDEDH(ub::vector<kv::autodif<kv::complex<double>>>* HresultTwoItr, std::vector<int>nonZeroRowIndices, unordered_map<string, Element*>* elements, int numOfCalcElements);
		virtual ub::matrix<kv::autodif<kv::complex<double>>> CalcDEDRho(ub::vector<kv::autodif<kv::complex<double>>>* rhoVec, ub::vector<kv::complex<double>>* Hresult, std::vector<int>nonZeroRowIndices, int numOfCalcElements);

		void CalcZ(unordered_map<string, Element*>* elements, int numOfCalcElements, int iOmega);
		Eigen::Matrix2cd CalcZ(vector < Eigen::Vector3cd> E, vector < Eigen::Vector3cd>H);

		virtual Eigen::Vector3cd CalcDEDH(int derID, int numOfCalcElements, unordered_map<string, Element*>* elements);
		void CalcDZDH(const ub::vector<kv::complex<double>>* HTwoItr, unordered_map<string, Element*>* elements, int numOfCalcElements, int iOmega);
		void CalcDZDRho(const ub::vector<kv::complex<double>>* rhoVecUb, const ub::vector<kv::complex<double>>* HresultTwoItr, const vector<Element*>* calcElementsVector, const int numOfCalcElements, const int iOmega);

		void CalcT(int iOmega);
		void CalcDTDH(int numOfCalcElements, int iOmega);


		void CalcLambdaDSumNCrossRhoRotHdSDRho(std::unordered_map<std::string, Element*>* elements, const ub::vector<complex<double>>* rhoVec,
			const vector<Eigen::VectorXcd>* HresultTwoItr, const vector<Element*>* calcElementsVector,
			const int numOfCalcElements, const int numOfInvertedResisElem, const Eigen::VectorXcd* lambdaEachOmega, Eigen::VectorXcd* lambdaDRDRho);

		void CalcSumDivergence(std::unordered_map<std::string, Element*>& elements, Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>& divergenceOperator, Eigen::SparseMatrix<double, Eigen::RowMajor>& divGradMatrix, Eigen::SparseMatrix<double, Eigen::RowMajor>& gradOperatorMatrix);
		void CalcDivergence(int i, std::unordered_map<std::string, Element*>& elements, Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>& divergenceOperator, Eigen::SparseMatrix<double, Eigen::RowMajor>& divGradMatrix, Eigen::SparseMatrix<double, Eigen::RowMajor>& gradOperatorMatrix);
		virtual void CalcDivHdS(Eigen::Vector3i pos, unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>& divHdS);
		virtual void CalcSumNDivHdS(unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* sumNDivHdS);

		void InitializeDistortionMatrix();
		
		double GetMaxResistivityInChildren(unordered_map<string, Element*>* elements);

		bool CheckThePointInside(Eigen::Vector3d p);
		void CalcSurfaceAndVolume();
		bool CheckThePointInside2D(Eigen::Vector3d p,int iSurf);

		void CalcNumOfRelatedCalcVariables();

		int CalcDeepestChildElementLayer(unordered_map<string, Element*>* elements);
		//void CalcNumOfRelatedCalcVariablesAdjoint(Eigen::VectorXi& reservedVectorAdjoint, unordered_map<string, Element*>& elements);
		virtual void  CalcOperatorSurface(Eigen::Vector3i pos, unordered_map<string, Element*>* elements, int numOfCalcElements, vector<Eigen::Triplet<double>>& nGradOperator,int& loc); 
		virtual void  CalcOperator(unordered_map<string, Element*>* elements, int numOfCalcElements, vector<Eigen::Triplet<double>>& sumNGradOperatorDs, int& locDivGrad,
			vector<Eigen::Triplet<double>>& divDsOperator, int& locDiv, vector<Eigen::Triplet<double>>& gradOperator, int& locgrad);
		virtual void CalcSumNDotHdSOperator(int numOfCalcElements, Eigen::SparseMatrix<double, Eigen::RowMajor>& sumNDotHdsOperator);

		void GetChildrenElements(unordered_map<string, Element*>* elements,vector<Element*>& elemVec);

	};
}