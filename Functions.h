/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include<Eigen/SparseLU> 
#include <iostream>
#include <omp.h>
#include <bitset>
#include <sstream>
#include <iomanip>

namespace Element {
	class Element;
}
//ïKóvÇ»ä÷êîÇÇ‹Ç∆ÇﬂÇƒÇ®Ç¢ÇƒÇ®Ç≠
using namespace std;
namespace Functions {
	inline Eigen::Matrix3d inv3(Eigen::Matrix3d A) {

		Eigen::Matrix3d inv(3, 3);

		double a11 = A.coeff(0, 0);
		double a12 = A.coeff(0, 1);
		double a13 = A.coeff(0, 2);

		double a21 = A.coeff(1, 0);
		double a22 = A.coeff(1, 1);
		double a23 = A.coeff(1, 2);

		double a31 = A.coeff(2, 0);
		double a32 = A.coeff(2, 1);
		double a33 = A.coeff(2, 2);


		// calc inv
		double coeff = a11 * a22 * a33 +
			+a12 * a23 * a31
			+ a13 * a21 * a32
			- a13 * a22 * a31
			- a12 * a21 * a33
			- a11 * a23 * a32;
		coeff = 1.0 / coeff;

		inv.coeffRef(0, 0) = a22 * a33 - a23 * a32;
		inv.coeffRef(0, 1) = -(a12 * a33 - a13 * a32);
		inv.coeffRef(0, 2) = a12 * a23 - a13 * a22;

		inv.coeffRef(1, 0) = -(a21 * a33 - a23 * a31);
		inv.coeffRef(1, 1) = a11 * a33 - a13 * a31;
		inv.coeffRef(1, 2) = -(a11 * a23 - a13 * a21);

		inv.coeffRef(2, 0) = a21 * a32 - a22 * a31;
		inv.coeffRef(2, 1) = -(a11 * a32 - a12 * a31);
		inv.coeffRef(2, 2) = a11 * a22 - a12 * a21;

		inv = coeff * inv;

		return inv;
	}
	template<typename T2>
	inline ub::vector<T2> SolveLeastSquare(Eigen::MatrixXd A, ub::vector<T2> rhs, Eigen::MatrixXd W) {
		int numOfEqs = A.rows();
		int numOfVars = A.cols();
		if (numOfVars != 3) {
			cout << "Cannnot Use SolveLeastSquare() when variables are more than 3 componensts." << endl;
			exit(1);
		}
		ub::vector<T2> result(numOfVars);
		Eigen::MatrixXd At{ numOfVars, numOfEqs };
		for (int i = 0; i < numOfEqs; i++) {
			for (int j = 0; j < numOfVars; j++) {
				At.coeffRef(j, i) = A.coeff(i, j);
			}
		}
		Eigen::MatrixXd AtA{ 3, 3 };
		AtA = At * W * A;

		Eigen::MatrixXd inv(3, 3);

		double a11 = AtA.coeff(0, 0);
		double a12 = AtA.coeff(0, 1);
		double a13 = AtA.coeff(0, 2);

		double a21 = AtA.coeff(1, 0);
		double a22 = AtA.coeff(1, 1);
		double a23 = AtA.coeff(1, 2);

		double a31 = AtA.coeff(2, 0);
		double a32 = AtA.coeff(2, 1);
		double a33 = AtA.coeff(2, 2);


		// calc inv
		double coeff = a11 * a22 * a33 +
			+a12 * a23 * a31
			+ a13 * a21 * a32
			- a13 * a22 * a31
			- a12 * a21 * a33
			- a11 * a23 * a32;
		coeff = 1.0 / coeff;

		inv.coeffRef(0, 0) = a22 * a33 - a23 * a32;
		inv.coeffRef(0, 1) = -(a12 * a33 - a13 * a32);
		inv.coeffRef(0, 2) = a12 * a23 - a13 * a22;

		inv.coeffRef(1, 0) = -(a21 * a33 - a23 * a31);
		inv.coeffRef(1, 1) = a11 * a33 - a13 * a31;
		inv.coeffRef(1, 2) = -(a11 * a23 - a13 * a21);

		inv.coeffRef(2, 0) = a21 * a32 - a22 * a31;
		inv.coeffRef(2, 1) = -(a11 * a32 - a12 * a31);
		inv.coeffRef(2, 2) = a11 * a22 - a12 * a21;

		inv = coeff * inv;

		//make AtA.inv()*At*W
		inv = inv * At*W;

		//calc result
		for (int i = 0; i < 3; i++) {
			T2 a;
			for (int j = 0; j < numOfEqs; j++) {
				a += inv.coeff(i, j) * rhs(j);
			}
			result(i) = a;
		}

		return result;
	}
	template <typename T>
	inline void SetAtoResultCoef1DotBPlusCoef2DotC
	(Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* A,
		const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* B,
		const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* C,
		const T coef1 = 1.0, const T coef2 = 1.0) {
		//A=B+C
		//This function is for preventing Eigen from allocate full matrix size
		//when multiplying const value
		//this function assumes that A is reserved enough and noncompressed mode
		A->setZero();
		int rows = B->rows();
		int cols = B->cols();
		int numOfNonzeros = B->nonZeros();
		for (int j = 0; j < B->outerSize(); ++j) {
			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*B, j); it; ++it) {
				int iRow = it.row();
				int iCol = it.col();
				A->coeffRef(iRow, iCol) += coef1 * B->coeff(iRow, iCol);
			}
		}
		rows = C->rows();
		cols = C->cols();
		numOfNonzeros = C->nonZeros();
		for (int j = 0; j < C->outerSize(); ++j) {
			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*C, j); it; ++it) {
				int iRow = it.row();
				int iCol = it.col();
				A->coeffRef(iRow, iCol) += coef2 * C->coeff(iRow, iCol);
			}
		}
		return;
	}
	template <typename T>
	inline void PlusEqual(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* A, const Eigen::SparseMatrix<double, Eigen::RowMajor>* B, const T cons = 1) {
		//This function is for preventing Eigen from allocate full matrix size
		//when multiplying const value
		//this function assumes that A is reserved enough and noncompressed mode
		int rows = B->rows();
		int cols = B->cols();
		int numOfNonzeros = B->nonZeros();
		for (int j = 0; j < B->outerSize(); ++j) {
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(*B, j); it; ++it) {
				int iRow = it.row();
				int iCol = it.col();
				A->coeffRef(iRow, iCol).real(A->coeffRef(iRow, iCol).real() + cons * B->coeff(iRow, iCol));
			}
		}
		return;
	}
	template <typename T>
	inline void PlusEqual(Eigen::SparseMatrix<double, Eigen::RowMajor>* A, const Eigen::SparseMatrix<double, Eigen::RowMajor>* B, const T cons = 1) {
		//This function is for preventing Eigen from allocate full matrix size
		//when multiplying const value
		//this function assumes that A is reserved enough and noncompressed mode
		int rows = B->rows();
		int cols = B->cols();
		int numOfNonzeros = B->nonZeros();
		for (int j = 0; j < B->outerSize(); ++j) {
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(*B, j); it; ++it) {
				int iRow = it.row();
				int iCol = it.col();
				A->coeffRef(iRow, iCol) += cons * B->coeff(iRow, iCol);
			}
		}
		return;
	}
	template <typename T>
	inline void PlusEqual(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* A, const Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* B, const T cons = 1) {
		//This function is for preventing Eigen from allocate full matrix size
		//when multiplying const value
		//this function assumes that A is reserved enough and noncompressed mode
		int rows = B->rows();
		int cols = B->cols();
		int numOfNonzeros = B->nonZeros();
		for (int j = 0; j < B->outerSize(); ++j) {
			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*B, j); it; ++it) {
				int iRow = it.row();
				int iCol = it.col();
				A->coeffRef(iRow, iCol) += cons * B->coeff(iRow, iCol);
			}
		}
		return;
	}

	//DotConst is alternative for not using assign_sparse_to_sparse.
	//bacause assign_sparse_to_sparse once reserves large memory (e.g.temp.reserve((std::min)(src.rows()*src.cols(), (std::max)(src.rows(),src.cols())*2));) 
	//This makes program so slow.
	template <typename T>
	inline void DotConstSelf(T cons, Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* R) {
		//This function is for preventing Eigen from allocate full matrix size
		//when multiplying const value
		for (int j = 0; j < R->outerSize(); ++j) {
			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*R, j); it; ++it) {
				int iRow = it.row();
				int iCol = it.col();
				R->coeffRef(iRow, iCol) = cons * R->coeff(iRow, iCol);
			}
		}
		return;
	}
	template <typename T>
	inline void DotConstSelf(T cons, Eigen::SparseMatrix<complex<double>, Eigen::ColMajor>* R) {
		//This function is for preventing Eigen from allocate full matrix size
		//when multiplying const value
		for (int j = 0; j < R->outerSize(); ++j) {
			for (Eigen::SparseMatrix<complex<double>, Eigen::ColMajor>::InnerIterator it(*R, j); it; ++it) {
				int iRow = it.row();
				int iCol = it.col();
				R->coeffRef(iRow, iCol) = cons * R->coeff(iRow, iCol);
			}
		}
		return;
	}
	template <typename T>
	inline void DotConstSelf(T cons, Eigen::SparseMatrix<double, Eigen::RowMajor>* R) {
		//This function is for preventing Eigen from allocate full matrix size
		//when multiplying const value
		for (int j = 0; j < R->outerSize(); ++j) {
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(*R, j); it; ++it) {
				int iRow = it.row();
				int iCol = it.col();
				R->coeffRef(iRow, iCol) = cons * R->coeff(iRow, iCol);
			}
		}
		return;
	}

	template <typename T>
	inline Eigen::SparseMatrix<double, Eigen::RowMajor> DotConst(T cons,const Eigen::SparseMatrix<double, Eigen::RowMajor>* R) {
		//This function is for preventing Eigen from allocate full matrix size
		//when multiplying const value
		int rows = R->rows();
		int cols = R->cols();
		int numOfNonzeros = R->nonZeros();
		Eigen::SparseMatrix<double, Eigen::RowMajor> C{ rows,cols };
		if (R->isCompressed()) {
			C.reserve(numOfNonzeros);
		}
		else {
			C.reserve(Eigen::VectorXi::Constant(rows, numOfNonzeros)); //too large if numOfNonzeros is large
		}
		for (int j = 0; j < R->outerSize(); ++j) {
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(*R, j); it; ++it) {
				int iRow = it.row();
				int iCol = it.col();
				C.coeffRef(iRow, iCol) = cons * R->coeff(iRow, iCol);
			}
		}
		return C;
	}
	template <typename T>
	inline Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> DotConst(T cons,const Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* R) {
		//This function is for preventing Eigen from allocate full matrix size
		//when multiplying const value
		int rows = R->rows();
		int cols = R->cols();
		int numOfNonzeros = R->nonZeros();
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> C{ rows,cols };
		if (R->isCompressed()) {
			C.reserve(numOfNonzeros);
		}
		else {
			C.reserve(Eigen::VectorXi::Constant(rows, numOfNonzeros)); //too large if numOfNonzeros is large
		}
		for (int j = 0; j < R->outerSize(); ++j) {
			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*R, j); it; ++it) {
				int iRow = it.row();
				int iCol = it.col();
				C.coeffRef(iRow, iCol) = cons * R->coeff(iRow, iCol);
			}
		}
		return C;
	}
	inline vector<double> CalcWeight(vector<Eigen::Vector3d> x, Eigen::Vector3d x0) {
		int length = x.size();
		double wSum = 0;
		vector<double> w;
		for (int i = 0; i < length; i++) {
			w.push_back(0);
			w[i] = 1 / (x[i] - x0).norm();
			wSum += w[i];
		}
		for (int i = 0; i < length; i++) {
			w[i] = w[i] / wSum;
		}
		return w;
	}
	inline string GetBinaryValue(int i, int j) {
		if (i > 2 && j > 2) {
			cout << "Error in GetBinaryValue" << endl;
		}
		string str = to_string(j) + to_string(i);
		return str;
	}
	inline int binary(int bina) {
		int ans = 0;
		for (int i = 0; bina > 0; i++)
		{
			ans = ans + (bina % 2)*pow(10, i);
			bina = bina / 2;
		}
		return ans;
	}
	inline std::string GetNeighborElement(std::unordered_map<std::string, Element::Element*> *elements, Element::Element* element, Eigen::Vector3i val,int nx,int ny,int nz) {
		if (val.coeff(0) > 1 || val.coeff(1) > 1 || val.coeff(2) > 1) {
			cout << "GetNeighborElement Function Could Not Seek Except Neighbor." << endl;
			exit(1);
		}
		int ipos = (val.coeff(0) + 1) + 3 * (val.coeff(1) + 1) + 9 * (val.coeff(2) + 1);
		if (ipos >= 0 && ipos < 27 && element->alreadyFoundNeighborID[ipos]!="NOT_FOUND") {
			return element->alreadyFoundNeighborID[ipos]; 
		}

		int length = (element->layer + 1) * 2;
		int xVal = 0;
		int yVal = 0;
		for (int i = 1; i < element->layer + 1; i++) {
			string str = element->ID.substr(9 + i * 2 + 1,1); //9 is nx,ny,nz place
			xVal += (stoi(str))*(pow(2,(element->layer - i)));

			str = element->ID.substr(9 + i * 2 , 1);
			yVal += (stoi(str))*(pow(2, (element->layer - i)));

		}
		int xAfter = xVal + val.coeff(0);
		int yAfter = yVal + val.coeff(1);
		
		int IDZAfter = element->IDZ;
		int IDYAfter = element->IDY;
		int IDXAfter = element->IDX;


		if (xAfter < 0) {
			if (element->IDX == 0) {
				return "-X_BOUNDARY";
			}
			else {
				IDXAfter -= 1;
				xAfter = 0;
				for (int i = 1; i < element->layer + 1; i++) {
					xAfter += 1*(pow(2, (element->layer - i)));
				}
			}
		}
		if (yAfter < 0) {
			if (element->IDY == 0) {
				return "-Y_BOUNDARY";
			}
			else {
				IDYAfter -= 1;
				yAfter = 0;
				for (int i = 1; i < element->layer + 1; i++) {
					yAfter += 1 * (pow(2, (element->layer - i)));
				}
			}
		}

		int quot = std::pow(10, 6);
		int rankX = 0;
		int tmp = 0;
		while (quot >= 0) {
			quot = xAfter /(int) std::pow(2,(double) tmp);
			if (quot > 0) {
				rankX = tmp+1;
			}
			else {
				break;
			}
			tmp += 1;
		}
		quot = std::pow(10, 6);
		int rankY = 0;
		tmp = 0;
		while (quot >= 0) {
			quot = yAfter / (int)std::pow(2, (double)tmp);
			if (quot > 0) {
				rankY = tmp+1;
			}
			else {
				break;
			}
			tmp += 1;
		}

		if (rankX > element->layer) {
			if (element->IDX == nx - 1) {
				return "+X_BOUNDARY";
			}
			else {
				IDXAfter += 1;
				xAfter = 0;
			}
		}
		if (rankY > element->layer) {
			if (element->IDY == ny - 1) {
				return "+Y_BOUNDARY";
			}
			else {
				IDYAfter += 1;
				yAfter = 0;
			}
		}
		if (val.coeff(2) == -1) {
			IDZAfter -= 1;
			if (IDZAfter < 0) {
				return "-Z_BOUNDARY";
			}
		}
		else if (val.coeff(2) == 1) {
			IDZAfter += 1;
			if (IDZAfter > nz - 1) {
				return "+Z_BOUNDARY";
			}
		}


		std::string xBin = std::to_string(binary(xAfter));
		std::string yBin = std::to_string(binary(yAfter));

		for (int i = xBin.length(); i < element->layer; i++) {
			xBin = "0" + xBin;
		}
		for (int i = yBin.length(); i < element->layer; i++) {
			yBin = "0" + yBin;
		}

		std::stringstream ssx;
		std::ostringstream ossx;
		ossx << IDXAfter;
		ssx << std::setw(3) << std::setfill('0') << ossx.str();
		std::stringstream ssy;
		std::ostringstream ossy;
		ossy << IDYAfter;
		ssy << std::setw(3) << std::setfill('0') << ossy.str();
		std::stringstream ssz;
		std::ostringstream ossz;
		ossz << IDZAfter;
		ssz << std::setw(3) << std::setfill('0') << ossz.str();
		std::string neighborID = ssz.str() + ssy.str() + ssx.str() + "00";
		for (int i = 0; i < element->layer; i++) {
			neighborID += yBin.substr(i,1) + xBin.substr(i,1);
		}
		if (elements->count(neighborID) == 1) {
			int ipos = (val.coeff(0)+1) + 3 * (val.coeff(1)+1) + 9 * (val.coeff(2)+1);
			if (ipos >= 0 && ipos < 27) {
				element->alreadyFoundNeighborID[ipos] = neighborID; //ìoò^
			}
			return neighborID;
		}
		else {
			std::string stmp = neighborID;
			int length = neighborID.length()-9;
			int layer = (int)length / 2;
			for (int i = 1; i < layer; i++) {
				stmp = neighborID.substr(0,9 + length - 2 * i);
				if (elements->count(stmp) == 1) {
					int ipos = (val.coeff(0) + 1) + 3 * (val.coeff(1) + 1) + 9 * (val.coeff(2) + 1);
					if (ipos >= 0 && ipos < 27) {
						element->alreadyFoundNeighborID[ipos] = stmp; //ìoò^
					}
					return stmp;
				}
			}
		}
		cout << "Could not find neighbor Element." << endl;
		exit(1);

	}
	inline std::string GetNeighborElement(std::unordered_map<std::string, Element::Element*>* elements, string ID, int layer, Eigen::Vector3i val,int nx,int ny,int nz) {
		if (val.coeff(0) > 1 || val.coeff(1) > 1 || val.coeff(2) > 1) {
			cout << "GetNeighborElement Function Could Not Seek Except Neighbor." << endl;
			exit(1);
		}
		int xVal = 0;
		int yVal = 0;
		for (int i = 1; i <layer + 1; i++) {
			string str = ID.substr(9 + i * 2 + 1, 1);//9 is nx,ny,nz place
			xVal += (stoi(str)) * (pow(2, (layer - i)));

			str = ID.substr(9 + i * 2, 1);
			yVal += (stoi(str)) * (pow(2, (layer - i)));

		}
		int xAfter = xVal + val.coeff(0);
		int yAfter = yVal + val.coeff(1);

		int IDZAfter = stoi(ID.substr(0, 3));
		int IDYAfter = stoi(ID.substr(3, 3));
		int IDXAfter = stoi(ID.substr(6, 3));

		if (xAfter < 0) {
			if (IDXAfter == 0) {
				return "-X_BOUNDARY";
			}
			else {
				IDXAfter -= 1;
				xAfter = 0;
				for (int i = 1; i < layer + 1; i++) {
					xAfter += 1 * (pow(2, (layer - i)));
				}
			}
		}
		if (yAfter < 0) {
			if (IDYAfter == 0) {
				return "-Y_BOUNDARY";
			}
			else {
				IDYAfter -= 1;
				yAfter = 0;
				for (int i = 1; i < layer + 1; i++) {
					yAfter += 1 * (pow(2, (layer - i)));
				}
			}
		}
		if (val.coeff(2) == -1) {
			IDZAfter -= 1;
			if (IDZAfter < 0) {
				return "-Z_BOUNDARY";
			}
		}
		else if (val.coeff(2) == 1) {
			IDZAfter += 1;
			if (IDZAfter > nz - 1) {
				return "+Z_BOUNDARY";
			}
		}

		int quot = std::pow(10, 6);
		int rankX = 0;
		int tmp = 0;
		while (quot >= 0) {
			quot = xAfter / (int)std::pow(2, (double)tmp);
			if (quot > 0) {
				rankX = tmp + 1;
			}
			else {
				break;
			}
			tmp += 1;
		}
		quot = std::pow(10, 6);
		int rankY = 0;
		tmp = 0;
		while (quot >= 0) {
			quot = yAfter / (int)std::pow(2, (double)tmp);
			if (quot > 0) {
				rankY = tmp + 1;
			}
			else {
				break;
			}
			tmp += 1;
		}

		
		if (rankX > layer) {
			if (IDXAfter== nx - 1) {
				return "+X_BOUNDARY";
			}
			else {
				IDXAfter += 1;
				xAfter = 0;
			}
		}
		if (rankY > layer) {
			if (IDYAfter == ny - 1) {
				return "+Y_BOUNDARY";
			}
			else {
				IDYAfter += 1;
				yAfter = 0;
			}
		}


		std::string xBin = std::to_string(binary(xAfter));
		std::string yBin = std::to_string(binary(yAfter));
		for (int i = xBin.length(); i < layer; i++) {
			xBin = "0" + xBin;
		}
		for (int i = yBin.length(); i <layer; i++) {
			yBin = "0" + yBin;
		}

		std::stringstream ssx;
		std::ostringstream ossx;
		ossx << IDXAfter;
		ssx << std::setw(3) << std::setfill('0') << ossx.str();
		std::stringstream ssy;
		std::ostringstream ossy;
		ossy << IDYAfter;
		ssy << std::setw(3) << std::setfill('0') << ossy.str();
		std::stringstream ssz;
		std::ostringstream ossz;
		ossz << IDZAfter;
		ssz << std::setw(3) << std::setfill('0') << ossz.str();
		std::string neighborID = ssz.str() + ssy.str() + ssx.str() + "00";
		for (int i = 0; i < layer; i++) {
			neighborID += yBin.substr(i, 1) + xBin.substr(i, 1);
		}
		if (elements->count(neighborID) == 1) {
			return neighborID;
		}
		else {
			std::string stmp = neighborID;
			int length = neighborID.length() - 9;
			int layer = (int)length / 2;
			for (int i = 1; i < layer; i++) {
				stmp = neighborID.substr(0,9 + length - 2 * i);
				if (elements->count(stmp) == 1) {
					return stmp;
				}
			}
		}
		cout << "Could not find neighbor Element." << endl;
		exit(1);

	}
	inline std::string GetVirturalNeighborElement(std::unordered_map<std::string, Element::Element*> *elements, std::string ID,int layer, Eigen::Vector3i val,int nx,int ny,int nz) {
		if (val.coeff(0) > 1 || val.coeff(1) > 1 || val.coeff(2) > 1) {
			cout << "GetNeighborElement Function Could Not Seek Except Neighbor." << endl;
			exit(1);
		}
		int xVal = 0;
		int yVal = 0;

		for (int i = 1; i < layer + 1; i++) {
			string str = ID.substr(9 + i * 2 + 1, 1);//9 is nx,ny,nz place
			xVal += (stoi(str)) * (pow(2, (layer - i)));

			str = ID.substr(9 + i * 2 , 1);
			yVal += (stoi(str)) * (pow(2, (layer - i)));



		}
		int xAfter = xVal + val.coeff(0);
		int yAfter = yVal + val.coeff(1);


		int IDZAfter = stoi(ID.substr(0, 3));
		int IDYAfter = stoi(ID.substr(3, 3));
		int IDXAfter = stoi(ID.substr(6, 3));

		if (xAfter < 0) {
			if (IDXAfter == 0) {
				return "-X_BOUNDARY";
			}
			else {
				IDXAfter -= 1;
				xAfter = 0;
				for (int i = 1; i < layer + 1; i++) {
					xAfter += 1 * (pow(2, (layer - i)));
				}
			}
		}
		if (yAfter < 0) {
			if (IDYAfter == 0) {
				return "-Y_BOUNDARY";
			}
			else {
				IDYAfter -= 1;
				yAfter = 0;
				for (int i = 1; i < layer + 1; i++) {
					yAfter += 1 * (pow(2, (layer - i)));
				}
			}
		}
		if (val.coeff(2) == -1) {
			IDZAfter -= 1;
			if (IDZAfter < 0) {
				return "-Z_BOUNDARY";
			}
		}
		else if (val.coeff(2) == 1) {
			IDZAfter += 1;
			if (IDZAfter > nz - 1) {
				return "+Z_BOUNDARY";
			}
		}

		int quot = std::pow(10, 6);
		int rankX = 0;
		int tmp = 0;
		while (quot >= 0) {
			quot = xAfter / (int)std::pow(2, (double)tmp);
			if (quot > 0) {
				rankX = tmp + 1;
			}
			else {
				break;
			}
			tmp += 1;
		}
		quot = std::pow(10, 6);
		int rankY = 0;
		tmp = 0;
		while (quot >= 0) {
			quot = yAfter / (int)std::pow(2, (double)tmp);
			if (quot > 0) {
				rankY = tmp + 1;
			}
			else {
				break;
			}
			tmp += 1;
		}

		if (rankX > layer) {
			if (IDXAfter == nx - 1) {
				return "+X_BOUNDARY";
			}
			else {
				IDXAfter += 1;
				xAfter = 0;
			}
		}
		if (rankY > layer) {
			if (IDYAfter == ny - 1) {
				return "+Y_BOUNDARY";
			}
			else {
				IDYAfter += 1;
				yAfter = 0;
			}
		}


		std::string xBin = std::to_string(binary(xAfter));
		std::string yBin = std::to_string(binary(yAfter));
		for (int i = xBin.length(); i < layer; i++) {
			xBin = "0" + xBin;
		}
		for (int i = yBin.length(); i < layer; i++) {
			yBin = "0" + yBin;
		}

		std::stringstream ssx;
		std::ostringstream ossx;
		ossx << IDXAfter;
		ssx << std::setw(3) << std::setfill('0') << ossx.str();
		std::stringstream ssy;
		std::ostringstream ossy;
		ossy << IDYAfter;
		ssy << std::setw(3) << std::setfill('0') << ossy.str();
		std::stringstream ssz;
		std::ostringstream ossz;
		ossz << IDZAfter;
		ssz << std::setw(3) << std::setfill('0') << ossz.str();
		std::string neighborID = ssz.str() + ssy.str() + ssx.str()   + "00";
		for (int i = 0; i < layer; i++) {
			neighborID += yBin.substr(i, 1) + xBin.substr(i, 1);
		}
		return neighborID;
	}

};
