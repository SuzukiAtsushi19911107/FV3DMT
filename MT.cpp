/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#pragma once
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"
#include <sys/stat.h>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include "ReadData.h"
#include <time.h>
#include "Analysis.h"
#include <kv/complex.hpp>
#include <boost/version.hpp>
#include <omp.h>


//template <class T> ub::vector<T> f(double b, ub::vector<T>& x,double a) {
//	ub::vector<T> y(2);
//
//	y(0) = 2. * x(0) * x(0) * x(1) - 1.;
//	y(1) = x(0) + 0.5 * x(1) * x(1) - 2.;
//	y(0) = y(0)*a*b;
//	return y;
//}

inline double booth_fn(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* opt_data)
{
	double x_1 = vals_inp(0);
	double x_2 = vals_inp(1);

	double obj_val = std::pow(x_1 + 2 * x_2 - 7.0, 2) + std::pow(2 * x_1 + x_2 - 5.0, 2);
	//
	if (grad_out) {
		(*grad_out)(0) = 2 * (x_1 + 2 * x_2 - 7.0) + 2 * (2 * x_1 + x_2 - 5.0) * 2;
		(*grad_out)(1) = 2 * (x_1 + 2 * x_2 - 7.0) * 2 + 2 * (2 * x_1 + x_2 - 5.0);
	}
	//
	return obj_val;
}

int main(int args, char* argv[])
{

	

	//ub::vector<kv::complex<double>> v1, v2;
	//ub::vector<kv::autodif<kv::complex<double>>> va1, va2;
	//ub::matrix<kv::complex<double>> m;

	//v1.resize(2);
	//v1(0) = 5.; v1(1) = 6.;

	//va1 = kv::autodif< kv::complex<double >>::init(v1);

	//va2 = f(2.0,va1,0.5);

	//kv::autodif<kv::complex<double>>::split(va2, v2, m);

	//std::cout << v2 << "\n"; // f(5, 6)
	//std::cout << m << "\n"; // Jacobian matrix
	bool forwardCalc = false;
	time_t start_t = time(NULL);
	if (argv[1] == NULL) {
		std::cout << "Calculation Data Must be Writen In argv!!" << std::endl;
		exit(1);
	}
	
	if (argv[2] != NULL) {
		forwardCalc = true;
		std::cout<<"Forward Calculation Will Start!!!!"<<std::endl;
	}
	
	std::string modelFileName = argv[1];

	struct stat st;
	const char* file = modelFileName.c_str();
	int ret = stat(file, &st);
	if (0 != ret) {
		std::cout << "Calculation Data Does Not Exist!!" << std::endl;
		exit(1);
	}

	ReadData::ReadData* readData = new ReadData::ReadData();
	readData->ReadFile(modelFileName, forwardCalc); //読んだデータのクラスを作成
	Analysis::Analysis*  analysis=new Analysis::Analysis{ readData }; //実際に解析をするクラスを作成

	//omp_set_num_threads(std::min(omp_get_max_threads(), int(analysis->boundary->omega.size())));
	std::cout << "omp_get_max_threads:" << omp_get_max_threads() << std::endl;

	//if (omp_get_max_threads() >= 2 * analysis->boundary->omega.size()) {
	//	omp_set_nested(1);
	//}
	//else {
		omp_set_nested(0);
	//}

	if (forwardCalc == true) {
		cout << "Calculation Type:Forward Calculation Mode" << endl;
		analysis->RunAnalysis(); //解析実行
	}
	else if (analysis->uncertaintyAnalysis->isCalc == true) {
		cout << "Calculation Type:Uncertainty Analysis Mode" << endl;
		analysis->RunUncertaintyAnalysis();
	}
	else if (analysis->locationCalcSettings->isCalc == true) {
		cout << "Calculation Type:Location Analysis Mode" << endl;
		analysis->RunLocationCalc();
	}
	else {
		cout << "Calculation Type:Optimization Mode" << endl;
		analysis->RunOptimize();
	}

	time_t end_t = time(NULL);
	std::cout << "Total CalcTime:"<<(end_t-start_t)/60<<" minute" << std::endl;


}

