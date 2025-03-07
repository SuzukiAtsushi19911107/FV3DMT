/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#pragma once
#include <iostream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
namespace ConstantValues {
	const double mu = 1.256637*std::pow(10, -6);
	const std::complex<double> imag(0.0, 1.0);
	const double pi = 3.141526535;
}