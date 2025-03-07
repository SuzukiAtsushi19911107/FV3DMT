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
#include "Boundary.h"

namespace Function {
	class Function {
	public:
		Function();
		int ID = -1;
		std::vector<double> variables;
		std::vector<double> values;
		double CalcValue(double variable);
	};
}
