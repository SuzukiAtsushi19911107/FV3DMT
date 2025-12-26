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




namespace Property {
	class Property {
	public:
		Property();
		int ID = -1;
		double resistivity;
		enum types {NORMAL,AIR,FIXED,SEA}; //type 0:normal 1:air 2:fixed
		types type;

	};
}
