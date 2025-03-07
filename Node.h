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

namespace Node {
	class Node 
	{
	public:
		Node();
		~Node();
		Eigen::Vector3d x = Eigen::Vector3d::Zero();
		int	ID = -1;
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Ç±ÇÃÉ}ÉNÉçÇí«â¡


	};
}

