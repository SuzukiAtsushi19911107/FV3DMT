/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#pragma once
#include <fstream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>
using namespace std;
namespace Element {
	class Element;
}
namespace Output {
	class Output {
	public:
		int outputIteration = 10;

		void DebugOutput(int loop, double omega, std::unordered_map<string, Element::Element*>* elements);
		void VTKFileOputput(double omega, std::unordered_map<string, Element::Element*>* elements, string type = "rho",string outputFile="None");
		void RhoOutput(std::unordered_map<string, Element::Element*>* elements, std::string filename = "Rho.vtk",bool outputAll=false);
		void AppRhoOutputSurface(double omega, std::unordered_map<string, Element::Element*>* elements);
		void PhiOutputSurface(double omega, std::unordered_map<string, Element::Element*>* elements);
		void TipperOutputSurface(int iOmega,double omega, std::unordered_map<string, Element::Element*>* elements);
		void TxtOutputAppRho(double omega, std::unordered_map<string, Element::Element*>* elements);
		void TxtOutputResistivity(std::unordered_map<string, Element::Element*>* elements, std::string filename);
		void ImpedanceOutputSurface(vector<double> omegas, std::unordered_map<string, Element::Element*>* elements);
		void OutputObsCalcImpedance(vector<double> omegas, std::vector< Element::Element*>* obsPointsElements, std::string filename);
		void OutputObsCalcTipper(vector<double> omegas, std::vector< Element::Element*>* obsPointsElements, std::string filename);
		void TipperOutputSurface(vector<double> omegas, std::unordered_map<string, Element::Element*>* elements);
		void OutputDistortionMatrix(std::vector<Element::Element*>* obsImpedanceElements,double termVal, std::string filename = "distortion.txt");
		void OutputDistortionMatrixForRestart(std::vector<Element::Element*>* obsImpedanceElements, std::string filename = "distortionForRestart.txt");
		void VTKFileOputput(std::vector< Element::Element*>* calcElements, Eigen::VectorXd* values, string outputFile = "None");
		void VTKObsPointsFileOutput(std::vector< Element::Element*>* obsPointsElements,  string outputFile = "ObsPoints.vtk");
	};
}
