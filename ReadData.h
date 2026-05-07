/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#pragma once
#include <string.h>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <iostream>
#include "Element.h"
#include "Property.h"
#include "Boundary.h"
#include "InvSettings.h"
#include "ObsData.h"
#include "InitialResisData.h"
#include "InitialDistData.h"
#include "Output.h"
#include "LocationCalcSettings.h"
#include "LocationData.h"
#include "UncertaintyAnalysis.h"
#include "Node.h"
namespace ReadData
{
	class ReadData
	{

	public:
		ReadData();
		std::unordered_map<std::string, Node::Node*> nodes;
		std::vector< Node::Node*> nodesVector;
		std::unordered_map<std::string, Element::Element*> elements;
		std::vector< Element::Element*> elementsVector;
		std::unordered_map<int, Property::Property*> properties;
		std::vector< Property::Property*> propertiesVector;
		Boundary::Boundary* boundary;
		InvSettings::InvSettings* invSettings;
		UncertaintyAnalysis::UncertaintyAnalysis* uncertaintyAnalysis;
		LocationCalcSettings::LocationCalcSettings* locationCalcSettings;
		Output::Output* output;
		std::string obsFileName;
		std::vector<ObsData::ObsData*> obsData;
		std::vector<LocationData::LocationData*> locationData;
		std::vector<InitialResistivityData::InitialResistivityData*> initialResistivityData;
		std::vector<InitialDistortionData::InitialDistortionData*> initialDistortionData;
		double weightForModelConstraint;

		int lastObsDataID = 0;

		bool isUnstructuredElements = false;

		bool calcJustDataMisfit = false;

		bool isFFTSensitivityMode = false;
		double attenuation = 0.1;
		int Nx = 51;
		int Ny = 51;
		int Nz = 51;
		int Kx = 25;
		int Ky = 25;
		int Kz = 25;
		int cells_window = 3;
		int cells_window_fft_x = -1;
		int cells_window_fft_y = -1;
		int cells_window_fft_z = -1;
		int cells_window_out_x = -1;
		int cells_window_out_y = -1;
		int cells_window_out_z = -1;
		int numEnsemble = 100;
		int corr_cells_x = -1;
		int corr_cells_y = -1;
		int corr_cells_z = -1;
		double minX = -10000;
		double maxX = 10000;
		double minY = -10000;
		double maxY = 10000;
		double minZ = -10000;
		double maxZ = 10000;

		double epsR = 0.1;
		double epsT = 0.0;
		double eps_window = 0.;
		double eps_window_fft_x = -1.0;
		double eps_window_fft_y = -1.0;
		double eps_window_fft_z = -1.0;
		double fd_eps = 1.0e-1;
		double null_sv_ratio_thresh = 1.0e-2;
		int max_null_directions = 5;
		int min_null_directions = 3;
		int max_redraw_trials = 100;
		int candidate_pool_target = -1;
		double max_allowed_selected_cosine = 0.7071067811865476;
		bool first_direction_random = false;
		double confidenceLevel1 = 0.05; //For line search to seek the solution within these values
		double confidenceLevel2 = 0.01;
		double deltaRMSLevel1 = -1.0;
		double deltaRMSLevel2 = -1.0;
		double lambda = -1.0;
		bool isCheckResult = false;
		bool usePreviousResult = false;
		string orthogonalize = "gd";
		string replacedResistivityFile = "";

		void ReadFile(std::string modelFileName, bool forwardCalc);
		std::string AnalysisTag(std::string line);
		std::vector<std::string> split(std::string line);
		std::vector<std::string> readNext(std::ifstream* f);
		void AnalysisNodes(std::ifstream* f);
		void AnalysisElements(std::ifstream* f);
		void AnalysisProperties(std::ifstream* f);
		void AnalysisBoundary(std::ifstream* f);
		void AnalysisInvSettings(std::ifstream* f);
		void AnalysisLocationCalcSettings(std::ifstream* f);
		void AnalysisFFTSensitivityAnalysis(std::ifstream* f);
		void AnalysisUncertaintyAnalysisSettings(std::ifstream* f);
		void AnalysisObsDataFile(std::ifstream* f);
		void ReadImpedanceObsData(string filename);
		void ReadTipperObsData(string filename);
		void ReadInitialResistivityData(string resisFile);
		void ReadInitialDistortionData(string distFile);
		void ReadLocationDataFile(string filename);

	};
	class HashFromCoordToSize_t {
	public:
		size_t operator()(const Eigen::Vector3d& x) const;
	};
}