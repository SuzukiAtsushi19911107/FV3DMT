/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#pragma once
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include "ReadData.h"
#include "Property.h"
#include "InvSettings.h"
#include <string.h>
#include <sys/stat.h>
#include "Node.h"
#include "UnstructuredElement.h"
#include <iostream>
#include <sstream>

ReadData::ReadData::ReadData() {
	invSettings = new InvSettings::InvSettings(0);
	uncertaintyAnalysis = new UncertaintyAnalysis::UncertaintyAnalysis();
	locationCalcSettings = new LocationCalcSettings::LocationCalcSettings();
	output = new Output::Output();
}
#include <iostream>
#include <string>
#include <vector>

std::vector<std::string> ReadData::ReadData::split(std::string str) {
	std::vector<std::string> result;
	std::string subStr;
	std::vector<char> del;
	del.push_back(' ');
	del.push_back ('\t');
	del.push_back ('\n');
	del.push_back ('\r');
	for (const char c : str) {
		bool delFlag = false;
		for (auto itr = del.begin(); itr != del.end(); ++itr) {
			if (c == *itr) {
				delFlag = true;
			}
		}
		//if (c == del) {
		if (delFlag){
			if (!subStr.empty()) {
				result.push_back(subStr);
				subStr.clear();
			}
		}
		else {
			subStr += c;
		}
	}
	
	if (!subStr.empty()) {
		result.push_back(subStr);
	}
	return result;
}
//
//std::vector<std::string> ReadData::ReadData::split(std::string* line) {
//
//	//std::string str;
//	std::vector<std::string> returnLine;
//	std::vector<char> del;
//	del.push_back(' ');
//	del.push_back('\t');
//	del.push_back('\n');
//	std::vector<int> splitPoint;
//	for (int i = 0; i < line->length(); i++) {
//		//std::cout << "test" << std::endl;
//		std::string tmpLine = *line;
//		for (auto iDel = del.begin(); iDel != del.end(); iDel++) {
//			if (tmpLine[i] == *iDel) {
//				splitPoint.push_back(i);
//			}
//		}
//	}
//	std::sort(splitPoint.begin(), splitPoint.end());//昇順ソート
//	int startPoint = 0;
//	//std::vector<std::string> str;
//	for (int i = 0; i < splitPoint.size(); i++) {
//		returnLine.push_back("");
//		if (i == 0) {
//			if (splitPoint[i] > 0) {
//				for (int j = 0; j < splitPoint[i]; j++) {
//					returnLine[i] += line[j];
//				}
//			}
//		}
//		else if (i == splitPoint.size() - 1) {
//			if (splitPoint[i] < line->length() - 1) {
//				for (int j = splitPoint[i]+1; j < line->length(); j++) {
//					returnLine[i] += line[j];
//				}
//			}
//		}
//		else {
//			if (splitPoint[i] != splitPoint[i + 1] - 1) {
//				for (int j = splitPoint[i] + 1; j < splitPoint[i + 1]; j++){
//					returnLine[i] += line[j];
//				}
//			}
//		}
//
//		//returnLine.push_back(str);
//	}
//	//for (int i = 0; i < line.length(); i++) {
//	//	if (line[i] == ' ' || line[i] == '\t' || line[i] == '\n') {
//	//		if (!str.empty()) {
//	//			returnLine.push_back(str);
//	//			str.clear();
//	//		}
//	//	}
//	//	else {
//	//		str += line[i];
//	//		
//	//		
//	//	}
//	//}
//	
//	//returnLine.push_back(str);
//	//std::cout << returnLine.size()<< std::endl;
//	return returnLine;
//}
std::string ReadData::ReadData::AnalysisTag(std::string tmpLine) {
	std::vector<std::string> line;
	line= split(tmpLine);
	std::string tag;
	for (auto itr = line.begin(); itr != line.end(); ++itr) {
		std::string word = *itr;
		const char* tmpWord = word.c_str();

		std::string compare1{ word[0] };
		std::string compare2{ "#" }; //For judgement if Comment or not.
		if (strcasecmp("Elements", tmpWord) == 0) { //大文字小文字区別せずタグがElementsの場合
			tag = "ELEMENTS";
			break;
		}
		else if (strcasecmp("PROPERTIES", tmpWord) == 0) {  //大文字小文字区別せずタグがPropertiesの場合
			tag = "PROPERTIES";
			break;
		}
		else if (strcasecmp("BOUNDARY", tmpWord) == 0) { // 大文字小文字区別せずタグがCONDITIONSの場合
			tag = "BOUNDARY";
			break;
		}
		else if (strcasecmp("OBSDATAFILE", tmpWord) == 0) { // 大文字小文字区別せずタグがCONDITIONSの場合
			tag = "OBSDATAFILE";
			break;
		}
		else if (strcasecmp("INVSETTINGS", tmpWord) == 0) { // 大文字小文字区別せずタグがINVSETTINGSの場合
			tag = "INVSETTINGS";
			break;
		}
		else if (strcasecmp("LocationCalcSettings", tmpWord) == 0) { // 大文字小文字区別せずタグがINVSETTINGSの場合
			tag = "LocationCalcSettings";
			break;
		}
		else if (strcasecmp("FFTSensitivityAnalysis", tmpWord) == 0) { // 大文字小文字区別せずタグがINVSETTINGSの場合
			tag = "FFTSensitivityAnalysis";
			break;
		}
		else if (strcasecmp("UncertaintyAnalysisSettings", tmpWord) == 0) { // 大文字小文字区別せずタグがINVSETTINGSの場合
			tag = "UncertaintyAnalysisSettings";
			break;
		}
		else if (strcasecmp("END_DATA", tmpWord) == 0) { // 大文字小文字区別せずタグがENDの場合
			tag = "END_DATA";
			break;
		}
		//else if (strcasecmp("#", &tmpWord[0]) == 0) {
		else if(compare1==compare2){
			tag = "COMMENT";
			break;
		}
		else {
			std::cout << "No Match Tag In Your Data" << std::endl;
			exit(1);
		}
	}


	return tag;
}



void ReadData::ReadData::ReadFile(std::string modelFileName, bool forwardCalc) {
	
	std::ifstream f(modelFileName);

	std::string line;
	bool endOfFile = false;
	std::getline(f, line);

	while (!f.eof()) {
		std::string tag = AnalysisTag(line);
		const char* tmpTag = tag.c_str();
		if (strcasecmp("ELEMENTS", tmpTag) == 0) {
			AnalysisElements(&f);
			std::getline(f, line);
			continue;
		}
		if (strcasecmp("PROPERTIES", tmpTag) == 0) {
			AnalysisProperties(&f);
			std::getline(f, line);
			continue;
		}
		if (strcasecmp("BOUNDARY", tmpTag) == 0) {
			AnalysisBoundary(&f);
			std::getline(f, line);
			continue;
		}
		if (strcasecmp("INVSETTINGS", tmpTag) == 0) {
			AnalysisInvSettings(&f);
			std::getline(f, line);
			continue;
		}
		if (strcasecmp("LocationCalcSettings", tmpTag) == 0) {
			AnalysisLocationCalcSettings(&f);
			std::getline(f, line);
			continue;
		}
		if (strcasecmp("FFTSensitivityAnalysis", tmpTag) == 0) {
			AnalysisFFTSensitivityAnalysis(&f);
			std::getline(f, line);
			continue;
		}
		if (strcasecmp("UncertaintyAnalysisSettings", tmpTag) == 0) {
			AnalysisUncertaintyAnalysisSettings(&f);
			std::getline(f, line);
			continue;
		}
		if (strcasecmp("COMMENT", tmpTag) == 0) {
			std::getline(f, line);
			continue;
		}
		if (strcasecmp("END_DATA", tmpTag) == 0) {
			break;
		}
		std::getline(f, line);
	}
	f.close();
	if (forwardCalc == false) {
		ReadImpedanceObsData(invSettings->impedanceFile);
		ReadTipperObsData(invSettings->tipperFile);
	}
}
std::vector<std::string> ReadData::ReadData::readNext(std::ifstream* f) {
	while (true) {
		std::string tmpLine;
		std::getline(*f, tmpLine);
		std::vector<std::string> line = split(tmpLine);
		if (line.size() == 0) {
			return line;
		}
		auto itr = line.begin();
		std::string word = *itr;
		const char* iChar = word.c_str();

		std::string compare1{ word[0] };
		std::string compare2{ "#" };

		if (compare1==compare2) {
			continue;
		}
		else {
			return line;
		}
	}
}


void ReadData::ReadData::AnalysisElements(std::ifstream* f) {
	std::vector<std::string> line = readNext(f);
	bool isNotRectangular = false;
	if (strcasecmp(line[0].c_str(), "NotRectangular") == 0) {
		isNotRectangular = true;
		line = readNext(f);
	}
	if (isNotRectangular == false) {
		int nodeID = 0;
		while (true) {
			if (strcasecmp(line[0].c_str(), "END") == 0) {
				if (strcasecmp(line[1].c_str(), "ELEMENTS") == 0) {
					break;
				}
			}
			if (line.size() != 9) {
				std::cout << "Wrong Elements Data" << std::endl;
				exit(1);
			}
			std::string ID = line[0];
			double rootCoord1 = stod(line[1]);
			double rootCoord2 = stod(line[2]);
			double rootCoord3 = stod(line[3]);
			double dx = stod(line[4]);
			double dy = stod(line[5]);
			double dz = stod(line[6]);
			int propID = stoi(line[7]);
			bool isParent = false;
			if (!strcasecmp("true", line[8].c_str())) {
				isParent = true;
			}
			Element::Element* element = new Element::Element();
			element->ID = ID;
			element->rootCoord.coeffRef(0) = rootCoord1;
			element->rootCoord.coeffRef(1) = rootCoord2;
			element->rootCoord.coeffRef(2) = rootCoord3;
			element->dx = dx;
			element->dy = dy;
			element->dz = dz;
			element->dv = dx * dy * dz;
			element->nodes.resize(8);
			for (int i = 0; i < 8; i++) {
				Node::Node* node = new Node::Node();
				node->ID = nodeID;
				if (i == 0) {
					node->x = element->rootCoord;
				}
				else if (i == 1) {
					node->x.coeffRef(0) = element->rootCoord.coeffRef(0) + dx;
					node->x.coeffRef(1) = element->rootCoord.coeffRef(1);
					node->x.coeffRef(2) = element->rootCoord.coeffRef(2);
				}
				else if (i == 2) {
					node->x.coeffRef(0) = element->rootCoord.coeffRef(0) + dx;
					node->x.coeffRef(1) = element->rootCoord.coeffRef(1) + dy;
					node->x.coeffRef(2) = element->rootCoord.coeffRef(2);
				}
				else if (i == 3) {
					node->x.coeffRef(0) = element->rootCoord.coeffRef(0);
					node->x.coeffRef(1) = element->rootCoord.coeffRef(1) + dy;
					node->x.coeffRef(2) = element->rootCoord.coeffRef(2);
				}
				else if (i == 4) {
					node->x.coeffRef(0) = element->rootCoord.coeffRef(0);
					node->x.coeffRef(1) = element->rootCoord.coeffRef(1);
					node->x.coeffRef(2) = element->rootCoord.coeffRef(2) + dz;
				}
				else if (i == 5) {
					node->x.coeffRef(0) = element->rootCoord.coeffRef(0) + dx;
					node->x.coeffRef(1) = element->rootCoord.coeffRef(1);
					node->x.coeffRef(2) = element->rootCoord.coeffRef(2) + dz;
				}
				else if (i == 6) {
					node->x.coeffRef(0) = element->rootCoord.coeffRef(0) + dx;
					node->x.coeffRef(1) = element->rootCoord.coeffRef(1) + dy;
					node->x.coeffRef(2) = element->rootCoord.coeffRef(2) + dz;
				}
				else if (i == 7) {
					node->x.coeffRef(0) = element->rootCoord.coeffRef(0);
					node->x.coeffRef(1) = element->rootCoord.coeffRef(1) + dy;
					node->x.coeffRef(2) = element->rootCoord.coeffRef(2) + dz;
				}
				element->nodes[i] = node;
			}
			element->centerCoord.coeffRef(0) = rootCoord1 + dx / 2;
			element->centerCoord.coeffRef(1) = rootCoord2 + dy / 2;
			element->centerCoord.coeffRef(2) = rootCoord3 + dz / 2;
			element->isParent = isParent;
			element->propID = propID;
			element->CalcSurfaceAndVolume();
			elements[ID] = element;
			elementsVector.push_back(element);
			line = readNext(f);
			if (f->eof()) {
				std::cout << "No END ELEMENTS In Data" << std::endl;
				exit(1);
			}
			nodeID++;
		}
	}
	else {
		int nodeID = 0;
		unordered_map<double,unordered_map<double,unordered_map<double, Node::Node*>>> mapNodes;
		while (true) {
			if (strcasecmp(line[0].c_str(), "END") == 0) {
				if (strcasecmp(line[1].c_str(), "ELEMENTS") == 0) {
					break;
				}
			}
			if (line.size() != 3) {
				std::cout << "Wrong Elements Data" << std::endl;
				exit(1);
			}
			std::string ID = line[0];
			int propID = stoi(line[1]);
			bool isParent = false;
			if (!strcasecmp("true", line[2].c_str())) {
				isParent = true;
			}
			vector<Eigen::Vector3d> nodes;
			Eigen::Vector3d centerCoord;
			centerCoord.setZero();
			nodes.resize(8);
			line = readNext(f);
			for (int i = 0; i < 8; i++) {
				if (line.size() != 3) {
					std::cout << "Wrong Node Value in Elements Data" << std::endl;
					exit(1);
				}
				for (int j = 0; j < 3; j++) {
					nodes[i].coeffRef(j) = stod(line[j]);
				}
				centerCoord += nodes[i];
				line = readNext(f);
			}
			centerCoord = centerCoord / 8.0;
			double dx = (nodes[2].coeff(0) + nodes[6].coeff(0)) * 0.5 - nodes[0].coeff(0); //average
			double dy = (nodes[2].coeff(1) + nodes[6].coeff(1)) * 0.5 - nodes[0].coeff(1); //average
			double dz = (nodes[4].coeff(2) + nodes[5].coeff(2) + nodes[6].coeff(2) + nodes[7].coeff(2)) * 0.25 - 
				(nodes[0].coeff(2) + nodes[1].coeff(2) + nodes[2].coeff(2) + nodes[3].coeff(2)) * 0.25; //average
			UnstructuredElement::UnstructuredElement* element = new UnstructuredElement::UnstructuredElement();
			element->nodes.resize(8);
			for (int i = 0; i < 8; i++) {
				if (mapNodes.find(nodes[i].coeff(0))==mapNodes.end() || mapNodes[nodes[i].coeff(0)].find(nodes[i].coeff(1)) == mapNodes[nodes[i].coeff(0)].end() ||
					mapNodes[nodes[i].coeff(0)][nodes[i].coeff(1)].find(nodes[i].coeff(2)) == mapNodes[nodes[i].coeff(0)][nodes[i].coeff(1)].end()) {
					Node::Node* node = new Node::Node();
					node->ID = nodeID;
					node->x = nodes[i];
					element->nodes[i] = node;
					mapNodes[nodes[i].coeff(0)][nodes[i].coeff(1)][nodes[i].coeff(2)] = node;
					nodeID++;
				}
				else {		
					element->nodes[i] = mapNodes[nodes[i].coeff(0)][nodes[i].coeff(1)][nodes[i].coeff(2)];
				}
				
			}
			element->ID = ID;
			element->rootCoord.coeffRef(0) = nodes[0].coeff(0);
			element->rootCoord.coeffRef(1) = nodes[0].coeff(1);
			element->rootCoord.coeffRef(2) = nodes[0].coeff(2);
			element->dx = dx;
			element->dy = dy;
			element->dz = dz;
			element->centerCoord.coeffRef(0) = centerCoord.coeff(0);
			element->centerCoord.coeffRef(1) = centerCoord.coeff(1);
			element->centerCoord.coeffRef(2) = centerCoord.coeff(2);
			element->isParent = isParent;
			element->propID = propID;
			element->CalcSurfaceAndVolume();
			elements[ID] = element;
			elementsVector.push_back(element);
			
			if (f->eof()) {
				std::cout << "No END ELEMENTS In Data" << std::endl;
				exit(1);
			}
		}
	}
}
void ReadData::ReadData::AnalysisProperties(std::ifstream* f) {
	std::vector<std::string> line = readNext(f);
	while (true) {
		if (strcasecmp(line[0].c_str(), "END") == 0) {
			if (strcasecmp(line[1].c_str(), "PROPERTIES") == 0) {
				break;
			}
		}
		if (strcasecmp(line[0].c_str(), "PROPERTY") == 0) {
			Property::Property* property = new Property::Property();
			property->ID = stoi(line[1]);

			line = readNext(f);
			while (true) {
				if (strcasecmp(line[0].c_str(), "END") == 0) {
					if (strcasecmp(line[1].c_str(), "PROPERTY") == 0) {
						break;
					}
				}
				if (strcasecmp(line[0].c_str(), "Resistivity") == 0) {
						property->resistivity = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "Type") == 0) {
					if (stod(line[1]) == Property::Property::types::NORMAL) {
						property->type = Property::Property::types::NORMAL;
					}
					else if (stod(line[1]) == Property::Property::types::AIR) {
						property->type = Property::Property::types::AIR;
					}
					else if (stod(line[1]) == Property::Property::types::FIXED) {
						property->type = Property::Property::types::FIXED;
					}
					else if (stod(line[1]) == Property::Property::types::SEA) {
						property->type = Property::Property::types::SEA;
					}
					else {
						std::cout << "Wrong Data in Property Type" << std::endl;
						exit(1);
					}
					line = readNext(f);
				}
				else {
					std::cout << "Wrong Data in Property" << std::endl;
					exit(1);
				}

				if (f->eof()) {
					std::cout << "No END Property In Data" << std::endl;
					exit(1);
				}
			}
			properties[property->ID] = property;
			propertiesVector.push_back(property);
		}
		else {
			std::cout << "Wrong Data in Properties" << std::endl;
			exit(1);
		}
		line = readNext(f);
		if (f->eof()) {
			std::cout << "No END Properties In Data" << std::endl;
			exit(1);
		}
	}
}
void ReadData::ReadData::AnalysisBoundary(std::ifstream* f) {
	boundary = new Boundary::Boundary();
	std::vector<std::string> line = readNext(f);
	while (true) {
		if (strcasecmp(line[0].c_str(), "END") == 0) {
			if (strcasecmp(line[1].c_str(), "BOUNDARY") == 0) {
				break;
			}
		}
		if (strcasecmp(line[0].c_str(), "omega") == 0) {
			int i = 0;
			line = readNext(f);
			while (true) {
				if (strcasecmp(line[0].c_str(), "END") == 0) {
					if (strcasecmp(line[1].c_str(), "omega") == 0) {
						break;
					}
				}
				boundary->omega.push_back( stod(line[0]));
				i++;
				line = readNext(f);
				if (f->eof()) {
					std::cout << "No END omega In Data" << std::endl;
					exit(1);
				}
			}
			//CHECK Omega Data
			if (boundary->omega.size() == 0) {
				cout << "Error::No Omega is in Your Data!!" << endl;
				exit(1);
			}
			for (int j = 1; j < boundary->omega.size(); j++) {
				if (boundary->omega[j] == boundary->omega[j - 1]) {
					std::cout << "\nWARNING::SAME FREQENCIES APPERAR IN Omega!!!!!!!" << std::endl;
					std::cout << boundary->omega[j] << " " << boundary->omega[j - 1] << std::endl;
				}
			}
		}
		else {
			std::cout << "Wrong Data in BOUNDARY" << std::endl;
			exit(1);
		}


		line = readNext(f);
		if (f->eof()) {
			std::cout << "No END BOUNDARY In Data" << std::endl;
			exit(1);
		}
	}
}

void ReadData::ReadData::ReadImpedanceObsData(string obsFileName) {
	//Impedance
	struct stat st;
	const char* file = obsFileName.c_str();
	int ret = stat(file, &st);
	if (obsFileName == "") {
		return;
	}
	else if (0 != ret) {
		std::cout << "Impedance File does not exist." << std::endl;
		exit(1);
	}

	std::ifstream f(obsFileName);

	std::vector<std::string> line;
	line = readNext(&f);

	int iID = lastObsDataID;
	int count = 0;
	while (!f.eof()) {
		
		if (line.size()!=2 && line.size() != 3) {
			std::cout << "Zobs Coordinate Setting is wrong." << std::endl;
			exit(1);
		}
		Eigen::Vector2d coord;
		double x = stod(line[0]);
		double y = stod(line[1]);
		string name = "";
		if (line.size() == 3) {
			name = line[2];
		}

		coord.coeffRef(0) = x;
		coord.coeffRef(1) = y;
		ObsData::ObsData* tmpObsData = new ObsData::ObsData();
		tmpObsData->isImpedanceData = true;
		tmpObsData->coord = coord;
		tmpObsData->name = name;
		for (int i = 0; i < boundary->omega.size(); i++) {
			line = readNext(&f);
			if (line.size() != 8) {
				std::cout << "Number Of Zobs is wrong." << std::endl;
				exit(1);
			}
			Eigen::Matrix2cd tmpZobs;
			tmpZobs.coeffRef(0, 0).real(stod(line[0]));
			tmpZobs.coeffRef(0, 0).imag(stod(line[1]));
			tmpZobs.coeffRef(0, 1).real(stod(line[2]));
			tmpZobs.coeffRef(0, 1).imag(stod(line[3]));
			tmpZobs.coeffRef(1, 0).real(stod(line[4]));
			tmpZobs.coeffRef(1, 0).imag(stod(line[5]));
			tmpZobs.coeffRef(1, 1).real(stod(line[6]));
			tmpZobs.coeffRef(1, 1).imag(stod(line[7]));
			tmpObsData->ZobsVector.push_back(tmpZobs);
			line = readNext(&f);
			Eigen::Matrix2d tmpWeightReal;
			Eigen::Matrix2d tmpWeightImag;
			tmpWeightReal.coeffRef(0, 0)=stod(line[0]);
			tmpWeightImag.coeffRef(0, 0) = stod(line[1]);
			tmpWeightReal.coeffRef(0, 1) = stod(line[2]);
			tmpWeightImag.coeffRef(0, 1) = stod(line[3]);
			tmpWeightReal.coeffRef(1, 0) = stod(line[4]);
			tmpWeightImag.coeffRef(1, 0) = stod(line[5]);
			tmpWeightReal.coeffRef(1, 1) = stod(line[6]);
			tmpWeightImag.coeffRef(1, 1) = stod(line[7]);
			tmpObsData->varianceZobsVectorReal.push_back(tmpWeightReal);
			tmpObsData->varianceZobsVectorImag.push_back(tmpWeightImag);
		}
		tmpObsData->ID = iID;
		iID++;
		lastObsDataID++;
		tmpObsData->impedanceID = count;
		count++;
		obsData.push_back(tmpObsData);
		line = readNext(&f);
	}
}

void ReadData::ReadData::ReadTipperObsData(string obsFileName) {
	//Impedance
	std::ifstream f(obsFileName);
	struct stat st;
	const char* file = obsFileName.c_str();
	int ret = stat(file, &st);
	if (obsFileName == "") {
		return;
	}
	else if (0 != ret) {
		std::cout << "Tipper File does not exist." << std::endl;
		exit(1);
	}
	std::vector<std::string> line;
	line = readNext(&f);

	int iID = lastObsDataID;
	int count = 0;
	while (!f.eof()) {

		if (line.size() != 2 && line.size() != 3) {
			std::cout << "Tobs Coordinate Setting is wrong." << std::endl;
			exit(1);
		}
		Eigen::Vector2d coord;
		double x = stod(line[0]);
		double y = stod(line[1]);
		coord.coeffRef(0) = x;
		coord.coeffRef(1) = y;
		string name = "";
		if (line.size() == 3) {
			name = line[2];
		}
		ObsData::ObsData* tmpObsData = new ObsData::ObsData();
		tmpObsData->isTipperData = true;
		tmpObsData->coord = coord;
		tmpObsData->name = name;
		for (int i = 0; i < boundary->omega.size(); i++) {
			line = readNext(&f);
			if (line.size() != 4) {
				std::cout << "Number Of Tobs is wrong." << std::endl;
				exit(1);
			}
			Eigen::Vector2cd tmpTobs;
			tmpTobs.coeffRef(0).real(stod(line[0]));
			tmpTobs.coeffRef(0).imag(stod(line[1]));
			tmpTobs.coeffRef(1).real(stod(line[2]));
			tmpTobs.coeffRef(1).imag(stod(line[3]));
			tmpObsData->TobsVector.push_back(tmpTobs);
			line = readNext(&f);
			Eigen::Vector2d tmpWeightReal;
			Eigen::Vector2d tmpWeightImag;
			tmpWeightReal.coeffRef(0) = stod(line[0]);
			tmpWeightImag.coeffRef(0) = stod(line[1]);
			tmpWeightReal.coeffRef(1) = stod(line[2]);
			tmpWeightImag.coeffRef(1) = stod(line[3]);
			tmpObsData->varianceTobsVectorReal.push_back(tmpWeightReal);
			tmpObsData->varianceTobsVectorImag.push_back(tmpWeightImag);
		}
		tmpObsData->ID = iID;
		iID++;
		lastObsDataID++;
		tmpObsData->tipperID = count;
		count++;
		obsData.push_back(tmpObsData);
		line = readNext(&f);
	}


}

void ReadData::ReadData::AnalysisInvSettings(std::ifstream* f) {
	std::vector<std::string> line = readNext(f);

	while (true) {
		if (strcasecmp(line[0].c_str(), "END") == 0) {
			if (strcasecmp(line[1].c_str(), "InvSettings") == 0) {
				break;
			}
		}
	
		if (strcasecmp(line[0].c_str(), "Parameters") == 0) {
			line = readNext(f);
			while (true) {
				if (strcasecmp(line[0].c_str(), "END") == 0) {
					if (strcasecmp(line[1].c_str(), "Parameters") == 0) {
						break;
					}
				}
				if (strcasecmp(line[0].c_str(), "StepSize") == 0) {
					invSettings->par_step_size = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "ParamN") == 0) {
					invSettings->paramLogNormalization = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "LambdaMax") == 0) {
					invSettings->modelConstraintMax = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "LambdaMin") == 0) {
					invSettings->modelConstraintMin = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "NumOfLambda") == 0) {
					invSettings->numOfCalcModelConstraint = stoi(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "maxIteration") == 0) {
					invSettings->maxIterationPerModelConstraint = stoi(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "maxResis") == 0) {
					invSettings->maxResis = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "minResis") == 0) {
					invSettings->minResis = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "outputInterval") == 0) {
					invSettings->outputInterval = stoi(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "gradErrorTol") == 0) {
					invSettings->grad_err_tol = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "thresholdResistivityChange") == 0) {
					invSettings->thresholdResistivityChange = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "thresholdObjFunctionChange") == 0) {
					invSettings->objFuncChangeThresholdForNextmodelConstraint = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "ManualSettingFile") == 0) {
					invSettings->manualSettingFile = line[1];
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "thresholdRMS") == 0) {
					if (stod(line[1]) < 0.0) {
						std::cout << "Value thresholdRMS must be 0 or more than 0." << std::endl;
						exit(1);
					}
					invSettings->thresholdRMS = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "modifyGradient") == 0) {
					if (!strcasecmp("true", line[1].c_str())) {
						invSettings->modifyGradient = true;
					}
					else {
						invSettings->modifyGradient = false;
					}
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "axForModifyGradient") == 0) {
					if (stod(line[1]) < 0.0) {
						std::cout << "Value axForModifyGradient must be 0 or more than 0." << std::endl;
						exit(1);
					}
					invSettings->axForModifyGradient = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "ayForModifyGradient") == 0) {
					if (stod(line[1]) < 0.0) {
						std::cout << "Value ayForModifyGradient must be 0 or more than 0." << std::endl;
						exit(1);
					}
					invSettings->ayForModifyGradient = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "CoeffForSearchStepSize") == 0) {
					invSettings->coeffForSearchStepSize = stod(line[1]);
					std::cout << "Warning!!!!!!!! \"CoeffForSearchStepSize\" is not used!!\n Please Use \"decreaseFactor\"!!!!!" << std::endl;
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "UseGD") == 0) {
					if (!strcasecmp("true", line[1].c_str())) {
						invSettings->optMethod = "GD";
					}
					else {
						invSettings->optMethod = "LBFGS";
					}
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "UseDistanceInModelConstraint") == 0) {
					if (!strcasecmp("true", line[1].c_str())) {
						invSettings->isUseDistanceInModelConstraint = true;
					}
					else {
						invSettings->isUseDistanceInModelConstraint = false;
					}
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "UseIterativeSolver") == 0) {
					if (!strcasecmp("true", line[1].c_str())) {
						invSettings->isDirectSolver = false;
					}
					else {
						invSettings->isDirectSolver = true;
					}
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "InitialGuessFileForIterativesolver") == 0) {
					invSettings->initialGuessFile=line[1];
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "InitialGuessOutputFileForIterativesolver") == 0) {
					invSettings->InitialGuessOutputFile = line[1];
					line = readNext(f);
					}


				else if (strcasecmp(line[0].c_str(), "ToleranceIterativeSolver") == 0) {
					if (stod(line[1]) < 0.0) {
						std::cout << "Value ToleranceIterativeSolver must be 0 or more than 0." << std::endl;
						exit(1);
					}
					invSettings->toleranceIterativeSolver = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "ToleranceIterativeSolverForAdjointEquation") == 0) {
					if (stod(line[1]) < 0.0) {
						std::cout << "Value ToleranceIterativeSolver must be 0 or more than 0." << std::endl;
						exit(1);
					}
					invSettings->toleranceIterativeSolverAdjoint = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "loosenFactor") == 0) {
					if (stod(line[1]) < 1.0) {
						std::cout << "Value loosenFactor must be 1 or more than 1." << std::endl;
						exit(1);
					}
					invSettings->loosenFactor = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "decreaseFactor") == 0) {
					if (stod(line[1]) > 1.0) {
						std::cout << "Value decreaseFactor must be 1 or less than 1." << std::endl;
						exit(1);
					}
					invSettings->decreaseFactor = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "maxIterationLineSearch") == 0) {
					if (stoi(line[1]) < 1) {
						std::cout << "Value maxIterationLineSearch must be 1 or more than 1." << std::endl;
						exit(1);
					}
					invSettings->maxIterationLineSearch = stoi(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "minStep") == 0) {
					if (stod(line[1]) <= 0.0) {
						std::cout << "Value minStep must be more than zero." << std::endl;
						exit(1);
					}
					invSettings->minStep = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "InheritPreviousSettingAdam") == 0) {
					if (!strcasecmp("true", line[1].c_str())) {
						invSettings->inheritPreviousSettingAdam = true;
					}
					else {
						invSettings->inheritPreviousSettingAdam = false;
					}
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "SettingEachLambda") == 0) {
					line = readNext(f);
					invSettings->lambdaVector.resize(0);
					invSettings->grad_err_tolVector.resize(0);
					invSettings->par_step_sizeVector.resize(0);
					invSettings->maxIterationVector.resize(0);
					while (true) {
						if (strcasecmp(line[0].c_str(), "END") == 0) {
							if (strcasecmp(line[1].c_str(), "SettingEachLambda") == 0) {
								break;
							}
							else {
								std::cout << "No End SettingEachLambda In Data" << std::endl;
								exit(1);
							}
						}
						else if (strcasecmp(line[0].c_str(), "data") == 0) {
							if (line.size() != 7) {
								std::cout << "Data In SettingEachLambda Is Wrong." << std::endl;
								exit(1);
							}
							invSettings->lambdaVector.push_back(stod(line[1]));
							invSettings->par_step_sizeVector.push_back(stod(line[2]));
							invSettings->grad_err_tolVector.push_back(stod(line[3]));
							invSettings->maxIterationVector.push_back(stoi(line[4]));
							invSettings->objFuncChangeThresholdVector.push_back(stod(line[5]));
							invSettings->thresholdRelativeResistivityChangeVector.push_back(stod(line[6]));

							line = readNext(f);
						}
						else {
							std::cout << "Wrong Data in SettingEachLambda" << std::endl;
							exit(1);
						}
					}
					line = readNext(f);

				}
				else if (strcasecmp(line[0].c_str(), "maxIterationBiCGSTAB") == 0) {
					invSettings->maxIterationBiCGSafe = stoi(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "minibatches") == 0) {
					invSettings->minibatches = stoi(line[1]);
					line = readNext(f);
				}

				else if (strcasecmp(line[0].c_str(), "RMSSwitchingToGD") == 0) {
					invSettings->RMSSwitchingToGD = stod(line[1]);
					if (stod(line[1]) <= 0.0) {
						std::cout << "Value RMSSwitchingToGD must be more than zero." << std::endl;
						exit(1);
					}
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "RMSSwitchingToLBFGS") == 0) {
					invSettings->RMSSwitchingToLBFGS = stod(line[1]);
					if (stod(line[1]) <= 0.0) {
						std::cout << "Value RMSSwitchingToLBFGS must be more than zero." << std::endl;
						exit(1);
					}
					line = readNext(f);
					}
				else if (strcasecmp(line[0].c_str(), "DistortionInversion") == 0) {
					if (!strcasecmp("true", line[1].c_str())) {
						invSettings->isInvertedDistortion = true;
					}
					else {
						invSettings->isInvertedDistortion = false;
					}
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "UseL1Norm") == 0) {
					if (!strcasecmp("true", line[1].c_str())) {
						invSettings->useL1Norm = true;
						if (line.size() > 2 && stod(line[2]) <= 1.0 && stod(line[2]) >= 0.0) {
							invSettings->rateL1Norm = stod(line[2]);
						}
					}
					else {
						invSettings->useL1Norm = false;
					}
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "fillFactorForILU") == 0) {
					invSettings->fillFactorForILU = stoi(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "safetyFactor") == 0) {
					invSettings->safetyFactor = stod(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "useLogScaleInterpolation") == 0) {
					if (!strcasecmp("true", line[1].c_str())) {
						invSettings->useLogScaleInterpolation = true;
					}
					else {
						invSettings->useLogScaleInterpolation = false;
					}
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "CalcJustDataMisfit") == 0) {
					if (!strcasecmp("true", line[1].c_str())) {
						calcJustDataMisfit = true;
					}
					else {
						calcJustDataMisfit = false;
					}
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "numOfWarmUp") == 0) {
					invSettings->numWarmUp = stoi(line[1]);
					if (stoi(line[1]) < 0.0) {
						std::cout << "Value numOfWarmUp must be more than or equal to zero." << std::endl;
						exit(1);
					}
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "minIteration") == 0) {
					invSettings->minIterations = stoi(line[1]);
					if (stoi(line[1]) < 0.0) {
						std::cout << "Value minIteration must be more than or equal to zero." << std::endl;
						exit(1);
					}
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "averageIteration") == 0) {
					invSettings->averageIterations = stoi(line[1]);
					if (stoi(line[1]) < 0.0) {
						std::cout << "Value averageIteration must be more than or equal to zero." << std::endl;
						exit(1);
					}
					line = readNext(f);
					}
				else if (strcasecmp(line[0].c_str(), "numTrunc") == 0) {
					invSettings->numTrunc = stoi(line[1]);
					if (stoi(line[1]) < 0.0) {
						std::cout << "Value numTrunc must be more than or equal to zero." << std::endl;
						exit(1);
					}
					line = readNext(f);
					}
				else {
					std::cout << "Wrong Data in Parameters:" << line[0].c_str()<< std::endl;
					exit(1);
				}

				if (f->eof()) {
					std::cout << "No END Parameters In Data" << std::endl;
					exit(1);
				}
			}
		}
		else if (strcasecmp(line[0].c_str(), "InitialSettings") == 0) {
			line = readNext(f);
			while (true) {
				if (strcasecmp(line[0].c_str(), "END") == 0) {
					if (strcasecmp(line[1].c_str(), "InitialSettings") == 0) {
						break;
					}
				}
				if (strcasecmp(line[0].c_str(), "InitialResistivityFile") == 0) {
					ReadInitialResistivityData(line[1]);
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "InitialDistortionFile") == 0) {
					ReadInitialDistortionData(line[1]);
					line = readNext(f);
				}
				else {
					std::cout << "Wrong Data in InitialSettings" << std::endl;
					exit(1);
				}

				if (f->eof()) {
					std::cout << "No END InitialSettings In Data" << std::endl;
					exit(1);
				}
			}
		}
		else if (strcasecmp(line[0].c_str(), "ObsFiles") == 0) {
			line = readNext(f);
			while (true) {
				if (strcasecmp(line[0].c_str(), "END") == 0) {
					if (strcasecmp(line[1].c_str(), "ObsFiles") == 0) {
						break;
					}
				}
				if (strcasecmp(line[0].c_str(), "ImpedanceFile") == 0) {
					invSettings->impedanceFile = line[1];
					line = readNext(f);
				}
				else if (strcasecmp(line[0].c_str(), "TipperFile") == 0) {
					invSettings->tipperFile = line[1];
					line = readNext(f);
				}
				else {
					std::cout << "Wrong Data in ObsFiles" << std::endl;
					exit(1);
				}

				if (f->eof()) {
					std::cout << "No END ObsFiles In Data" << std::endl;
					exit(1);
				}
			}
		}
		else if (strcasecmp(line[0].c_str(), "OutputSettings") == 0) {
			line = readNext(f);
			while (true) {
				if (strcasecmp(line[0].c_str(), "END") == 0) {
					if (strcasecmp(line[1].c_str(), "OutputSettings") == 0) {
						break;
					}
				}
				else if (strcasecmp(line[0].c_str(), "iteration") == 0) {
					output->outputIteration = stoi(line[1]);
					line = readNext(f);
				}
				else {
					std::cout << "Wrong Data in OutputSettings" << std::endl;
					exit(1);
				}

				if (f->eof()) {
					std::cout << "No END OutputSettings In Data" << std::endl;
					exit(1);
				}
			}
		}
		else {
			std::cout << "Wrong Data in InvSettings" << std::endl;
			exit(1);
		}
		line = readNext(f);
		if (f->eof()) {
			std::cout << "No END InvSettings In Data" << std::endl;
			exit(1);
		}


	}
}


void ReadData::ReadData::ReadInitialResistivityData (string resisFile) {
	//Impedance
	std::ifstream f(resisFile);
	struct stat st;
	const char* file = resisFile.c_str();
	int ret = stat(file, &st);
	if (resisFile == "") {
		return;
	}
	else if (0 != ret) {
		std::cout << "Initial Resistivity File does not exist." << std::endl;
		exit(1);
	}
	std::vector<std::string> line;
	line = readNext(&f);

	int iID = 0;
	while (!f.eof()) {

		if (line.size() != 4) {
			std::cout << "Coordinate Setting In Initial Resistivity File is wrong." << std::endl;
			exit(1);
		}
		Eigen::Vector3d coord;
		double x = stod(line[0]);
		double y = stod(line[1]);
		double z = stod(line[2]);
		double resis = stod(line[3]);
		coord.coeffRef(0) = x;
		coord.coeffRef(1) = y;
		coord.coeffRef(2) = z;

		InitialResistivityData::InitialResistivityData* tmpInitialData = new InitialResistivityData::InitialResistivityData();
		tmpInitialData->ID = iID;
		tmpInitialData->coord = coord;
		tmpInitialData->resistivity = resis;
		
		initialResistivityData.push_back(tmpInitialData);

		iID++;
		line = readNext(&f);

	}


}


void ReadData::ReadData::ReadInitialDistortionData(string distFile) {
	//Impedance
	std::ifstream f(distFile);
	struct stat st;
	const char* file = distFile.c_str();
	int ret = stat(file, &st);
	if (distFile == "") {
		return;
	}
	else if (0 != ret) {
		std::cout << "Initial Distortion File does not exist." << std::endl;
		exit(1);
	}
	std::vector<std::string> line;
	line = readNext(&f);

	int iID = 0;
	while (!f.eof()) {

		if (line.size() != 7) {
			std::cout << "Coordinate Setting In Initial Distortion File is wrong." << std::endl;
			exit(1);
		}
		Eigen::Vector3d coord;
		Eigen::Matrix2d mat;
		double x = stod(line[0]);
		double y = stod(line[1]);
		//double z = stod(line[2]);
		mat.coeffRef(0,0) = stod(line[3]);
		mat.coeffRef(0, 1) = stod(line[4]);
		mat.coeffRef(1, 0) = stod(line[5]);
		mat.coeffRef(1, 1) = stod(line[6]);
		coord.coeffRef(0) = x;
		coord.coeffRef(1) = y;
		//coord.coeffRef(2) = z;

		InitialDistortionData::InitialDistortionData* tmpInitialData = new InitialDistortionData::InitialDistortionData();
		tmpInitialData->ID = iID;
		tmpInitialData->coord = coord;
		tmpInitialData->distortionMatrix = mat;

		initialDistortionData.push_back(tmpInitialData);

		iID++;
		line = readNext(&f);

	}


}



void ReadData::ReadData::AnalysisLocationCalcSettings(std::ifstream* f) {
	
	std::vector<std::string> line = readNext(f);
	locationCalcSettings->isCalc = true;

	while (true) {
		if (strcasecmp(line[0].c_str(), "END") == 0) {
			
			if (strcasecmp(line[1].c_str(), "LocationCalcSettings") == 0) {
				break;
				
			}
		}
		else if (strcasecmp(line[0].c_str(), "ExecuteCalculation") == 0) {
			if (!strcasecmp("false", line[1].c_str())) {
				locationCalcSettings->isCalc = false;
			}
			else {
				locationCalcSettings->isCalc = true;
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "toleranceIterativeSolver") == 0) {
			invSettings->toleranceIterativeSolver = stod(line[1]);
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "locationFile") == 0) {
			ReadLocationDataFile(line[1]);
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "widthRatio") == 0) {
			locationCalcSettings->widthImpedance = stod(line[1]);
			if (locationCalcSettings->widthImpedance <= 1.0) {
				std::cout << "widthRatio In Location Calc Settings File must be equal to or more than 1." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "numOfSplit") == 0) {
			locationCalcSettings->numOfSplit = stoi(line[1]);
			if (locationCalcSettings->numOfSplit<=0) {
				std::cout << "numOfSplit In Location Calc Settings File must be equal to or more than 1." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "ResistivityFile") == 0) {
			ReadInitialResistivityData(line[1]);
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "ImpedanceFile") == 0) {
			invSettings->impedanceFile = line[1]; //流用
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "InitialDistortionFile") == 0) { //流用
			if (line[1] != "") {
				invSettings->isInvertedDistortion = true;
			}
			ReadInitialDistortionData(line[1]);
			line = readNext(f);
		}
		else {
			std::cout << "Wrong Data in LocationCalcSettings" << std::endl;
			exit(1);
		}
		if (f->eof()) {
			std::cout << "No END LocationCalcSettings In Data" << std::endl;
			exit(1);
		}
	}
}

void ReadData::ReadData::ReadLocationDataFile(string obsFileName) {
	struct stat st;
	const char* file = obsFileName.c_str();
	int ret = stat(file, &st);
	if (obsFileName == "") {
		return;
	}
	else if (0 != ret) {
		std::cout << "Location File does not exist." << std::endl;
		exit(1);
	}

	std::ifstream f(obsFileName);

	std::vector<std::string> line;
	line = readNext(&f);

	int iID = 0;
	while (!f.eof()) {

		if (line.size() != 2) {
			std::cout << "Location Data Coordinate Setting is wrong." << std::endl;
			exit(1);
		}
		Eigen::Vector2d coord;
		double x = stod(line[0]);
		double y = stod(line[1]);
		coord.coeffRef(0) = x;
		coord.coeffRef(1) = y;
		LocationData::LocationData* tmpData = new LocationData::LocationData();
		tmpData->coord = coord;
		tmpData->ID = iID;
		iID++;

		locationData.push_back(tmpData);
		line = readNext(&f);
	}
}



void ReadData::ReadData::AnalysisUncertaintyAnalysisSettings(std::ifstream* f) {

	std::vector<std::string> line = readNext(f);
	uncertaintyAnalysis->isCalc = true;

	while (true) {
		if (strcasecmp(line[0].c_str(), "END") == 0) {

			if (strcasecmp(line[1].c_str(), "UncertaintyAnalysisSettings") == 0) {
				break;
			}
		}
		else if (strcasecmp(line[0].c_str(), "ExecuteCalculation") == 0) {
			if (!strcasecmp("false", line[1].c_str())) {
				uncertaintyAnalysis->isCalc = false;
			}
			else {
				uncertaintyAnalysis->isCalc = true;
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "toleranceIterativeSolver") == 0) {
			invSettings->toleranceIterativeSolver = stod(line[1]);
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "thresHoldDeltaRMS") == 0) {
			uncertaintyAnalysis->thresHoldDeltaRMS = stod(line[1]);
			if (uncertaintyAnalysis->thresHoldDeltaRMS <= 0.0) {
				std::cout << "thresHoldDeltaRMS In UncertaintyAnalysisSettings must be between more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "priorModelStandardDeviation") == 0) {
			uncertaintyAnalysis->priorModelStandardDeviation = stod(line[1]);
			if (uncertaintyAnalysis->priorModelStandardDeviation < 0.0) {
				std::cout << "priorModelStandardDeviation In UncertaintyAnalysisSettings must be between more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "numOfValues") == 0) {
			uncertaintyAnalysis->numOfValues = stoi(line[1]);
			if (uncertaintyAnalysis->numOfValues <= 0) {
				std::cout << "numOfValues In UncertaintyAnalysisSettings must be between more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "numOfSamples") == 0) {
			uncertaintyAnalysis->numOfSamples = stoi(line[1]);
			if (uncertaintyAnalysis->numOfSamples <= 0) {
				std::cout << "numOfSamples In UncertaintyAnalysisSettings must be between more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "ResistivityFile") == 0 || strcasecmp(line[0].c_str(), "InitialResistivityFile") == 0) {
			ReadInitialResistivityData(line[1]);
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "ImpedanceFile") == 0) {
			invSettings->impedanceFile = line[1]; //流用
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "InitialDistortionFile") == 0) { //流用
			if (line[1] != "") {
				invSettings->isInvertedDistortion = true;
			}
			ReadInitialDistortionData(line[1]);
			line = readNext(f);
		}
		else {
			std::cout << "Wrong Data in UncertaintyAnalysisSettings" << std::endl;
			exit(1);
		}
		if (f->eof()) {
			std::cout << "No END UncertaintyAnalysisSettings In Data" << std::endl;
			exit(1);
		}
	}
}

size_t ReadData::HashFromCoordToSize_t::operator()(const Eigen::Vector3d& x) const {
	int digits = 12;
	string r = "";
	std::ostringstream ss;
	ss.precision(digits);

	for (int i = 0; i < 3; i++) {
		ss << x.coeff(i);

		r = r + ss.str();
	}
	std::stringstream sstream(r);
	size_t result;
	sstream >> result;
	return result;
}
void ReadData::ReadData::AnalysisFFTSensitivityAnalysis(std::ifstream* f) {

	std::vector<std::string> line = readNext(f);
	isFFTSensitivityMode = true;

	while (true) {
		if (strcasecmp(line[0].c_str(), "END") == 0) {

			if (strcasecmp(line[1].c_str(), "FFTSensitivityAnalysis") == 0) {
				break;

			}
		}
		else if (strcasecmp(line[0].c_str(), "attenuation") == 0) {
			attenuation = stod(line[1]);
			if (attenuation <= 0.0) {
				std::cout << "attenuation must be equal to or more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "Nx") == 0) {
			Nx = stoi(line[1]);
			if (Nx <= 0.0) {
				std::cout << "Nx must be equal more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "Ny") == 0) {
			Ny = stoi(line[1]);
			if (Ny <= 0.0) {
				std::cout << "Ny must be equal more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "Nz") == 0) {
			Nz = stoi(line[1]);
			if (Nz <= 0.0) {
				std::cout << "Nz must be equal more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "K") == 0) {
			K = stoi(line[1]);
			if (K <= 0.0) {
				std::cout << "K must be equal more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "cellsWindow") == 0) {
			cells_window = stoi(line[1]);
			if (cells_window <= 0.0) {
				std::cout << "cellsWindow must be equal more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "numEnsemble") == 0) {
			numEnsemble = stoi(line[1]);
			if (numEnsemble <= 0.0) {
				std::cout << "numEnsemble must be equal more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "rangeX") == 0) {
			minX = stod(line[1]);
			maxX = stod(line[2]);
			if (minX >= maxX) {
				std::cout << "minX must be less than maxX." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "rangeY") == 0) {
			minY = stod(line[1]);
			maxY = stod(line[2]);
			if (minY >= maxY) {
				std::cout << "minY must be less than maxY." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "rangeZ") == 0) {
			minZ = stod(line[1]);
			maxZ = stod(line[2]);
			if (minZ >= maxZ) {
				std::cout << "minZ must be less than maxZ." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "epsR") == 0) {
			epsR = stod(line[1]);
			if (epsR <= 0.0) {
				std::cout << "epsR must be equal more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "epsT") == 0) {
			epsT = stod(line[1]);
			if (epsT <= 0.0) {
				std::cout << "epsT must be equal more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "usePreviousResult") == 0) {
			if (!strcasecmp("false", line[1].c_str())) {
				usePreviousResult = false;
			}
			else {
				usePreviousResult = true;
			}
			line = readNext(f);
			}
		else if (strcasecmp(line[0].c_str(), "epsWindow") == 0) {
			eps_window = stod(line[1]);
			if (eps_window <= 0.0) {
				std::cout << "epsWindow must be equal more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "lambda") == 0) {
			lambda = stod(line[1]);
			if (lambda <= 0.0) {
				std::cout << "lambda must be equal more than 0." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "confidenceLevels") == 0) {
			confidenceLevel1 = stod(line[1]);
			confidenceLevel2 = stod(line[2]);
			if (confidenceLevel1 < 0.0 || confidenceLevel1>1) {
				std::cout << "confidenceLevel1 must be between 0 and 1." << std::endl;
				exit(1);
			}
			if (confidenceLevel2 < 0.0 || confidenceLevel2>1) {
				std::cout << "confidenceLevel2 must be between 0 and 1." << std::endl;
				exit(1);
			}
			if (confidenceLevel1 < confidenceLevel2) {
				std::cout << "confidenceLevel1 must be more than confidenceLevel2." << std::endl;
				exit(1);
			}
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "orthogonalizationMethod") == 0) {
			if (strcasecmp(line[0].c_str(), "gd") == 0) {
				orthogonalize = "gd";
			}
			else if (strcasecmp(line[0].c_str(), "both") == 0) {
				orthogonalize = "both";
			}
			else {
				orthogonalize = "objectiveFunction";
			}
			line = readNext(f);
			}
		else if (strcasecmp(line[0].c_str(), "InitialResistivityFile") == 0) {
			ReadInitialResistivityData(line[1]);
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "InitialDistortionFile") == 0) {
			ReadInitialDistortionData(line[1]);
			line = readNext(f);
		}

		else if (strcasecmp(line[0].c_str(), "ImpedanceFile") == 0) {
			invSettings->impedanceFile = line[1];
			line = readNext(f);
		}
		else if (strcasecmp(line[0].c_str(), "TipperFile") == 0) {
			invSettings->tipperFile = line[1];
			line = readNext(f);
		}

		else {
			std::cout << "Wrong Data in FFTSensitivityAnalysis" << std::endl;
			exit(1);
		}
		if (f->eof()) {
			std::cout << "No END FFTSensitivityAnalysis In Data" << std::endl;
			exit(1);
		}
	}
}
