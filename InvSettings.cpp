/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#pragma once 
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#pragma warning(disable : 4996)
#include "optim.hpp"
#include <iostream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
#include "InvSettings.h"
#include <fstream>
#include "optim.hpp"
#include <ostream>
#define strcasecmp _stricmp
InvSettings::InvSettings::InvSettings(int numOfLambda) {
	objFuncChangeThresholdVector.resize(numOfLambda);
	thresholdRelativeResistivityChangeVector.resize(numOfLambda);
}
std::vector<std::string> InvSettings::InvSettings::split(std::string str) {
	std::vector<std::string> result;
	std::string subStr;
	std::vector<char> del;
	del.push_back(' ');
	del.push_back('\t');
	del.push_back('\n');
	del.push_back('\r');
	for (const char c : str) {
		bool delFlag = false;
		for (auto itr = del.begin(); itr != del.end(); ++itr) {
			if (c == *itr) {
				delFlag = true;
			}
		}
		//if (c == del) {
		if (delFlag) {
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

void InvSettings::InvSettings::ReadManualSettingData(optim::algo_settings_t *settings) {
	if (manualSettingFile == "None") {
		return;
	}
	double tmpStepSize = settings->gd_settings.par_step_size;
	bool tmpIsFinishOptimize = settings->isFinishOptimize;
	
	try {
		FILE* fp = fopen(manualSettingFile.c_str(), "r");
		if (fp == NULL) {
			std::cout << "Warning::Manual Inversion Setting File Does Not Exist!" << std::endl;
			return;
		}
		fclose(fp);
		std::ifstream f(manualSettingFile);
		while (!f.eof()) {
			std::vector<std::string> line = readNext(&f);
			if (line.size() < 2) {
				continue;
			}
			if (strcasecmp(line[0].c_str(), "stepSize") == 0) {
				if (std::stod(line[1]) > 0) {
					settings->gd_settings.par_step_size = std::stod(line[1]);
				}
				else {
					cout << "Warning::Step Size In Manual Inversion Setting File is Wrong. " << endl;
				}
			}
			else if (strcasecmp(line[0].c_str(), "GoingNext") == 0) {
				if (strcasecmp(line[1].c_str(), "True") == 0) {
					settings->isFinishOptimize = true;
					cout << "Going Next Lambda." << endl;
				}
				else {
					cout << "Continue This Lambda." << endl;
				}
			}
		}
		f.close();
		std::ofstream wf;
		wf.open(manualSettingFile, std::ios_base::trunc); //Delete the contents.
		wf.close();
	}
	catch (...) {
		settings->gd_settings.par_step_size = tmpStepSize;
		settings->isFinishOptimize = tmpIsFinishOptimize;
		std::cout << "Warning:: Could not Read Manual Inversion Setting File. No Change has occurred." << std::endl;
		std::cout << "Manual Inversion Setting File Is:" << manualSettingFile << std::endl;
		return;
	}

}

std::vector<std::string> InvSettings::InvSettings::readNext(std::ifstream* f) {
	while (true) {
		if (f->eof()) {
			std::vector<std::string> line;
			return line;
		}
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

		if (compare1 == compare2) {
			continue;
		}
		else {
			return line;
		}
	}
}
