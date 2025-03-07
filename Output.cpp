/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#pragma once
#include <string> 
#include <vector>
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unordered_map>
#include "Output.h"
#include "Element.h"
#include "ConstantValues.h"

void Output::Output::OutputObsCalcTipper(vector<double> omegas, std::vector< Element::Element*>* obsPointElements, std::string filename) {
	std::ofstream f;
	f.open(filename, std::ios::trunc);
	f << "Freq" << endl;
	for (int i = 0; i < omegas.size(); i++) {
		f << omegas[i] / 2.0 / ConstantValues::pi << endl;
	}

	for (int i = 0; i < obsPointElements->size(); i++) {
		if ((*obsPointElements)[i]->isInversionTipper == true) {
			f << i << " " << (*obsPointElements)[i]->centerCoord.coeff(0) << " " << (*obsPointElements)[i]->centerCoord.coeff(1) <<" "<<
				(*obsPointElements)[i]->impedanceObsData->name<<endl;
			for (int j = 0; j < omegas.size(); j++) {
				f << (*obsPointElements)[i]->T[j].coeff(0) << " " << (*obsPointElements)[i]->T[j].coeff(1) << endl;
				f << (*obsPointElements)[i]->tipperObsData->TobsVector[j].coeff(0) << " " << (*obsPointElements)[i]->tipperObsData->TobsVector[j].coeff(1) <<endl;
			}
		}
	}
	f.close();
}
void Output::Output::OutputObsCalcImpedance(vector<double> omegas, std::vector< Element::Element*>* obsPointElements,std::string filename) {
	std::ofstream f;
	f.open(filename, std::ios::trunc);
	f << "Freq" << endl;
	for (int i = 0; i < omegas.size(); i++) {
		f << omegas[i] / 2.0 / ConstantValues::pi << endl;
	}

	for (int i = 0; i < obsPointElements->size(); i++) {
		if ((*obsPointElements)[i]->isInversionImpedance == true) {
			f << i << " " << (*obsPointElements)[i]->centerCoord.coeff(0) << " " << (*obsPointElements)[i]->centerCoord.coeff(1) << " " <<
				(*obsPointElements)[i]->impedanceObsData->name << endl;
			for (int j = 0; j < omegas.size(); j++) {
				Eigen::Matrix2cd Z= (*obsPointElements)[i]->distortionMatrix * (*obsPointElements)[i]->Z[j];
				f << Z.coeff(0, 0) << " " << Z.coeff(0, 1) << " "
					<< Z.coeff(1, 0) << " " << Z.coeff(1, 1) << endl;
				f << (*obsPointElements)[i]->impedanceObsData->ZobsVector[j].coeff(0, 0) << " " << (*obsPointElements)[i]->impedanceObsData->ZobsVector[j].coeff(0, 1) << " "
					<< (*obsPointElements)[i]->impedanceObsData->ZobsVector[j].coeff(1, 0) << " " << (*obsPointElements)[i]->impedanceObsData->ZobsVector[j].coeff(1, 1) << endl;
			}
		}
	}
	f.close();
}
void Output::Output::VTKFileOputput( double omega,  std::unordered_map<string, Element::Element*>* elements,string type,string outputFile) {
	for (int loop = 0; loop < 2; loop++) {
		std::string filename;
		if (outputFile == "None") {
			if (type == "H") {
				filename = "Hx_" + std::to_string(loop) + "_" + to_string(omega) + ".vtk";
			}
			else if (type == "E") {
				filename = "Ex_" + std::to_string(loop) + "_" + to_string(omega) + ".vtk";
			}
			else if (type == "PHI") {
				filename = "Phi_" + std::to_string(loop) + "_" + to_string(omega) + ".vtk";
			}
			else if (type == "Z") {
				filename = "Z_" + std::to_string(loop) + "_" + to_string(omega) + ".vtk";
			}
			else if (type == "ObsPoints") {
				if (loop != 0) {
					break;
				}
				filename = "Obspoints.vtk";
			}
			else if (type == "Debug") {
				if (loop != 0) {
					break;
				}
				filename = "Debug.vtk";
			}
			else {
				filename = "LambdaX" + std::to_string(loop) + "_" + to_string(omega) + ".vtk";
			}
		}
		else {
			if (loop != 0) {
				break;
			}
			filename = outputFile;
		}
		const char* filename2 = filename.c_str();
		std::ofstream f;
		f.open(filename2, std::ios::trunc);

		int count = 0;
		for (auto itr = elements->begin(); itr != elements->end(); itr++) {
			Element::Element* element = itr->second;
			if (element->isParent == false) {
				count++;
			}
		}

		Eigen::VectorXd X{ count };
		Eigen::VectorXd Y{ count };
		Eigen::VectorXd Z{ count };
		Eigen::VectorXd rho{ count };
		Eigen::VectorXd phi{ count };
		Eigen::VectorXd HrX{ count };
		Eigen::VectorXd ErX{ count };
		Eigen::VectorXd HcX{ count };
		Eigen::VectorXd EcX{ count };
		Eigen::VectorXd HrY{ count };
		Eigen::VectorXd ErY{ count };
		Eigen::VectorXd HcY{ count };
		Eigen::VectorXd EcY{ count };
		Eigen::VectorXd LambdarX{ count };
		Eigen::VectorXd LambdarY{ count };
		Eigen::VectorXd LambdarZ{ count };
		Eigen::VectorXd LambdacX{ count };
		Eigen::VectorXd LambdacY{ count };
		Eigen::VectorXd LambdacZ{ count };
		Eigen::VectorXd obsPoints{ count };
		Eigen::VectorXd debug{ count };
		vector<string> ID;
		int i = 0;
		for (auto itr = elements->begin(); itr != elements->end(); itr++) {
			Element::Element* element = itr->second;
			if (element->isParent == false) {
				X[i] = element->rootCoord.coeff(0);
				Y[i] = element->rootCoord.coeff(1);
				Z[i] = element->rootCoord.coeff(2);
				ID.push_back(element->ID);
				if (loop == 0) {
					rho[i] = element->rhoXY;
					phi[i] = element->phiXY;
					HrX[i]=element->H[0][0].real();
					ErX[i] = element->E[0][0].real();
					HcX[i] = element->H[0][0].imag();
					EcX[i] = element->E[0][0].imag();
					HrY[i] = element->H[0][1].real();
					ErY[i] = element->E[0][1].real();
					HcY[i] = element->H[0][1].imag();
					EcY[i] = element->E[0][1].imag();
					LambdarX[i] = element->lambda1[0].real();
					LambdacX[i] = element->lambda1[0].imag();
					LambdarY[i] = element->lambda1[1].real();
					LambdacY[i] = element->lambda1[1].imag();
					LambdarZ[i] = element->lambda1[2].real();
					LambdacZ[i] = element->lambda1[2].imag();
					obsPoints[i] = element->isObservationElement;
					debug[i] = element->debug;
				}
				else {
					rho[i] = element->rhoYX;
					phi[i] = element->phiYX;
					HrX[i] = element->H[1][0].real();
					ErX[i] = element->E[1][0].real();
					HcX[i] = element->H[1][0].imag();
					EcX[i] = element->E[1][0].imag();
					HrY[i] = element->H[1][1].real();
					ErY[i] = element->E[1][1].real();
					HcY[i] = element->H[1][1].imag();
					EcY[i] = element->E[1][1].imag();
					LambdarX[i] = element->lambda2[0].real();
					LambdacX[i] = element->lambda2[0].imag();
					LambdarY[i] = element->lambda2[1].real();
					LambdacY[i] = element->lambda2[1].imag();
					LambdarZ[i] = element->lambda2[2].real();
					LambdacZ[i] = element->lambda2[2].imag();
				}
				i++;
			}
		}
		f << "# vtk DataFile Version 2.0" << endl;
		f << "Header" << endl;
		f << "ASCII" << endl;
		f << "DATASET UNSTRUCTURED_GRID" << endl;
		int vtkID = 0;
		unordered_map<int, int> nodesIDtoVTKID;
		vector<Eigen::Vector3d> nodes;
		for (int i = 0; i < count; i++) {
			Eigen::VectorXd x{ 8 };
			Eigen::VectorXd y{ 8 };
			Eigen::VectorXd z{ 8 };
			for (int j = 0; j < 8; j++) {
				if (nodesIDtoVTKID.find((*elements)[ID[i]]->nodes[j]->ID) == nodesIDtoVTKID.end()) {
					nodesIDtoVTKID[(*elements)[ID[i]]->nodes[j]->ID] = vtkID;
					nodes.push_back((*elements)[ID[i]]->nodes[j]->x);
					vtkID++;
				}
			}
		}
		string str = "POINTS " + to_string(nodes.size()) + " float";
		f << str << endl;
		for (int i = 0; i < nodes.size(); i++) {
			double x;
			double y;
			double z;
			x = nodes[i].coeff(0);
			y = nodes[i].coeff(1);
			z = nodes[i].coeff(2);
			str = to_string(x) + " " + to_string(y) + " " + to_string(z);
			f << str << endl;
			vtkID++;
		}
		str = "CELLS " + to_string(count) + " " + to_string(count * 9);
		f << str << endl;
		for (int i = 0; i < count; i++) {
			str = "8 ";
			for (int j = 0; j < 8; j++) {
				str += to_string(nodesIDtoVTKID[(*elements)[ID[i]]->nodes[j]->ID]) + " ";
			}
			f << str << endl;
		}
		f << "CELL_TYPES " + to_string(count) << endl;
		for (i = 0; i < count; i++) {
			str = "12";
			f << str << endl;
		}
		f << "CELL_DATA " + to_string(count) << endl;
		if (type != "PHI" && type != "Z" && type!="ObsPoints" && type!="Debug") {
			f << "SCALARS cell_scalars float 2" << endl;
		}
		else {
			f << "SCALARS cell_scalars float" << endl;
		}
		f << "LOOKUP_TABLE default" << endl;
		for (i = 0; i < count; i++) {
			if (type == "H") {
				f << to_string(HrX[i]) <<" "<< to_string(HcX[i]) << endl;
			}
			else if (type == "E") {
				f  << to_string(ErX[i])<<" " << to_string(EcX[i]) << endl;
			}
			else if (type == "PHI") {
				f << to_string(phi[i]) << endl;
			}
			else if(type == "Z") {
				f << to_string(rho[i]) << endl;
			}
			else if (type == "ObsPoints") {
				f << to_string(obsPoints[i]) << endl;
			}
			else if (type == "Debug") {
				f << to_string(debug[i]) << endl;
			}
			else {
				f << to_string(LambdarX[i]) << " " << to_string(LambdacX[i]) << endl;
			}
		}
		f.close();
	}

	for (int loop = 0; loop < 2; loop++) {
		std::string filename;
		if (type == "H") {
			filename = "Hy_" + std::to_string(loop) + "_" + to_string(omega) + ".vtk";
		}
		else if (type == "E") {
			filename = "Ey_" + std::to_string(loop) + "_" + to_string(omega) + ".vtk";
		}
		else if (type == "ObsPoints" || type=="Debug") {
			break;
		}
		else {
			filename = "LambdaY_" + std::to_string(loop) + "_" + to_string(omega) + ".vtk";;
		}
		const char* filename2 = filename.c_str();
		std::ofstream f;
		f.open(filename2, std::ios::trunc);

		int count = 0;
		for (auto itr = elements->begin(); itr != elements->end(); itr++) {
			Element::Element* element = itr->second;
			if (element->isParent == false) {
				count++;
			}
		}

		Eigen::VectorXd X{ count };
		Eigen::VectorXd Y{ count };
		Eigen::VectorXd Z{ count };
		Eigen::VectorXd rho{ count };
		
		Eigen::VectorXd HrX{ count };
		Eigen::VectorXd ErX{ count };
		Eigen::VectorXd HcX{ count };
		Eigen::VectorXd EcX{ count };
		Eigen::VectorXd HrY{ count };
		Eigen::VectorXd ErY{ count };
		Eigen::VectorXd HcY{ count };
		Eigen::VectorXd EcY{ count };
		Eigen::VectorXd LambdarX{ count };
		Eigen::VectorXd LambdarY{ count };
		Eigen::VectorXd LambdarZ{ count };
		Eigen::VectorXd LambdacX{ count };
		Eigen::VectorXd LambdacY{ count };
		Eigen::VectorXd LambdacZ{ count };
		vector<string> ID;
		int i = 0;
		for (auto itr = elements->begin(); itr != elements->end(); itr++) {
			Element::Element* element = itr->second;
			if (element->isParent == false) {
				X[i] = element->rootCoord.coeff(0);
				Y[i] = element->rootCoord.coeff(1);
				Z[i] = element->rootCoord.coeff(2);
				ID.push_back(element->ID);
				if (loop == 0) {
					rho[i] = element->rhoXY;
					
					HrX[i] = element->H[0][0].real();
					ErX[i] = element->E[0][0].real();
					HcX[i] = element->H[0][0].imag();
					EcX[i] = element->E[0][0].imag();
					HrY[i] = element->H[0][1].real();
					ErY[i] = element->E[0][1].real();
					HcY[i] = element->H[0][1].imag();
					EcY[i] = element->E[0][1].imag();
					LambdarX[i] = element->lambda1[0].real();
					LambdacX[i] = element->lambda1[0].imag();
					LambdarY[i] = element->lambda1[1].real();
					LambdacY[i] = element->lambda1[1].imag();
					LambdarZ[i] = element->lambda1[2].real();
					LambdacZ[i] = element->lambda1[2].imag();
				}
				else {
					rho[i] = element->rhoYX;
					
					HrX[i] = element->H[1][0].real();
					ErX[i] = element->E[1][0].real();
					HcX[i] = element->H[1][0].imag();
					EcX[i] = element->E[1][0].imag();
					HrY[i] = element->H[1][1].real();
					ErY[i] = element->E[1][1].real();
					HcY[i] = element->H[1][1].imag();
					EcY[i] = element->E[1][1].imag();
					LambdarX[i] = element->lambda2[0].real();
					LambdacX[i] = element->lambda2[0].imag();
					LambdarY[i] = element->lambda2[1].real();
					LambdacY[i] = element->lambda2[1].imag();
					LambdarZ[i] = element->lambda2[2].real();
					LambdacZ[i] = element->lambda2[2].imag();
				}
				i++;
			}
		}
		f << "# vtk DataFile Version 2.0" << endl;
		f << "Header" << endl;
		f << "ASCII" << endl;
		f << "DATASET UNSTRUCTURED_GRID" << endl;
		int vtkID = 0;
		unordered_map<int, int> nodesIDtoVTKID;
		vector<Eigen::Vector3d> nodes;
		for (int i = 0; i < count; i++) {
			Eigen::VectorXd x{ 8 };
			Eigen::VectorXd y{ 8 };
			Eigen::VectorXd z{ 8 };
			for (int j = 0; j < 8; j++) {
				if (nodesIDtoVTKID.find((*elements)[ID[i]]->nodes[j]->ID) == nodesIDtoVTKID.end()) {
					nodesIDtoVTKID[(*elements)[ID[i]]->nodes[j]->ID] = vtkID;
					nodes.push_back((*elements)[ID[i]]->nodes[j]->x);
					vtkID++;
				}
			}
		}
		string str = "POINTS " + to_string(nodes.size()) + " float";
		f << str << endl;
		for (int i = 0; i < nodes.size(); i++) {
			double x;
			double y;
			double z;
			x = nodes[i].coeff(0);
			y = nodes[i].coeff(1);
			z = nodes[i].coeff(2);
			str = to_string(x) + " " + to_string(y) + " " + to_string(z);
			f << str << endl;
			vtkID++;
		}
		str = "CELLS " + to_string(count) + " " + to_string(count * 9);
		f << str << endl;
		for (int i = 0; i < count; i++) {
			str = "8 ";
			for (int j = 0; j < 8; j++) {
				str += to_string(nodesIDtoVTKID[(*elements)[ID[i]]->nodes[j]->ID]) + " ";
			}
			f << str << endl;
		}
		f << "CELL_TYPES " + to_string(count) << endl;
		for (i = 0; i < count; i++) {
			str = "12";
			f << str << endl;
		}
		f << "CELL_DATA " + to_string(count) << endl;
		if (type != "PHI" && type != "Z") {
			f << "SCALARS cell_scalars float 2" << endl;
		}
		else {
			f << "SCALARS cell_scalars float" << endl;
		}
		f << "LOOKUP_TABLE default" << endl;
		for (i = 0; i < count; i++) {
			if (type == "H") {
				f << to_string(HrY[i]) << " " << to_string(HcY[i]) << endl;
			}
			else if (type == "E") {
				f << to_string(ErY[i]) << " " << to_string(EcY[i]) << endl;
			}

			else {
				f << to_string(LambdarY[i]) << " " << to_string(LambdacY[i]) << endl;
			}
		}
		f.close();
	}

	for (int loop = 0; loop < 2; loop++) {
		std::string filename;
		if (type == "H") {
			filename = "Hz_" + std::to_string(loop) + "_" + to_string(omega) + ".vtk";
		}
		else if (type == "ObsPoints" || type == "Debug") {
			break;
		}
		else {
			filename = "LambdaZ_" + std::to_string(loop) + "_" + to_string(omega) + ".vtk";;
		}
		const char* filename2 = filename.c_str();
		std::ofstream f;
		f.open(filename2, std::ios::trunc);

		int count = 0;
		for (auto itr = elements->begin(); itr != elements->end(); itr++) {
			Element::Element* element = itr->second;
			if (element->isParent == false) {
				count++;
			}
		}

		Eigen::VectorXd X{ count };
		Eigen::VectorXd Y{ count };
		Eigen::VectorXd Z{ count };
		Eigen::VectorXd HrZ{ count };
		Eigen::VectorXd HcZ{ count };
		Eigen::VectorXd LambdarZ{ count };
		Eigen::VectorXd LambdacZ{ count };
		vector<string> ID;
		int i = 0;
		for (auto itr = elements->begin(); itr != elements->end(); itr++) {
			Element::Element* element = itr->second;
			if (element->isParent == false) {
				X[i] = element->rootCoord.coeff(0);
				Y[i] = element->rootCoord.coeff(1);
				Z[i] = element->rootCoord.coeff(2);
				ID.push_back(element->ID);
				if (loop == 0) {
					HrZ[i] = element->H[0][1].real();
					HcZ[i] = element->H[0][1].imag();
					LambdarZ[i] = element->lambda1[2].real();
					LambdacZ[i] = element->lambda1[2].imag();
				}
				else {
					HrZ[i] = element->H[1][2].real();
					HcZ[i] = element->H[1][2].imag();
					LambdarZ[i] = element->lambda2[2].real();
					LambdacZ[i] = element->lambda2[2].imag();
				}
				i++;
			}
		}
		f << "# vtk DataFile Version 2.0" << endl;
		f << "Header" << endl;
		f << "ASCII" << endl;
		f << "DATASET UNSTRUCTURED_GRID" << endl;
		int vtkID = 0;
		unordered_map<int, int> nodesIDtoVTKID;
		vector<Eigen::Vector3d> nodes;
		for (int i = 0; i < count; i++) {
			Eigen::VectorXd x{ 8 };
			Eigen::VectorXd y{ 8 };
			Eigen::VectorXd z{ 8 };
			for (int j = 0; j < 8; j++) {
				if (nodesIDtoVTKID.find((*elements)[ID[i]]->nodes[j]->ID) == nodesIDtoVTKID.end()) {
					nodesIDtoVTKID[(*elements)[ID[i]]->nodes[j]->ID] = vtkID;
					nodes.push_back((*elements)[ID[i]]->nodes[j]->x);
					vtkID++;
				}
			}
		}
		string str = "POINTS " + to_string(nodes.size()) + " float";
		f << str << endl;
		for (int i = 0; i < nodes.size(); i++) {
			double x;
			double y;
			double z;
			x = nodes[i].coeff(0);
			y = nodes[i].coeff(1);
			z = nodes[i].coeff(2);
			str = to_string(x) + " " + to_string(y) + " " + to_string(z);
			f << str << endl;
			vtkID++;
		}
		str = "CELLS " + to_string(count) + " " + to_string(count * 9);
		f << str << endl;
		for (int i = 0; i < count; i++) {
			str = "8 ";
			for (int j = 0; j < 8; j++) {
				str += to_string(nodesIDtoVTKID[(*elements)[ID[i]]->nodes[j]->ID]) + " ";
			}
			f << str << endl;
		}
		f << "CELL_TYPES " + to_string(count) << endl;
		for (i = 0; i < count; i++) {
			str = "12";
			f << str << endl;
		}
		f << "CELL_DATA " + to_string(count) << endl;
		if (type != "PHI" && type != "Z") {
			f << "SCALARS cell_scalars float 2" << endl;
		}
		else {
			f << "SCALARS cell_scalars float" << endl;
		}
		f << "LOOKUP_TABLE default" << endl;
		for (i = 0; i < count; i++) {
			if (type == "H") {
				f << to_string(HrZ[i]) << " " << to_string(HcZ[i]) << endl;
			}
			else {
				f << to_string(LambdarZ[i]) << " " << to_string(LambdacZ[i]) << endl;
			}
		}
		f.close();
	}

}

void Output::Output::DebugOutput(int loop, double omega, std::unordered_map<string, Element::Element*>* elements) {
	std::string filename = "H" + std::to_string(omega) + "_" + std::to_string(loop) + ".txt";
	const char* filename2 = filename.c_str();

	std::ofstream f;
	f.open(filename2, std::ios::trunc);


	double freq = omega / 2 / ConstantValues::pi;
	for (auto itr = elements->begin(); itr != elements->end(); itr++) {
		bool isPlot = true;
		Element::Element* element = itr->second;
		//if (element->boundary=="NOT_BOUNDARY") {
			//double appRhoXY = pow(std::sqrt(std::pow(element->Z.back().coeff(0, 1).real(), 2.0) + std::pow(element->Z.back().coeff(0, 1).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double appRhoYX = pow(std::sqrt(std::pow(element->Z.back().coeff(1, 0).real(), 2.0) + std::pow(element->Z.back().coeff(1, 0).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double phaseXY = std::atan(element->Z.back().coeff(0, 1).imag() / element->Z.back().coeff(0, 1).real());
			//double phaseYX = std::atan(element->Z.back().coeff(1, 0).imag() / element->Z.back().coeff(1, 0).real());
			//f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
			//	appRhoXY << " " << appRhoYX << " " << phaseXY << " " << phaseYX << " " << element->resistivity
			//	<< std::endl;
			f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
				element->H[loop].coeff(loop) << std::endl;

		//}
	}
	f.close();

	filename = "E" + std::to_string(omega) + "_" + std::to_string(loop) + ".txt";
	const char* filename3 = filename.c_str();

	std::ofstream f2;
	f2.open(filename3, std::ios::trunc);



	for (auto itr = elements->begin(); itr != elements->end(); itr++) {
		bool isPlot = true;
		Element::Element* element = itr->second;
		if (element->boundary == "NOT_BOUNDARY") {
			//double appRhoXY = pow(std::sqrt(std::pow(element->Z.back().coeff(0, 1).real(), 2.0) + std::pow(element->Z.back().coeff(0, 1).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double appRhoYX = pow(std::sqrt(std::pow(element->Z.back().coeff(1, 0).real(), 2.0) + std::pow(element->Z.back().coeff(1, 0).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double phaseXY = std::atan(element->Z.back().coeff(0, 1).imag() / element->Z.back().coeff(0, 1).real());
			//double phaseYX = std::atan(element->Z.back().coeff(1, 0).imag() / element->Z.back().coeff(1, 0).real());
			//f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
			//	appRhoXY << " " << appRhoYX << " " << phaseXY << " " << phaseYX << " " << element->resistivity
			//	<< std::endl;
			int i=-loop + 1;
			f2 << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
				element->E[loop].coeff(i) << std::endl;

		}
	}



}
void Output::Output::TxtOutputAppRho(double omega, std::unordered_map<string, Element::Element*>* elements) {
	std::string filename = "AppXY_" + std::to_string(omega) + ".txt";
	const char* filename2 = filename.c_str();

	std::ofstream f;
	f.open(filename2, std::ios::trunc);


	double freq = omega / 2 / ConstantValues::pi;
	for (auto itr = elements->begin(); itr != elements->end(); itr++) {
		bool isPlot = true;
		Element::Element* element = itr->second;
		//if (element->boundary=="NOT_BOUNDARY") {
			//double appRhoXY = pow(std::sqrt(std::pow(element->Z.back().coeff(0, 1).real(), 2.0) + std::pow(element->Z.back().coeff(0, 1).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double appRhoYX = pow(std::sqrt(std::pow(element->Z.back().coeff(1, 0).real(), 2.0) + std::pow(element->Z.back().coeff(1, 0).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double phaseXY = std::atan(element->Z.back().coeff(0, 1).imag() / element->Z.back().coeff(0, 1).real());
			//double phaseYX = std::atan(element->Z.back().coeff(1, 0).imag() / element->Z.back().coeff(1, 0).real());
			//f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
			//	appRhoXY << " " << appRhoYX << " " << phaseXY << " " << phaseYX << " " << element->resistivity
			//	<< std::endl;
		f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
			element->rhoXY << std::endl;

		//}
	}
	f.close();

	filename = "AppYX_" + std::to_string(omega)+ ".txt";
	const char* filename3 = filename.c_str();

	std::ofstream f2;
	f2.open(filename3, std::ios::trunc);



	for (auto itr = elements->begin(); itr != elements->end(); itr++) {
		bool isPlot = true;
		Element::Element* element = itr->second;
		if (element->boundary == "NOT_BOUNDARY") {
			//double appRhoXY = pow(std::sqrt(std::pow(element->Z.back().coeff(0, 1).real(), 2.0) + std::pow(element->Z.back().coeff(0, 1).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double appRhoYX = pow(std::sqrt(std::pow(element->Z.back().coeff(1, 0).real(), 2.0) + std::pow(element->Z.back().coeff(1, 0).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double phaseXY = std::atan(element->Z.back().coeff(0, 1).imag() / element->Z.back().coeff(0, 1).real());
			//double phaseYX = std::atan(element->Z.back().coeff(1, 0).imag() / element->Z.back().coeff(1, 0).real());
			//f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
			//	appRhoXY << " " << appRhoYX << " " << phaseXY << " " << phaseYX << " " << element->resistivity
			//	<< std::endl;
			f2 << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
				element->rhoYX << std::endl;

		}
	}



}
void Output::Output::RhoOutput(std::unordered_map<string, Element::Element*>* elements, std::string filename,bool outputAll) {
	const char* filename2 = filename.c_str();
	std::ofstream f;
	f.open(filename2, std::ios::trunc);

	int count = 0;
	for (auto itr = elements->begin(); itr != elements->end(); itr++) {
		Element::Element* element = itr->second;
		if (element->isParent == false) {
			if (outputAll || element->property->type != Property::Property::types::AIR) {
				count++;
			}
		}
	}

	Eigen::VectorXd X{ count };
	Eigen::VectorXd Y{ count };
	Eigen::VectorXd Z{ count };
	Eigen::VectorXd val{ count };
	vector<string> ID;
	int i = 0;
	for (auto itr = elements->begin(); itr != elements->end(); itr++) {
		Element::Element* element = itr->second;
		if (element->isParent == false){
			if (outputAll || element->property->type != Property::Property::types::AIR) {
				X[i] = element->rootCoord.coeff(0);
				Y[i] = element->rootCoord.coeff(1);
				Z[i] = element->rootCoord.coeff(2);
				ID.push_back(element->ID);
				val[i] = element->resistivity;
				i++;
			}
		}
	}
	f << "# vtk DataFile Version 2.0" << endl;
	f << "Header" << endl;
	f << "ASCII" << endl;
	f << "DATASET UNSTRUCTURED_GRID" << endl;
	int vtkID = 0;
	unordered_map<int, int> nodesIDtoVTKID;
	vector<Eigen::Vector3d> nodes;
	for (int i = 0; i < count; i++) {
		Eigen::VectorXd x{ 8 };
		Eigen::VectorXd y{ 8 };
		Eigen::VectorXd z{ 8 };
		for (int j = 0; j < 8; j++) {
			if (nodesIDtoVTKID.find((*elements)[ID[i]]->nodes[j]->ID) == nodesIDtoVTKID.end()) {
				nodesIDtoVTKID[(*elements)[ID[i]]->nodes[j]->ID] = vtkID;
				nodes.push_back((*elements)[ID[i]]->nodes[j]->x);
				vtkID++;
			}
		}
	}
	string str = "POINTS " + to_string(nodes.size()) + " float";
	f << str << endl;
	for (int i = 0; i < nodes.size(); i++) {
		double x;
		double y;
		double z;
		x = nodes[i].coeff(0);
		y = nodes[i].coeff(1);
		z = nodes[i].coeff(2);
		str = to_string(x) + " " + to_string(y) + " " + to_string(z);
		f << str << endl;
		vtkID++;
	}
	str = "CELLS " + to_string(count) + " " + to_string(count * 9);
	f << str << endl;
	for (int i = 0; i < count; i++) {
		str = "8 ";
		for (int j = 0; j < 8; j++) {
			str += to_string(nodesIDtoVTKID[(*elements)[ID[i]]->nodes[j]->ID]) + " ";
		}
		f << str << endl;
	}
	f << "CELL_TYPES " + to_string(count) << endl;
	for (i = 0; i < count; i++) {
		str = "12";
		f << str << endl;
	}
	f << "CELL_DATA " + to_string(count) << endl;
	f << "SCALARS cell_scalars float" << endl;
	f << "LOOKUP_TABLE default" << endl;
	for (i = 0; i < count; i++) {
		f << to_string(val[i]) << endl;
	}
	f.close();
}


void Output::Output::AppRhoOutputSurface(double omega,std::unordered_map<string, Element::Element*>* elements) {
	std::string filename = "AppXYSurface_" + std::to_string(omega) + ".txt";
	const char* filename2 = filename.c_str();

	std::ofstream f;
	f.open(filename2, std::ios::trunc);


	double freq = omega / 2 / ConstantValues::pi;
	for (auto itr = elements->begin(); itr != elements->end(); itr++) {
		
		bool isPlot = false;
		Element::Element* element = itr->second;
		if (element->property->type == Property::Property::AIR) {
			continue;
		}


		Eigen::Vector3i pos;
		//for (int i = 0; i < 6; i++) {
		pos.setZero();
		//if (i == 0)pos[0] = -1;
		//if (i == 1)pos[0] =  1;
		//if (i == 2)pos[1] = -1;
		//if (i == 3)pos[1] =  1;
		//if (i == 4)pos[2] = -1;
		//if (i == 5)pos[2] =  1;
		pos.coeffRef(2) = -1;
		int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);

		if (element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false &&element->neighborElements[ipos]->isAirGroundBoundaryCell == true) {
			isPlot = true;

		}
		//}
		if (isPlot == false) {
			continue;
		}


		Element::Element* tmpElement = element;
		Element::Element* plotElement = element;
		//Comment out Below Because We set Air Cell Around Ground To AirGroundBoundaryCell 
		//while (true) {
		//	bool isFindPlotElement = true;
		//	for (int i = 0; i < 3; i++) {
		//		for (int j = 0; j < 3; j++) {
		//			for (int k = 0; k < 3; k++) {
		//				ipos = i + 3 * j + 9 * k;
		//				if (tmpElement->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && tmpElement->neighborElements[ipos]->property->type == Property::Property::AIR) {
		//					isFindPlotElement = false;
		//				}
		//			}
		//		}
		//	}
		//	if (isFindPlotElement) {
		//		plotElement = tmpElement;
		//		break;
		//	}
		//	else {
		//		tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1‚Â[‚¢ƒZƒ‹‚Ö
		//		//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1];
		//		if (tmpElement->isParent == true) {
		//			plotElement = tmpElement->neighborElements[1 + 3 + 9 * 0];
		//			break;
		//		}
		//	}
		//}


		//if (element->boundary=="NOT_BOUNDARY") {
			//double appRhoXY = pow(std::sqrt(std::pow(element->Z.back().coeff(0, 1).real(), 2.0) + std::pow(element->Z.back().coeff(0, 1).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double appRhoYX = pow(std::sqrt(std::pow(element->Z.back().coeff(1, 0).real(), 2.0) + std::pow(element->Z.back().coeff(1, 0).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double phaseXY = std::atan(element->Z.back().coeff(0, 1).imag() / element->Z.back().coeff(0, 1).real());
			//double phaseYX = std::atan(element->Z.back().coeff(1, 0).imag() / element->Z.back().coeff(1, 0).real());
			//f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
			//	appRhoXY << " " << appRhoYX << " " << phaseXY << " " << phaseYX << " " << element->resistivity
			//	<< std::endl;
		/*f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " <<
			element->rhoXY<< " " << element->layer << std::endl;*/
		//ipos = 1 + 3 * 1 + 9 * 1;
		f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " <<
			plotElement->rhoXY << " " << element->layer << std::endl;

		//}
	}
	f.close();

	filename = "AppYXSurface_" + std::to_string(omega) + ".txt";
	const char* filename3 = filename.c_str();

	std::ofstream f2;
	f2.open(filename3, std::ios::trunc);



	for (auto itr = elements->begin(); itr != elements->end(); itr++) {
		bool isPlot = false;
		Element::Element* element = itr->second;
		if (element->property->type == Property::Property::AIR) {
			continue;
		}
		Eigen::Vector3i pos;
		
		//for (int i = 0; i < 6; i++) {
			pos.setZero();
			//if (i == 0)pos[0] = -1;
			//if (i == 1)pos[0] = 1;
			//if (i == 2)pos[1] = -1;
			//if (i == 3)pos[1] = 1;
			//if (i == 4)pos[2] = -1;
			//if (i == 5)pos[2] = 1;
			pos.coeffRef(2) = -1;
			int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
			if (element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false && (*elements)[element->alreadyFoundNeighborID[ipos]]->isAirGroundBoundaryCell == true && element->property->type != Property::Property::AIR) {
				isPlot = true;

			}
		//}
		if (isPlot == false) {
			continue;
		}
		Element::Element* tmpElement = element;
		Element::Element* plotElement = element;
		//Comment out Below Because We set Air Cell Around Ground To AirGroundBoundaryCell 
		//while (true) {
		//	bool isFindPlotElement = true;
		//	for (int i = 0; i < 3; i++) {
		//		for (int j = 0; j < 3; j++) {
		//			for (int k = 0; k < 3; k++) {
		//				ipos = i + 3 * j + 9 * k;
		//				if (tmpElement->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && tmpElement->neighborElements[ipos]->property->type == Property::Property::AIR) {
		//					isFindPlotElement = false;
		//				}
		//			}
		//		}
		//	}
		//	if (isFindPlotElement) {
		//		plotElement = tmpElement;
		//		break;
		//	}
		//	else {
		//		tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1‚Â[‚¢ƒZƒ‹‚Ö
		//		//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1];
		//	}
		//}


		//if (element->boundary=="NOT_BOUNDARY") {
			//double appRhoXY = pow(std::sqrt(std::pow(element->Z.back().coeff(0, 1).real(), 2.0) + std::pow(element->Z.back().coeff(0, 1).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double appRhoYX = pow(std::sqrt(std::pow(element->Z.back().coeff(1, 0).real(), 2.0) + std::pow(element->Z.back().coeff(1, 0).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double phaseXY = std::atan(element->Z.back().coeff(0, 1).imag() / element->Z.back().coeff(0, 1).real());
			//double phaseYX = std::atan(element->Z.back().coeff(1, 0).imag() / element->Z.back().coeff(1, 0).real());
			//f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
			//	appRhoXY << " " << appRhoYX << " " << phaseXY << " " << phaseYX << " " << element->resistivity
			//	<< std::endl;
		//ipos = 1 + 3 * 1 + 9 * 1;
		f2 << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " <<
			plotElement->rhoYX <<" "<<element->layer<< std::endl;

		
	}
	f2.close();

}

void Output::Output::PhiOutputSurface(double omega, std::unordered_map<string, Element::Element*>* elements) {
	std::string filename = "PhiXYSurface_" + std::to_string(omega) + ".txt";
	const char* filename2 = filename.c_str();

	std::ofstream f;
	f.open(filename2, std::ios::trunc);


	double freq = omega / 2 / ConstantValues::pi;
	for (auto itr = elements->begin(); itr != elements->end(); itr++) {

		bool isPlot = false;
		Element::Element* element = itr->second;
		if (element->property->type == Property::Property::AIR) {
			continue;
		}
		Eigen::Vector3i pos;
		//for (int i = 0; i < 6; i++) {
			pos.setZero();
			//if (i == 0)pos[0] = -1;
			//if (i == 1)pos[0] = 1;
			//if (i == 2)pos[1] = -1;
			//if (i == 3)pos[1] = 1;
			//if (i == 4)pos[2] = -1;
			//if (i == 5)pos[2] = 1;
			pos.coeffRef(2) = -1;
			int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
			if (element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false && (*elements)[element->alreadyFoundNeighborID[ipos]]->isAirGroundBoundaryCell == true && element->property->type != Property::Property::AIR) {
				isPlot = true;

			}
		//}
		if (isPlot == false) {
			continue;
		}
		Element::Element* tmpElement = element;
		Element::Element* plotElement = element;
		//Comment out Below Because We set Air Cell Around Ground To AirGroundBoundaryCell 
		 
		//while (true) {
		//	bool isFindPlotElement = true;
		//	for (int i = 0; i < 3; i++) {
		//		for (int j = 0; j < 3; j++) {
		//			for (int k = 0; k < 3; k++) {
		//				ipos = i + 3 * j + 9 * k;
		//				if (tmpElement->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && tmpElement->neighborElements[ipos]->property->type == Property::Property::AIR) {
		//					isFindPlotElement = false;
		//				}
		//			}
		//		}
		//	}
		//	if (isFindPlotElement) {
		//		plotElement = tmpElement;
		//		break;
		//	}
		//	else {
		//		tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1‚Â[‚¢ƒZƒ‹‚Ö
		//		//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1];
		//	}
		//}


		//if (element->boundary=="NOT_BOUNDARY") {
			//double appRhoXY = pow(std::sqrt(std::pow(element->Z.back().coeff(0, 1).real(), 2.0) + std::pow(element->Z.back().coeff(0, 1).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double appRhoYX = pow(std::sqrt(std::pow(element->Z.back().coeff(1, 0).real(), 2.0) + std::pow(element->Z.back().coeff(1, 0).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double phaseXY = std::atan(element->Z.back().coeff(0, 1).imag() / element->Z.back().coeff(0, 1).real());
			//double phaseYX = std::atan(element->Z.back().coeff(1, 0).imag() / element->Z.back().coeff(1, 0).real());
			//f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
			//	appRhoXY << " " << appRhoYX << " " << phaseXY << " " << phaseYX << " " << element->resistivity
			//	<< std::endl;
		//ipos = 1 + 3 * 1 + 9 * 1;
		f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " <<
			plotElement->phiXY << " " << element->layer << std::endl;

		//}
	}
	f.close();

	filename = "PhiYXSurface_" + std::to_string(omega) + ".txt";
	const char* filename3 = filename.c_str();

	std::ofstream f2;
	f2.open(filename3, std::ios::trunc);



	for (auto itr = elements->begin(); itr != elements->end(); itr++) {
		bool isPlot = false;
		Element::Element* element = itr->second;
		if (element->property->type == Property::Property::AIR) {
			continue;
		}
		Eigen::Vector3i pos;

		//for (int i = 0; i < 6; i++) {
			pos.setZero();
			//if (i == 0)pos[0] = -1;
			//if (i == 1)pos[0] = 1;
			//if (i == 2)pos[1] = -1;
			//if (i == 3)pos[1] = 1;
			//if (i == 4)pos[2] = -1;
			//if (i == 5)pos[2] = 1;
			pos.coeffRef(2) = -1;
			int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
			if (element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false && (*elements)[element->alreadyFoundNeighborID[ipos]]->isAirGroundBoundaryCell == true && element->property->type != Property::Property::AIR) {
				isPlot = true;

			}
		//}
		if (isPlot == false) {
			continue;
		}
		Element::Element* tmpElement = element;
		Element::Element* plotElement = element;
		//Comment out Below Because We set Air Cell Around Ground To AirGroundBoundaryCell 

		//while (true) {
		//	bool isFindPlotElement = true;
		//	for (int i = 0; i < 3; i++) {
		//		for (int j = 0; j < 3; j++) {
		//			for (int k = 0; k < 3; k++) {
		//				ipos = i + 3 * j + 9 * k;
		//				if (tmpElement->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && tmpElement->neighborElements[ipos]->property->type == Property::Property::AIR) {
		//					isFindPlotElement = false;
		//				}
		//			}
		//		}
		//	}
		//	if (isFindPlotElement) {
		//		plotElement = tmpElement;
		//		break;
		//	}
		//	else {
		//		tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1‚Â[‚¢ƒZƒ‹‚Ö
		//		//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1];
		//	}
		//}


		//if (element->boundary=="NOT_BOUNDARY") {
			//double appRhoXY = pow(std::sqrt(std::pow(element->Z.back().coeff(0, 1).real(), 2.0) + std::pow(element->Z.back().coeff(0, 1).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double appRhoYX = pow(std::sqrt(std::pow(element->Z.back().coeff(1, 0).real(), 2.0) + std::pow(element->Z.back().coeff(1, 0).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double phaseXY = std::atan(element->Z.back().coeff(0, 1).imag() / element->Z.back().coeff(0, 1).real());
			//double phaseYX = std::atan(element->Z.back().coeff(1, 0).imag() / element->Z.back().coeff(1, 0).real());
			//f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
			//	appRhoXY << " " << appRhoYX << " " << phaseXY << " " << phaseYX << " " << element->resistivity
			//	<< std::endl;
		//ipos = 1 + 3 * 1 + 9 * 1;
		f2 << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " <<
			plotElement->phiYX << " " << element->layer << std::endl;


	}
	f2.close();

}

void Output::Output::ImpedanceOutputSurface(vector<double> omegas, std::unordered_map<string, Element::Element*>* elements) {
	std::string filename = "ImpedanceSurface.txt";
	const char* filename2 = filename.c_str();

	std::ofstream f;
	f.open(filename2, std::ios::trunc);

	for (auto itr = elements->begin(); itr != elements->end(); itr++) {

		bool isPlot = false;
		Element::Element* element = itr->second;
		if (element->property->type == Property::Property::AIR) {
			continue;
		}
		Eigen::Vector3i pos;
		//for (int i = 0; i < 6; i++) {
		pos.setZero();
		//if (i == 0)pos[0] = -1;
		//if (i == 1)pos[0] = 1;
		//if (i == 2)pos[1] = -1;
		//if (i == 3)pos[1] = 1;
		//if (i == 4)pos[2] = -1;
		//if (i == 5)pos[2] = 1;
		pos.coeffRef(2) = -1;
		int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
		if (element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false && (*elements)[element->alreadyFoundNeighborID[ipos]]->isAirGroundBoundaryCell == true && element->property->type != Property::Property::AIR) {
			isPlot = true;

		}
		//}
		if (isPlot == false) {
			continue;
		}
		Element::Element* tmpElement = element;
		Element::Element* plotElement = element;
		//Comment out Below Because We set Air Cell Around Ground To AirGroundBoundaryCell 

		//while (true) {
		//	bool isFindPlotElement = true;
		//	for (int i = 0; i < 3; i++) {
		//		for (int j = 0; j < 3; j++) {
		//			for (int k = 0; k < 3; k++) {
		//				ipos = i + 3 * j + 9 * k;
		//				if (tmpElement->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && tmpElement->neighborElements[ipos]->property->type == Property::Property::AIR) {
		//					isFindPlotElement = false;
		//				}
		//			}
		//		}
		//	}
		//	if (isFindPlotElement) {
		//		plotElement = tmpElement;
		//		break;
		//	}
		//	else {
		//		tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1‚Â[‚¢ƒZƒ‹‚Ö
		//		//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1];
		//	}
		//}


		//if (element->boundary=="NOT_BOUNDARY") {
			//double appRhoXY = pow(std::sqrt(std::pow(element->Z.back().coeff(0, 1).real(), 2.0) + std::pow(element->Z.back().coeff(0, 1).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double appRhoYX = pow(std::sqrt(std::pow(element->Z.back().coeff(1, 0).real(), 2.0) + std::pow(element->Z.back().coeff(1, 0).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
			//double phaseXY = std::atan(element->Z.back().coeff(0, 1).imag() / element->Z.back().coeff(0, 1).real());
			//double phaseYX = std::atan(element->Z.back().coeff(1, 0).imag() / element->Z.back().coeff(1, 0).real());
			//f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
			//	appRhoXY << " " << appRhoYX << " " << phaseXY << " " << phaseYX << " " << element->resistivity
			//	<< std::endl;
		//ipos = 1 + 3 * 1 + 9 * 1;
		f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << endl;
		for (int i = 0; i < omegas.size(); i++) {
			f << plotElement->Z[i].coeff(0, 0).real() << " " << plotElement->Z[i].coeff(0, 0).imag() << " "
				<< plotElement->Z[i].coeff(0, 1).real() << " " << plotElement->Z[i].coeff(0, 1).imag() << " "
				<< plotElement->Z[i].coeff(1, 0).real() << " " << plotElement->Z[i].coeff(1, 0).imag() << " "
				<< plotElement->Z[i].coeff(1, 1).real() << " " << plotElement->Z[i].coeff(1, 1).imag()  << endl;
			f << "1 1 1 1 1 1 1 1 " << endl; //data variance
		}

		//}
	}
	f.close();

}

void Output::Output::TipperOutputSurface(vector<double> omegas, std::unordered_map<string, Element::Element*>* elements) {
	std::string filename = "TipperSurface.txt";
	const char* filename2 = filename.c_str();

	std::ofstream f;
	f.open(filename2, std::ios::trunc);

	for (auto itr = elements->begin(); itr != elements->end(); itr++) {

		bool isPlot = false;
		Element::Element* element = itr->second;
		if (element->property->type == Property::Property::AIR) {
			continue;
		}
		Eigen::Vector3i pos;
		//for (int i = 0; i < 6; i++) {
		pos.setZero();
		//if (i == 0)pos[0] = -1;
		//if (i == 1)pos[0] = 1;
		//if (i == 2)pos[1] = -1;
		//if (i == 3)pos[1] = 1;
		//if (i == 4)pos[2] = -1;
		//if (i == 5)pos[2] = 1;
		pos.coeffRef(2) = -1;
		int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
		if (element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false && (*elements)[element->alreadyFoundNeighborID[ipos]]->isAirGroundBoundaryCell == true && element->property->type != Property::Property::AIR) {
			isPlot = true;

		}
		//}
		if (isPlot == false) {
			continue;
		}
		Element::Element* tmpElement = element;
		Element::Element* plotElement = element;
		//Comment out Below Because We set Air Cell Around Ground To AirGroundBoundaryCell 

		//while (true) {
		//	bool isFindPlotElement = true;
		//	for (int i = 0; i < 3; i++) {
		//		for (int j = 0; j < 3; j++) {
		//			for (int k = 0; k < 3; k++) {
		//				ipos = i + 3 * j + 9 * k;
		//				if (tmpElement->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && tmpElement->neighborElements[ipos]->property->type == Property::Property::AIR) {
		//					isFindPlotElement = false;
		//				}
		//			}
		//		}
		//	}
		//	if (isFindPlotElement) {
		//		plotElement = tmpElement;
		//		break;
		//	}
		//	else {
		//		tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1‚Â[‚¢ƒZƒ‹‚Ö
		//		//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1];
		//	}
		//}

		f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << endl;
		for (int i = 0; i < omegas.size(); i++) {
			f << plotElement->T[i].coeff(0).real() << " " << plotElement->T[i].coeff(0).imag() << " "
				<< plotElement->T[i].coeff(1).real() << " " << plotElement->T[i].coeff(1).imag() << endl;
			f << "1 1 1 1" << endl; //data variance
		}

		//}
	}
	f.close();

}


void Output::Output::TipperOutputSurface(int iOmega,double omega, std::unordered_map<string, Element::Element*>* elements) {
	{
		std::string filename = "TipperZXRealSurface_" + std::to_string(omega) + ".txt";
		const char* filename2 = filename.c_str();

		std::ofstream f;
		f.open(filename2, std::ios::trunc);


		double freq = omega / 2 / ConstantValues::pi;
		for (auto itr = elements->begin(); itr != elements->end(); itr++) {

			bool isPlot = false;
			Element::Element* element = itr->second;
			if (element->property->type == Property::Property::AIR) {
				continue;
			}


			Eigen::Vector3i pos;
			//for (int i = 0; i < 6; i++) {
			pos.setZero();
			//if (i == 0)pos[0] = -1;
			//if (i == 1)pos[0] =  1;
			//if (i == 2)pos[1] = -1;
			//if (i == 3)pos[1] =  1;
			//if (i == 4)pos[2] = -1;
			//if (i == 5)pos[2] =  1;
			pos.coeffRef(2) = -1;
			int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);

			if (element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false && element->neighborElements[ipos]->isAirGroundBoundaryCell == true) {
				isPlot = true;

			}
			//}
			if (isPlot == false) {
				continue;
			}

			Element::Element* tmpElement = element;
			Element::Element* plotElement = element;
			//Comment out Below Because We set Air Cell Around Ground To AirGroundBoundaryCell 

			//while (true) {
			//	bool isFindPlotElement = true;
			//	for (int i = 0; i < 3; i++) {
			//		for (int j = 0; j < 3; j++) {
			//			for (int k = 0; k < 3; k++) {
			//				ipos = i + 3 * j + 9 * k;
			//				if (tmpElement->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && tmpElement->neighborElements[ipos]->property->type == Property::Property::AIR) {
			//					isFindPlotElement = false;
			//				}
			//			}
			//		}
			//	}
			//	if (isFindPlotElement) {
			//		plotElement = tmpElement;
			//		break;
			//	}
			//	else {
			//		tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1‚Â[‚¢ƒZƒ‹‚Ö
			//		//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1];
			//	}
			//}


			f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " <<
				plotElement->T[iOmega].coeff(0).real() << " " << element->layer << std::endl;

			//}
		}
		f.close();
	}
	{
		std::string filename = "TipperZXImagSurface_" + std::to_string(omega) + ".txt";
		const char* filename2 = filename.c_str();

		std::ofstream f;
		f.open(filename2, std::ios::trunc);


		double freq = omega / 2 / ConstantValues::pi;
		for (auto itr = elements->begin(); itr != elements->end(); itr++) {

			bool isPlot = false;
			Element::Element* element = itr->second;
			if (element->property->type == Property::Property::AIR) {
				continue;
			}


			Eigen::Vector3i pos;
			//for (int i = 0; i < 6; i++) {
			pos.setZero();
			//if (i == 0)pos[0] = -1;
			//if (i == 1)pos[0] =  1;
			//if (i == 2)pos[1] = -1;
			//if (i == 3)pos[1] =  1;
			//if (i == 4)pos[2] = -1;
			//if (i == 5)pos[2] =  1;
			pos.coeffRef(2) = -1;
			int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);

			if (element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false && element->neighborElements[ipos]->isAirGroundBoundaryCell == true) {
				isPlot = true;

			}
			//}
			if (isPlot == false) {
				continue;
			}

			Element::Element* tmpElement = element;
			Element::Element* plotElement = element;
			//Comment out Below Because We set Air Cell Around Ground To AirGroundBoundaryCell 

			//while (true) {
			//	bool isFindPlotElement = true;
			//	for (int i = 0; i < 3; i++) {
			//		for (int j = 0; j < 3; j++) {
			//			for (int k = 0; k < 3; k++) {
			//				ipos = i + 3 * j + 9 * k;
			//				if (tmpElement->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && tmpElement->neighborElements[ipos]->property->type == Property::Property::AIR) {
			//					isFindPlotElement = false;
			//				}
			//			}
			//		}
			//	}
			//	if (isFindPlotElement) {
			//		plotElement = tmpElement;
			//		break;
			//	}
			//	else {
			//		tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1‚Â[‚¢ƒZƒ‹‚Ö
			//		//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1];
			//	}
			//}


			f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " <<
				plotElement->T[iOmega].coeff(0).imag() << " " << element->layer << std::endl;

			//}
		}
		f.close();
	}
	{
		std::string filename = "TipperZYRealSurface_" + std::to_string(omega) + ".txt";
		const char* filename2 = filename.c_str();

		std::ofstream f;
		f.open(filename2, std::ios::trunc);


		double freq = omega / 2 / ConstantValues::pi;
		for (auto itr = elements->begin(); itr != elements->end(); itr++) {

			bool isPlot = false;
			Element::Element* element = itr->second;
			if (element->property->type == Property::Property::AIR) {
				continue;
			}


			Eigen::Vector3i pos;
			//for (int i = 0; i < 6; i++) {
			pos.setZero();
			//if (i == 0)pos[0] = -1;
			//if (i == 1)pos[0] =  1;
			//if (i == 2)pos[1] = -1;
			//if (i == 3)pos[1] =  1;
			//if (i == 4)pos[2] = -1;
			//if (i == 5)pos[2] =  1;
			pos.coeffRef(2) = -1;
			int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);

			if (element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false && element->neighborElements[ipos]->isAirGroundBoundaryCell == true) {
				isPlot = true;

			}
			//}
			if (isPlot == false) {
				continue;
			}

			Element::Element* tmpElement = element;
			Element::Element* plotElement = element;
			//Comment out Below Because We set Air Cell Around Ground To AirGroundBoundaryCell 

			//while (true) {
			//	bool isFindPlotElement = true;
			//	for (int i = 0; i < 3; i++) {
			//		for (int j = 0; j < 3; j++) {
			//			for (int k = 0; k < 3; k++) {
			//				ipos = i + 3 * j + 9 * k;
			//				if (tmpElement->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && tmpElement->neighborElements[ipos]->property->type == Property::Property::AIR) {
			//					isFindPlotElement = false;
			//				}
			//			}
			//		}
			//	}
			//	if (isFindPlotElement) {
			//		plotElement = tmpElement;
			//		break;
			//	}
			//	else {
			//		tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1‚Â[‚¢ƒZƒ‹‚Ö
			//		//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1];
			//	}
			//}


			f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " <<
				plotElement->T[iOmega].coeff(1).real() << " " << element->layer << std::endl;

			//}
		}
		f.close();
	}
	{
		std::string filename = "TipperZYImagSurface_" + std::to_string(omega) + ".txt";
		const char* filename2 = filename.c_str();

		std::ofstream f;
		f.open(filename2, std::ios::trunc);


		double freq = omega / 2 / ConstantValues::pi;
		for (auto itr = elements->begin(); itr != elements->end(); itr++) {

			bool isPlot = false;
			Element::Element* element = itr->second;
			if (element->property->type == Property::Property::AIR) {
				continue;
			}


			Eigen::Vector3i pos;
			//for (int i = 0; i < 6; i++) {
			pos.setZero();
			//if (i == 0)pos[0] = -1;
			//if (i == 1)pos[0] =  1;
			//if (i == 2)pos[1] = -1;
			//if (i == 3)pos[1] =  1;
			//if (i == 4)pos[2] = -1;
			//if (i == 5)pos[2] =  1;
			pos.coeffRef(2) = -1;
			int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);

			if (element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false && element->neighborElements[ipos]->isAirGroundBoundaryCell == true) {
				isPlot = true;

			}
			//}
			if (isPlot == false) {
				continue;
			}

			Element::Element* tmpElement = element;
			Element::Element* plotElement = element;
			//Comment out Below Because We set Air Cell Around Ground To AirGroundBoundaryCell 

			//while (true) {
			//	bool isFindPlotElement = true;
			//	for (int i = 0; i < 3; i++) {
			//		for (int j = 0; j < 3; j++) {
			//			for (int k = 0; k < 3; k++) {
			//				ipos = i + 3 * j + 9 * k;
			//				if (tmpElement->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && tmpElement->neighborElements[ipos]->property->type == Property::Property::AIR) {
			//					isFindPlotElement = false;
			//				}
			//			}
			//		}
			//	}
			//	if (isFindPlotElement) {
			//		plotElement = tmpElement;
			//		break;
			//	}
			//	else {
			//		tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1‚Â[‚¢ƒZƒ‹‚Ö
			//		//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1];
			//	}
			//}


			f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " <<
				plotElement->T[iOmega].coeff(1).imag() << " " << element->layer << std::endl;

			//}
		}
		f.close();
	}
}
void Output::Output::TxtOutputResistivity(std::unordered_map<string, Element::Element*>* elements, std::string filename) {

	std::ofstream f;
	f.open(filename, std::ios::trunc);


	for (auto itr = elements->begin(); itr != elements->end(); itr++) {
		Element::Element* element = itr->second;
		if (element->isParent == false) {
			f << itr->second->centerCoord.coeff(0) << " " << itr->second->centerCoord.coeff(1) << " " << itr->second->centerCoord.coeff(2) << " " <<
				element->resistivity << std::endl;
		}

	}
	f.close();
}
void Output::Output::OutputDistortionMatrix(std::vector<Element::Element*>* obsImpedanceElements, double termVal, std::string filename) {
	std::ofstream f;
	f.open(filename, std::ios::trunc);

	f << "Constraint term for Distortion Matrix:" << termVal << endl;
	for (int i = 0; i < obsImpedanceElements->size(); i++) {
		Element::Element* element = (*obsImpedanceElements)[i];
		f << "Element ImpedanceID:" << element->impedanceObsID << endl;
		f << "Nx,Ny,Nz:" << element->IDX << " " << element->IDY << " " << element->IDZ << endl;
		f << "Coordinate(X,Y):" << element->impedanceObsData->coord.coeff(0) << " " << element->impedanceObsData->coord.coeff(1)<< std::endl;
		f << "Element ID" << element->ID << endl;
		f << "Distortion Matrix:" << endl;
		f << "coeff(0,0):" << element->distortionMatrix.coeff(0, 0) << endl;
		f << "coeff(0,1):" << element->distortionMatrix.coeff(0, 1) << endl;
		f << "coeff(1,0):" << element->distortionMatrix.coeff(1, 0) << endl;
		f << "coeff(1,1):" << element->distortionMatrix.coeff(1, 1) << endl;
		
	}
	f.close();
}
void Output::Output::OutputDistortionMatrixForRestart(std::vector<Element::Element*>* obsImpedanceElements, std::string filename) {
	std::ofstream f;
	f.open(filename, std::ios::trunc);
	for (int i = 0; i < obsImpedanceElements->size(); i++) {
		Element::Element* element = (*obsImpedanceElements)[i];
		f << element->impedanceObsData->coord.coeff(0) << " " << element->impedanceObsData->coord.coeff(1) << " " <<
			0.0 << " "
		  << element->distortionMatrix.coeff(0, 0) << " " << element->distortionMatrix.coeff(0, 1) << " " <<
			element->distortionMatrix.coeff(1, 0) << " " << element->distortionMatrix.coeff(1, 1) << endl;
	}
	f.close();
}



void Output::Output::VTKFileOputput(std::vector<Element::Element*>* calcElements, Eigen::VectorXd* values, string outputFile) {

	const char* filename2 = outputFile.c_str();
	std::ofstream f;
	f.open(filename2, std::ios::trunc);

	int count = calcElements->size();

	f << "# vtk DataFile Version 2.0" << endl;
	f << "Header" << endl;
	f << "ASCII" << endl;
	f << "DATASET UNSTRUCTURED_GRID" << endl;
	
	int vtkID = 0;
	unordered_map<int, int> nodesIDtoVTKID;
	vector<Eigen::Vector3d> nodes;
	for (int i = 0; i < count; i++) {
		Eigen::VectorXd x{ 8 };
		Eigen::VectorXd y{ 8 };
		Eigen::VectorXd z{ 8 };
		for (int j = 0; j < 8; j++) {
			if (nodesIDtoVTKID.find((*calcElements)[i]->nodes[j]->ID) == nodesIDtoVTKID.end()) {
				nodesIDtoVTKID[(*calcElements)[i]->nodes[j]->ID] = vtkID;
				nodes.push_back((*calcElements)[i]->nodes[j]->x);
				vtkID++;
			}
		}
	}
	string str = "POINTS " + to_string(nodes.size()) + " float";
	f << str << endl;
	for (int i = 0; i < nodes.size(); i++) {
		double x;
		double y;
		double z;
		x = nodes[i].coeff(0);
		y = nodes[i].coeff(1);
		z = nodes[i].coeff(2);
		str = to_string(x) + " " + to_string(y) + " " + to_string(z);
		f << str << endl;
		vtkID++;
	}
	str = "CELLS " + to_string(count) + " " + to_string(count * 9);
	f << str << endl;
	for (int i = 0; i < count; i++) {
		str = "8 ";
		for (int j = 0; j < 8; j++) {
			str += to_string(nodesIDtoVTKID[(*calcElements)[i]->nodes[j]->ID]) + " ";
		}
		f << str << endl;
	}
	f << "CELL_TYPES " + to_string(count) << endl;
	for (int i = 0; i < count; i++) {
		str = "12";
		f << str << endl;
	}
	f << "CELL_DATA " + to_string(count) << endl;
	f << "SCALARS cell_scalars float" << endl;
	f << "LOOKUP_TABLE default" << endl;
	for (int i = 0; i < count; i++) {

		f << to_string(values->coeff(i)) << endl;
	}
	f.close();
}
void Output::Output::VTKObsPointsFileOutput(std::vector<Element::Element*>* obsPointsElements, string outputFile) {

	const char* filename2 = outputFile.c_str();
	std::ofstream f;
	f.open(filename2, std::ios::trunc);

	int count = obsPointsElements->size();

	f << "# vtk DataFile Version 2.0" << endl;
	f << "Header" << endl;
	f << "ASCII" << endl;
	f << "DATASET UNSTRUCTURED_GRID" << endl;

	
	string str = "POINTS " + to_string(obsPointsElements->size()) + " float";
	f << str << endl;
	for (int i = 0; i < obsPointsElements->size(); i++) {
		double x;
		double y;
		double z;
		x = (* obsPointsElements)[i]->centerCoord.coeff(0);
		y = (*obsPointsElements)[i]->centerCoord.coeff(1);
		z = (*obsPointsElements)[i]->centerCoord.coeff(2);
		str = to_string(x) + " " + to_string(y) + " " + to_string(z);
		f << str << endl;
	}
	str = "CELLS " + to_string(obsPointsElements->size()) + " " + to_string(obsPointsElements->size() * 2);
	f << str << endl;
	for (int i = 0; i < count; i++) {
		str = "1 ";
		str += to_string(i) + " ";
		
		f << str << endl;
	}
	f << "CELL_TYPES " + to_string(obsPointsElements->size()) << endl;
	for (int i = 0; i < obsPointsElements->size(); i++) {
		str = "1";
		f << str << endl;
	}
	f.close();
}
