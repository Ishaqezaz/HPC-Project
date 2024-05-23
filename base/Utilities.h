#ifndef UTILITIES_H
#define UTILITIES_H

#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <fstream>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>
#include <H5Cpp.h>

using namespace std;
using namespace Eigen;
using namespace H5;

// Function declarations
void loadVolumeData(const std::string& filename, Tensor<double, 3>& data);
Eigen::Tensor<double, 3> interpn(const Tensor<double, 3>& data, const MatrixXd& points, int N);
Eigen::MatrixXd transferFunction(const Eigen::VectorXd& x);
void linspace(double start, double end, int num, Eigen::MatrixXd& result);
Eigen::MatrixXd createPointsMatrix(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& z);

#endif // UTILITIES_H
