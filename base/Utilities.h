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


#include <unsupported/Eigen/CXX11/Tensor>
#include <H5Cpp.h>

using namespace std;
using namespace Eigen;
using namespace H5;

void loadVolumeData(const std::string& filename, Eigen::Tensor<double, 3>& data);
Eigen::MatrixXd transferFunction(const Eigen::VectorXd& x);
void createMeshgrid(Eigen::Tensor<double, 3>& qx, Eigen::Tensor<double, 3>& qy, Eigen::Tensor<double, 3>& qz, int N);
Eigen::VectorXd interpolation(const Eigen::Tensor<double, 3>& values, const Eigen::MatrixXd& points, const Eigen::MatrixXd& qi);
#endif
