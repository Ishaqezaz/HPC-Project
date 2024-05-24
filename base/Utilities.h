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

void loadVolumeData(const std::string& filename, Eigen::Tensor<double, 3>& data);
Eigen::MatrixXd transferFunction(const Eigen::VectorXd& x);
void linspace(double start, double end, int num, Eigen::VectorXd& result);
Eigen::MatrixXd createPointsMatrix(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& z);

void findCellIndices(const Eigen::MatrixXd& grid, const Eigen::Vector3d& point, int& ix, int& iy, int& iz);
Eigen::MatrixXd createCustomGridMatrix(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& z);
Eigen::VectorXd trilinearInterpolateMultiple(const Eigen::Tensor<double, 3>& values, const Eigen::MatrixXd& grid, const Eigen::MatrixXd& qi);
Eigen::MatrixXd gridPoints(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& z);

                                        
#endif
