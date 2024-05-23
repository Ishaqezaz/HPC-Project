#include "Utilities.h"
#include <unsupported/Eigen/CXX11/Tensor>

using namespace cv;
using namespace Eigen;

int main() {
    // Load data
    Tensor<double, 3> data;
    std::string filename = "datacube.hdf5";
    loadVolumeData(filename, data);

    int Nx = data.dimension(0);
    int Ny = data.dimension(1);
    int Nz = data.dimension(2);

    Eigen::MatrixXd x, y, z;
    linspace(-Nx / 2.0, Nx / 2.0, Nx, x);
    linspace(-Ny / 2.0, Ny / 2.0, Ny, y);
    linspace(-Nz / 2.0, Nz / 2.0, Nz, z);

    Eigen::MatrixXd points = createPointsMatrix(x, y, z);

    int Nangles = 10;

    for(int i = 0; i < Nangles; i++){
        std::cout << "Rendering Scene " << i + 1 << " of " << Nangles << ".\n";
        //Camera Grid / Query Points -- rotate camera view
        size_t angle = M_PI/2 * i / Nangles;
        int N = 180;

        //spacing
        Eigen::Tensor<float, 1> c(N);
        for(int i = 0; i < N; ++i) {
            c(i) = -N/2 + i; 
        }

        //meshgrid
        Eigen::Tensor<float, 3> qx(N, N, N), qy(N, N, N), qz(N, N, N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) {
                    qx(i, j, k) = c(i);
                    qy(i, j, k) = c(j);
                    qz(i, j, k) = c(k);
                }
            }
        }

        // rotation
        Eigen::Tensor<float, 3> qyR(N, N, N), qzR(N, N, N);
        Eigen::Tensor<float, 3> qxR = qx;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) {
                    qyR(i, j, k) = qy(i, j, k) * cos(angle) - qz(i, j, k) * sin(angle);
                    qzR(i, j, k) = qy(i, j, k) * sin(angle) + qz(i, j, k) * cos(angle);
                }
            }
        }

		 // flatten and combined into a single matrix
        Eigen::MatrixXd qi(N * N * N, 3);
        auto flat_qxR = Eigen::Map<Eigen::VectorXf>(qxR.data(), qxR.size());
        auto flat_qyR = Eigen::Map<Eigen::VectorXf>(qyR.data(), qyR.size());
        auto flat_qzR = Eigen::Map<Eigen::VectorXf>(qzR.data(), qzR.size());

        for (int idx = 0; idx < flat_qxR.size(); ++idx) {
            qi(idx, 0) = flat_qxR(idx);
            qi(idx, 1) = flat_qyR(idx);
            qi(idx, 2) = flat_qzR(idx);
        }   

        Eigen::Tensor<double, 3> cameraGrid = interpn(data, qi, N);
        

        //RENDERING
        /*cv::Mat image(N, N, CV_32FC3, cv::Scalar(0,0,0)); // OpenCV image matrix initialized to zero
        for (int k = 0; k < N; ++k) {
            Eigen::VectorXd vectorSlice(N * N);

            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    vectorSlice(i * N + j) = cameraGrid(i, j, k);
                }
            }

            Eigen::Tensor<double, 2> rgba = transferFunction(vectorSlice);
            std::cout << "Sample RGBA for slice " << k << ": " << rgba(0, 0) << ", " << rgba(1, 0) << ", " << rgba(2, 0) << ", " << rgba(3, 0) << std::endl;

            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    int idx = i * N + j;
                    float a = rgba(3, idx);
                    cv::Vec3f& pixel = image.at<cv::Vec3f>(i, j);

                    pixel[0] = a * rgba(0, idx) + (1.0f - a) * pixel[0];
                    pixel[1] = a * rgba(1, idx) + (1.0f - a) * pixel[1];
                    pixel[2] = a * rgba(2, idx) + (1.0f - a) * pixel[2];
                }
            }
        }

        cv::minMaxIdx(image, &min, &max);
        std::cout << "Final Image min: " << min << ", max: " << max << std::endl;

        image.convertTo(outputImage, CV_8UC3, 255.0);
        cv::imwrite(filename, outputImage);*/

    }




    return 0;
}
