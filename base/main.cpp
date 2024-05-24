#include "Utilities.h"
#include <unsupported/Eigen/CXX11/Tensor>

using namespace cv;
using namespace Eigen;

int main() {
    Tensor<double, 3> data;
    std::string filename = "datacube.hdf5";
    loadVolumeData(filename, data);

    int Nx = data.dimension(0);
    int Ny = data.dimension(1);
    int Nz = data.dimension(2);
    std::cout << "Nx: " << Nx << " Ny: " << Ny << " Nz: " << Nz << std::endl;
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(Nx, -Nx / 2.0, Nx / 2.0);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(Ny, -Ny / 2.0, Ny / 2.0);
    Eigen::VectorXd z = Eigen::VectorXd::LinSpaced(Nz, -Nz / 2.0, Nz / 2.0);

    Eigen::MatrixXd points(Nx, 3);
    points.col(0) = x;
    points.col(1) = y;
    points.col(2) = z;

    int Nangles = 10;
    double angleOffset = 100.0 * M_PI / 180.0;

    for (int angleIndex = 0; angleIndex < Nangles; angleIndex++) {
        std::cout << "Rendering Scene " << angleIndex + 1 << " of " << Nangles << ".\n";

        double angle = angleOffset + M_PI / 2 * angleIndex / Nangles;
        int N = 180;

        Eigen::VectorXd c = Eigen::VectorXd::LinSpaced(N, -N / 2.0, N / 2.0);
        // Create meshgrid
        Eigen::Tensor<double, 3> qx(N, N, N);
        Eigen::Tensor<double, 3> qy(N, N, N);
        Eigen::Tensor<double, 3> qz(N, N, N);

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
        Eigen::Tensor<double, 3> qxR = qx;
        Eigen::Tensor<double, 3> qyR = qy * std::cos(angle) - qz * std::sin(angle);
        Eigen::Tensor<double, 3> qzR = qy * std::sin(angle) + qz * std::cos(angle);

        // Flattening
        Eigen::VectorXd qxR_flat(N * N * N);
        Eigen::VectorXd qyR_flat(N * N * N);
        Eigen::VectorXd qzR_flat(N * N * N);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) {
                    int flat_index = i * N * N + j * N + k;
                    qxR_flat(flat_index) = qxR(i, j, k);
                    qyR_flat(flat_index) = qyR(i, j, k);
                    qzR_flat(flat_index) = qzR(i, j, k);
                }
            }
        }

        Eigen::MatrixXd qi(N * N * N, 3);
        qi.col(0) = qxR_flat;
        qi.col(1) = qyR_flat;
        qi.col(2) = qzR_flat;

        // Interpolate
        Eigen::VectorXd cameraGrid = trilinearInterpolateMultiple(data, points, qi);

        int cameraGridDim1 = 180;
        int cameraGridDim2 = 180;

        // image intializing
        cv::Mat image(cameraGridDim1, cameraGridDim2, CV_64FC3, cv::Scalar(0, 0, 0)); // Using double precision

        // Iterate over slices
        for (int i = 0; i < cameraGridDim1; ++i) {
            Eigen::VectorXd dataslice = cameraGrid.segment(i * cameraGridDim2 * cameraGridDim2, cameraGridDim2 * cameraGridDim2);
            Eigen::VectorXd logDataslice = dataslice.array().log();

            Eigen::MatrixXd transferResults = transferFunction(logDataslice);

            for (int j = 0; j < cameraGridDim2; ++j) {
                for (int k = 0; k < cameraGridDim2; ++k) {
                    int idx = j * cameraGridDim2 + k;
                    double alpha = transferResults(3, idx);
                    double red = transferResults(2, idx); //openCV (bgr)
                    double green = transferResults(1, idx);
                    double blue = transferResults(0, idx);

                    cv::Vec3d& pixel = image.at<cv::Vec3d>(j, k);
                    pixel[0] = alpha * blue + (1 - alpha) * pixel[0]; // Blue channel
                    pixel[1] = alpha * green + (1 - alpha) * pixel[1]; // Green channel
                    pixel[2] = alpha * red + (1 - alpha) * pixel[2]; // Red channel
                }
            }
        }

        // 8 bit
        cv::Mat imageUC;
        cv::resize(image, image, cv::Size(960, 960), 0, 0, cv::INTER_LINEAR);

        image.convertTo(imageUC, CV_8UC3, 255.0);
        cv::Mat outputImage;
        cv::cvtColor(imageUC, outputImage, cv::COLOR_BGR2RGB);

        // Resizing and saving
        std::string imageName = "volumerender_" + std::to_string(angleIndex) + ".png";
        cv::imwrite(imageName, outputImage);
    }

    return 0;
}
