#include "Utilities.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <chrono>

using namespace cv;
using namespace Eigen;
using namespace std::chrono;


int main(int argc, char** argv){
    
    int Nangles = 10; // default
    if (argc > 1) {
        Nangles = std::stoi(argv[1]);
    }


   auto start = high_resolution_clock::now();

    // loading data
    Tensor<double, 3> data;
    std::string filename = "datacube.hdf5";
    loadVolumeData(filename, data);
    int Nx = data.dimension(0);
    int Ny = data.dimension(1);
    int Nz = data.dimension(2);
    
    // VALID POINTS OF DATA 
    Eigen::MatrixXd points(Nx, 3);
    points.col(0) = Eigen::VectorXd::LinSpaced(Nx, -Nx / 2.0, Nx / 2.0);
    points.col(1) = Eigen::VectorXd::LinSpaced(Ny, -Ny / 2.0, Ny / 2.0);
    points.col(2) = Eigen::VectorXd::LinSpaced(Nz, -Nz / 2.0, Nz / 2.0);
    
    // meshgrid creation
    int N = 180;
    Eigen::Tensor<double, 3> qx(N, N, N);
    Eigen::Tensor<double, 3> qy(N, N, N);
    Eigen::Tensor<double, 3> qz(N, N, N);
    createMeshgrid(qx, qy, qz, N);

    for (int angleIndex = 0; angleIndex < Nangles; angleIndex++) {
        std::cout << "Rendering Scene " << angleIndex + 1 << " of " << Nangles << ".\n";

        double angle =  M_PI / 2 * angleIndex / Nangles;

        // rotation
        Eigen::Tensor<double, 3> qxR = qx;
        Eigen::Tensor<double, 3> qyR = qy * std::cos(angle) - qz * std::sin(angle);
        Eigen::Tensor<double, 3> qzR = qy * std::sin(angle) + qz * std::cos(angle);

        // Mapping, avoid copying when creating qi
        Eigen::Map<Eigen::VectorXd> qxR_flat(qxR.data(), qxR.size());
        Eigen::Map<Eigen::VectorXd> qyR_flat(qyR.data(), qyR.size());
        Eigen::Map<Eigen::VectorXd> qzR_flat(qzR.data(), qzR.size());
        Eigen::MatrixXd qi(N * N * N, 3);
        qi.col(0) = qxR_flat;
        qi.col(2) = qyR_flat;
        qi.col(1) = qzR_flat;
        
        // interpolation
        Eigen::VectorXd cameraGrid = interpolation(data, points, qi);

        int cameraGridDim1 = 180;
        int cameraGridDim2 = 180;

        // transfer and mapping
        cv::Mat image(cameraGridDim1, cameraGridDim2, CV_64FC3, cv::Scalar(0, 0, 0));
        for (int i = 0; i < cameraGridDim1; ++i) {
            Eigen::VectorXd dataslice = cameraGrid.segment(i * cameraGridDim2 * cameraGridDim2, cameraGridDim2 * cameraGridDim2);
            Eigen::VectorXd logDataslice = dataslice.array().log();
            Eigen::MatrixXd transferResults = transferFunction(logDataslice);

            for (int j = 0; j < cameraGridDim2; ++j) {
                cv::Vec3d* row_ptr = image.ptr<cv::Vec3d>(j);
                for (int k = 0; k < cameraGridDim2; ++k) {
                    int idx = j * cameraGridDim2 + k;
                    double alpha = transferResults(3, idx);
                    double red = transferResults(2, idx);   // OpenCV uses BGR format
                    double green = transferResults(1, idx);
                    double blue = transferResults(0, idx);

                    cv::Vec3d& pixel = row_ptr[k];
                    pixel[0] = alpha * blue + (1 - alpha) * pixel[0];
                    pixel[1] = alpha * green + (1 - alpha) * pixel[1];
                    pixel[2] = alpha * red + (1 - alpha) * pixel[2];
                }
            }
        }
      
        cv::Mat imageUC;
        cv::resize(image, imageUC, cv::Size(960, 960), 0, 0, cv::INTER_LINEAR);
        imageUC.convertTo(imageUC, CV_8UC3, 255.0);
        cv::Mat outputImage;
        cv::cvtColor(imageUC, outputImage, cv::COLOR_BGR2RGB);
        std::string imageName = "images/volumerender" + std::to_string(angleIndex) + ".png";
        cv::imwrite(imageName, outputImage);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long seconds = duration.count() / 1000;
    long long milliseconds = duration.count() % 1000;

    std::cout << "Execution time for " << Nangles << " images: " << seconds << "." << milliseconds << " seconds" << std::endl;

    return 0;
}
