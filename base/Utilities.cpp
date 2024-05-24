#include "Utilities.h"


void loadVolumeData(const std::string& filename, Eigen::Tensor<double, 3>& data) {
    H5File file(filename.c_str(), H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet("density");
    DataSpace dataspace = dataset.getSpace();

    hsize_t dims[3];
    dataspace.getSimpleExtentDims(dims, NULL);
    int Nx = static_cast<int>(dims[2]);
    int Ny = static_cast<int>(dims[1]);
    int Nz = static_cast<int>(dims[0]);

    data.resize(Nx, Ny, Nz);

    std::vector<double> buffer(Nx * Ny * Nz);
    dataset.read(buffer.data(), PredType::NATIVE_DOUBLE);

    // Adjust the indexing to reflect Python's row-major order
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                data(i, j, k) = buffer[k * Ny * Nx + j * Nx + i];  // Adjust the index calculation
            }
        }
    }
}

Eigen::MatrixXd createCustomGridMatrix(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& z) {
    int Nx = x.size();
    int Ny = y.size();
    int Nz = z.size();
    Eigen::MatrixXd grid(Nx * Ny * Nz, 3);
    int idx = 0;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                grid(idx, 0) = x(i);
                grid(idx, 1) = y(j);
                grid(idx, 2) = z(k);
                ++idx;
            }
        }
    }
    return grid;
}

Eigen::VectorXd trilinearInterpolateMultiple(const Eigen::Tensor<double, 3>& values, const Eigen::MatrixXd& points, const Eigen::MatrixXd& qi) {
    int Nx = points.rows();
    int Ny = points.rows();
    int Nz = points.rows();

    Eigen::VectorXd results(qi.rows());
    for (int n = 0; n < qi.rows(); ++n) {
        Eigen::Vector3d point = qi.row(n);

        int ix = std::max(0, std::min(Nx - 2, static_cast<int>(std::floor((point(0) - points(0, 0)) / (points(1, 0) - points(0, 0))))));
        int iy = std::max(0, std::min(Ny - 2, static_cast<int>(std::floor((point(1) - points(0, 1)) / (points(1, 1) - points(0, 1))))));
        int iz = std::max(0, std::min(Nz - 2, static_cast<int>(std::floor((point(2) - points(0, 2)) / (points(1, 2) - points(0, 2))))));

        double x1 = points(ix, 0), x2 = points(ix + 1, 0);
        double y1 = points(iy, 1), y2 = points(iy + 1, 1);
        double z1 = points(iz, 2), z2 = points(iz + 1, 2);

        double xd = (point(0) - x1) / (x2 - x1);
        double yd = (point(1) - y1) / (y2 - y1);
        double zd = (point(2) - z1) / (z2 - z1);

        double c00 = values(ix, iy, iz) * (1 - xd) + values(ix + 1, iy, iz) * xd;
        double c01 = values(ix, iy, iz + 1) * (1 - xd) + values(ix + 1, iy, iz + 1) * xd;
        double c10 = values(ix, iy + 1, iz) * (1 - xd) + values(ix + 1, iy + 1, iz) * xd;
        double c11 = values(ix, iy + 1, iz + 1) * (1 - xd) + values(ix + 1, iy + 1, iz + 1) * xd;

        double c0 = c00 * (1 - yd) + c10 * yd;
        double c1 = c01 * (1 - yd) + c11 * yd;

        double interpolatedValue = c0 * (1 - zd) + c1 * zd;

        results(n) = interpolatedValue;
    }
    return results;
}

void linspace(double start, double end, int num, Eigen::VectorXd& result) {
    result.resize(num);

    if (num > 1) {
        double step = (end - start) / (num - 1);
        for (int i = 0; i < num; ++i) {
            result(i) = start + i * step;
        }
    } else if (num == 1) {
        result(0) = start;
    }
}

Eigen::MatrixXd transferFunction(const Eigen::VectorXd& x) {
    Eigen::MatrixXd output(4, x.size());

    for (int i = 0; i < x.size(); i++) {
        double r =  std::exp(-std::pow(x(i) - 9.0, 2) / 1.0) + 0.1 * std::exp(-std::pow(x(i) - 3.0, 2) / 0.1) + 0.1 * std::exp(-std::pow(x(i) + 3.0, 2) / 0.5);
        double g =  std::exp(-std::pow(x(i) - 9.0, 2) / 1.0) + std::exp(-std::pow(x(i) - 3.0, 2) / 0.1) + 0.1 * std::exp(-std::pow(x(i) + 3.0, 2) / 0.5);
        double b = 0.1 * std::exp(-std::pow(x(i) - 9.0, 2) / 1.0) + 0.1 * std::exp(-std::pow(x(i) - 3.0, 2) / 0.1) + std::exp(-std::pow(x(i) + 3.0, 2) / 0.5);
        double a = 0.6 * std::exp(-std::pow(x(i) - 9.0, 2) / 1.0) + 0.1 * std::exp(-std::pow(x(i) - 3.0, 2) / 0.1) + 0.01 * std::exp(-std::pow(x(i) + 3.0, 2) / 0.5);

        output(0, i) = r;
        output(1, i) = g;
        output(2, i) = b;
        output(3, i) = a;
    }

    return output;
}
