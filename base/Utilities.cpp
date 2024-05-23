#include "Utilities.h"

void loadVolumeData(const std::string& filename, Tensor<double, 3>& data) {
    H5File file(filename.c_str(), H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet("density");
    DataSpace dataspace = dataset.getSpace();

    hsize_t dims[3];
    dataspace.getSimpleExtentDims(dims, NULL);
    int Nx = static_cast<int>(dims[0]);
    int Ny = static_cast<int>(dims[1]);
    int Nz = static_cast<int>(dims[2]);

    data.resize(Nx, Ny, Nz);

    std::vector<double> buffer(Nx * Ny * Nz);
    dataset.read(buffer.data(), PredType::NATIVE_DOUBLE);

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                data(i, j, k) = buffer[i * Ny * Nz + j * Nz + k];
            }
        }
    }
}

Eigen::Tensor<double, 3> interpn(const Tensor<double, 3>& data, const MatrixXd& points, int N) {
    Eigen::Tensor<double, 3> cameraGrid(N, N, N);

    for (int index = 0; index < points.rows(); ++index) {
        int i = index / (N * N);
        int j = (index % (N * N)) / N;
        int k = index % N;

        double x = points(index, 0);
        double y = points(index, 1);
        double z = points(index, 2);

        int x0 = static_cast<int>(floor(x));
        int x1 = x0 + 1;
        int y0 = static_cast<int>(floor(y));
        int y1 = y0 + 1;
        int z0 = static_cast<int>(floor(z));
        int z1 = z0 + 1;

        // Clamping indices to prevent out-of-bound access
        x0 = std::max(0, std::min(x0, static_cast<int>(data.dimension(0) - 1)));
        x1 = std::max(0, std::min(x1, static_cast<int>(data.dimension(0) - 1)));
        y0 = std::max(0, std::min(y0, static_cast<int>(data.dimension(1) - 1)));
        y1 = std::max(0, std::min(y1, static_cast<int>(data.dimension(1) - 1)));
        z0 = std::max(0, std::min(z0, static_cast<int>(data.dimension(2) - 1)));
        z1 = std::max(0, std::min(z1, static_cast<int>(data.dimension(2) - 1)));

        // Ensure that you do not access the tensor with an invalid index
        if (x0 < 0 || y0 < 0 || z0 < 0 || x1 >= data.dimension(0) || y1 >= data.dimension(1) || z1 >= data.dimension(2)) {
            std::cerr << "Index out of bounds: (" << x0 << "," << y0 << "," << z0 << ") - (" << x1 << "," << y1 << "," << z1 << ")" << std::endl;
            continue;  // Skip this iteration safely or handle error appropriately
        }

        double xd = (x - x0);
        double yd = (y - y0);
        double zd = (z - z0);

        double c000 = data(x0, y0, z0);
        double c100 = data(x1, y0, z0);
        double c010 = data(x0, y1, z0);
        double c001 = data(x0, y0, z1);
        double c101 = data(x1, y0, z1);
        double c011 = data(x0, y1, z1);
        double c110 = data(x1, y1, z0);
        double c111 = data(x1, y1, z1);

        double c00 = c000 * (1 - xd) + c100 * xd;
        double c01 = c001 * (1 - xd) + c101 * xd;
        double c10 = c010 * (1 - xd) + c110 * xd;
        double c11 = c011 * (1 - xd) + c111 * xd;

        double c0 = c00 * (1 - yd) + c10 * yd;
        double c1 = c01 * (1 - yd) + c11 * yd;

        double interpnValue = c0 * (1 - zd) + c1 * zd;

        cameraGrid(i, j, k) = interpnValue;
    }

    return cameraGrid;
}



Eigen::MatrixXd transferFunction(const Eigen::VectorXd& x) {
    Eigen::MatrixXd output(4, x.size());

    for (int i = 0; i < x.size(); i++) {
        double r = std::exp(-std::pow(x(i) - 9.0, 2) / 1.0) + 0.1 * std::exp(-std::pow(x(i) - 3.0, 2) / 0.1) + 0.1 * std::exp(-std::pow(x(i) + 3.0, 2) / 0.5);
        double g = std::exp(-std::pow(x(i) - 9.0, 2) / 1.0) + std::exp(-std::pow(x(i) - 3.0, 2) / 0.1) + 0.1 * std::exp(-std::pow(x(i) + 3.0, 2) / 0.5);
        double b = 0.1 * std::exp(-std::pow(x(i) - 9.0, 2) / 1.0) + 0.1 * std::exp(-std::pow(x(i) - 3.0, 2) / 0.1) + std::exp(-std::pow(x(i) + 3.0, 2) / 0.5);
        double a = 0.6 * std::exp(-std::pow(x(i) - 9.0, 2) / 1.0) + 0.1 * std::exp(-std::pow(x(i) - 3.0, 2) / 0.1) + 0.01 * std::exp(-std::pow(x(i) + 3.0, 2) / 0.5);

        
        output(0, i) = r;
        output(1, i) = g;
        output(2, i) = b;
        output(3, i) = a;
    }

    return output;
}

void linspace(double start, double end, int num, Eigen::MatrixXd& result) {
    result.resize(num, 1);

    if (num > 1) {
        double step = (end - start) / (num - 1);
        for (int i = 0; i < num; ++i) {
            result(i, 0) = start + i * step;
        }
    } else if (num == 1) {
        result(0, 0) = start;
    }
}

Eigen::MatrixXd createPointsMatrix(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& z) {
    int Nx = x.size();
    int Ny = y.size();
    int Nz = z.size();
    Eigen::MatrixXd points(Nx * Ny * Nz, 3);
    int idx = 0;

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                points(idx, 0) = x(i);
                points(idx, 1) = y(j);
                points(idx, 2) = z(k);
                ++idx;
            }
        }
    }

    return points;
}
