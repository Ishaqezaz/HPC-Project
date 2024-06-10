#include "Utilities.h"

void loadVolumeData(const std::string& filename, Eigen::Tensor<double, 3>& data) {
    H5File file(filename.c_str(), H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet("density");
    DataSpace dataspace = dataset.getSpace();

    hsize_t dims[3];
    dataspace.getSimpleExtentDims(dims, NULL);
    int Nx = dims[0];
    int Ny = dims[1];
    int Nz = dims[2];

    data.resize(Nx, Ny, Nz);
    dataset.read(data.data(), PredType::NATIVE_DOUBLE);
}

Eigen::VectorXd interpolation(const Eigen::Tensor<double, 3>& values, const Eigen::MatrixXd& points, const Eigen::MatrixXd& qi) {
    int Nx = points.rows();
    int Ny = points.rows();
    int Nz = points.rows();

    Eigen::VectorXd results(qi.rows());

    // precomputing distances
    double dx = points(1, 0) - points(0, 0);
    double dy = points(1, 1) - points(0, 1);
    double dz = points(1, 2) - points(0, 2);

    for (int n = 0; n < qi.rows(); ++n) {
        Eigen::Vector3d point = qi.row(n);

        int ix = std::max(0, std::min(Nx - 2, static_cast<int>((point(0) - points(0, 0)) / dx)));
        int iy = std::max(0, std::min(Ny - 2, static_cast<int>((point(1) - points(0, 1)) / dy)));
        int iz = std::max(0, std::min(Nz - 2, static_cast<int>((point(2) - points(0, 2)) / dz)));

        double x1 = points(ix, 0), x2 = points(ix + 1, 0);
        double y1 = points(iy, 1), y2 = points(iy + 1, 1);
        double z1 = points(iz, 2), z2 = points(iz + 1, 2);

        double xd = (point(0) - x1) / dx;
        double yd = (point(1) - y1) / dy;
        double zd = (point(2) - z1) / dz;

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

Eigen::MatrixXd transferFunction(const Eigen::VectorXd& x) {
    int n = x.size();
    Eigen::MatrixXd output(4, n);

    // precompute frequently operations
    Eigen::ArrayXd xMinus9Squared = (x.array() - 9.0).square();
    Eigen::ArrayXd xMinus3Squared = (x.array() - 3.0).square();
    Eigen::ArrayXd xPlus3Squared = (x.array() + 3.0).square();

    Eigen::ArrayXd expMinus9 = (-xMinus9Squared / 1.0).exp();
    Eigen::ArrayXd expMinus3 = (-xMinus3Squared / 0.1).exp();
    Eigen::ArrayXd expPlus3 = (-xPlus3Squared / 0.5).exp();

    output.row(0) = (expMinus9 + 0.1 * expMinus3 + 0.1 * expPlus3).matrix();
    output.row(1) = (expMinus9 + expMinus3 + 0.1 * expPlus3).matrix();
    output.row(2) = (0.1 * expMinus9 + 0.1 * expMinus3 + expPlus3).matrix();
    output.row(3) = (0.6 * expMinus9 + 0.1 * expMinus3 + 0.01 * expPlus3).matrix();

    return output;
}

void createMeshgrid(Eigen::Tensor<double, 3>& qx, Eigen::Tensor<double, 3>& qy, Eigen::Tensor<double, 3>& qz, int N) {    
    Eigen::VectorXd c = Eigen::VectorXd::LinSpaced(N, -N / 2.0, N / 2.0);
    Eigen::Tensor<double, 1> c_tensor = Eigen::TensorMap<Eigen::Tensor<const double, 1>>(c.data(), N);

    Eigen::array<int, 3> bcast_x = {1, N, N};
    Eigen::array<int, 3> bcast_y = {N, 1, N};
    Eigen::array<int, 3> bcast_z = {N, N, 1};

    qx = c_tensor.reshape(Eigen::array<int, 3>{N, 1, 1}).broadcast(bcast_x);
    qy = c_tensor.reshape(Eigen::array<int, 3>{1, N, 1}).broadcast(bcast_y);
    qz = c_tensor.reshape(Eigen::array<int, 3>{1, 1, N}).broadcast(bcast_z);
}
