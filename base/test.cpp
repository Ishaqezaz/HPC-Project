#include <gtest/gtest.h>
#include "Utilities.h"

TEST(VolumeRenderTest, TrilinearInterpolation) {
    Tensor<double, 3> data(3, 3, 3);
    data.setConstant(1.0);
    MatrixXd points(1, 3);
    points << 1.0, 1.0, 1.0;
    int N = 1;

    Eigen::Tensor<double, 3> result = interpn(data, points, N);
    EXPECT_EQ(result(0, 0, 0), 1.0);
}

TEST(VolumeRenderTest, KnownValueInterpolation) {
    Tensor<double, 3> data;
    std::string filename = "datacube.hdf5";
    loadVolumeData(filename, data);

    int N = 1;
    MatrixXd points(6, 3);
    points << data.dimension(0) / 2.0, data.dimension(1) / 2.0, data.dimension(2) / 2.0,
              data.dimension(0) / 4.0, data.dimension(1) / 4.0, data.dimension(2) / 3.0,
              3 * data.dimension(0) / 4.0, 3 * data.dimension(1) / 4.0, 2 * data.dimension(2) / 3.0,
              data.dimension(0) / 8.0, data.dimension(1) / 8.0, data.dimension(2) / 8.0,
              7 * data.dimension(0) / 8.0, 7 * data.dimension(1) / 8.0, 7 * data.dimension(2) / 8.0,
              data.dimension(0) / 2.0, 3 * data.dimension(1) / 4.0, data.dimension(2) / 4.0;

    Eigen::Tensor<double, 3> interpolatedValues = interpn(data, points, N);
    double expected[6] = {416.0301, 0.217467, 0.4252013, 0.07251948, 0.0707175, 0.4265221};

    for (int i = 0; i < 6; i++) {
        EXPECT_NEAR(interpolatedValues(0, 0, i), expected[i], 0.01);
    }
}

// Loading data
TEST(VolumeRenderTest, LoadVolumeData) {
    Tensor<double, 3> data;
    std::string filename = "datacube.hdf5";

    ASSERT_NO_THROW(loadVolumeData(filename, data));
    EXPECT_EQ(data.dimension(0), 256); 
    EXPECT_EQ(data.dimension(1), 256);
    EXPECT_EQ(data.dimension(2), 256);
}

// Transfer function
TEST(VolumeRenderTest, TransferFuncCorrectResults) {
    Eigen::VectorXd scalarValues(3);
    scalarValues << 10.0, 4.0, -2.0;

    Eigen::MatrixXd expected(4, 3);
    expected << 3.67879441e-01, 4.54000686e-06, 1.35335283e-02,
                3.67879441e-01, 4.53999437e-05, 1.35335283e-02,
                3.67879441e-02, 4.53999437e-06, 1.35335283e-01,
                2.20727665e-01, 4.54000131e-06, 1.35335283e-03;

    Eigen::MatrixXd rgbaValues = transferFunction(scalarValues);
    ASSERT_TRUE(rgbaValues.isApprox(expected, 1e-5));
}

// Transfer function
TEST(VolumeRenderTest, TransferFuncEdgeCases) {
    Eigen::VectorXd scalarValues(3);
    scalarValues << std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), 0.0;  // Scalar values for the function input

    Eigen::MatrixXd expected(4, 3);
    expected << 0.0, 0.0, 1.52299797e-09,
                0.0, 0.0, 1.52299797e-09,
                0.0, 0.0, 1.52299797e-08,
                0.0, 0.0, 1.52299797e-10;

    Eigen::MatrixXd rgbaValues = transferFunction(scalarValues);
    ASSERT_TRUE(rgbaValues.isApprox(expected, 1e-5));
}

// Test for the linspace function
TEST(VolumeRenderTest, Linspace) {
    int Nx = 5;
    int Ny = 5;
    int Nz = 5;

    Eigen::MatrixXd x, y, z;
    linspace(-Nx / 2.0, Nx / 2.0, Nx, x);
    linspace(-Ny / 2.0, Ny / 2.0, Ny, y);
    linspace(-Nz / 2.0, Nz / 2.0, Nz, z);

    Eigen::MatrixXd expectedx(5, 1);
    expectedx << -2.5, -1.25, 0, 1.25, 2.5;

    Eigen::MatrixXd expectedy(5, 1);
    expectedy << -2.5, -1.25, 0, 1.25, 2.5;

    Eigen::MatrixXd expectedz(5, 1);
    expectedz << -2.5, -1.25, 0, 1.25, 2.5;

    ASSERT_TRUE(x.isApprox(expectedx, 1e-5));
    ASSERT_TRUE(y.isApprox(expectedy, 1e-5));
    ASSERT_TRUE(z.isApprox(expectedz, 1e-5));
}

// Test for the linspace function
TEST(VolumeRenderTest, LinspaceEdge) {
    int Nx = 1;
    int Ny = 1;
    int Nz = 1;

    Eigen::MatrixXd x, y, z;
    linspace(-Nx / 3.0, Nx / 5.0, Nx, x);
    linspace(-Ny / 1.0, Ny / 4.0, Ny, y);
    linspace(-Nz / 2.0, Nz / 2.0, Nz, z);

    Eigen::MatrixXd expectedx(1, 1);
    expectedx << -Nx / 3.0;

    Eigen::MatrixXd expectedy(1, 1);
    expectedy << -Ny / 1.0;

    Eigen::MatrixXd expectedz(1, 1);
    expectedz << -Nz / 2.0;

    cout << "Real x: " << x << endl;
    cout << "Real y: " << y << endl;
    cout << "Real z: " << z << endl;

    cout << "Expected x: " << expectedx << endl;
    cout << "Expected y " << expectedy << endl;
    cout << "Expected z: " << expectedz << endl;

    ASSERT_TRUE(x.isApprox(expectedx, 1e-5));
    ASSERT_TRUE(y.isApprox(expectedy, 1e-5));
    ASSERT_TRUE(z.isApprox(expectedz, 1e-5));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
