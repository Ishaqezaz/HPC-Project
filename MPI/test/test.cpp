#include <gtest/gtest.h>
#include "Utilities.h"

TEST(InterpolationTest, BasicInterpolation) {
    int Nx = 3, Ny = 3, Nz = 3;

    Eigen::Tensor<double, 3> data(3, 3, 3);
    data.setValues({
        {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
        {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}},
        {{19, 20, 21}, {22, 23, 24}, {25, 26, 27}}
    });

    Eigen::MatrixXd points(3, 3);
    points << -1, -1, -1,
              0,  0,  0,
              1,  1,  1;

    Eigen::MatrixXd qi(3, 3);
    qi << -1, -1, -1,
           0,  0,  0,
           1,  1,  1;

    Eigen::VectorXd results = interpolation(data, points, qi);
    Eigen::VectorXd expected(3);
    expected << 1.0, 14.0, 27.0;

    for (int i = 0; i < expected.size(); ++i) {
        ASSERT_NEAR(results(i), expected(i), 0.01);
    }
}

TEST(InterpolationTest, interpolationEdge) {
    int Nx = 3, Ny = 3, Nz = 3;

    Eigen::Tensor<double, 3> data(Nx, Ny, Nz);
    data.setZero();

    Eigen::MatrixXd points(Nx, 3);
    points.col(0) = Eigen::VectorXd::LinSpaced(Nx, 0, 2);
    points.col(1) = Eigen::VectorXd::LinSpaced(Ny, 0, 2);
    points.col(2) = Eigen::VectorXd::LinSpaced(Nz, 0, 2);

    Eigen::MatrixXd qi(3, 3);
    qi << 0, 0, 0,
          1, 1, 1,
          2, 2, 2;

    Eigen::VectorXd results = interpolation(data, points, qi);

    Eigen::VectorXd expected(3);
    expected << 0.0, 0.0, 0.0;

    for (int i = 0; i < expected.size(); ++i) {
        ASSERT_NEAR(results(i), expected(i), 0.01);
    }
}


// loading data
TEST(VolumeRenderTest, LoadVolumeData) {
    Tensor<double, 3> data;
    std::string filename = "datacube.hdf5";

    ASSERT_NO_THROW(loadVolumeData(filename, data));
    EXPECT_EQ(data.dimension(0), 256); 
    EXPECT_EQ(data.dimension(1), 256);
    EXPECT_EQ(data.dimension(2), 256);
}

// transfer function
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

// transfer function
TEST(VolumeRenderTest, TransferFuncEdgeCases) {
    Eigen::VectorXd scalarValues(3);
    scalarValues << std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), 0.0;

    Eigen::MatrixXd expected(4, 3);
    expected << 0.0, 0.0, 1.52299797e-09,
                0.0, 0.0, 1.52299797e-09,
                0.0, 0.0, 1.52299797e-08,
                0.0, 0.0, 1.52299797e-10;

    Eigen::MatrixXd rgbaValues = transferFunction(scalarValues);
    ASSERT_TRUE(rgbaValues.isApprox(expected, 1e-5));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
