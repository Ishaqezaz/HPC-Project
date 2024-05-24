#include <gtest/gtest.h>
#include "Utilities.h"

TEST(VolumeRenderTest, TrilinearInterpolationBasic) {
    Eigen::Tensor<double, 3> data(3, 3, 3);
    int value = 1;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                data(i, j, k) = value++;
            }
        }
    }

    Eigen::VectorXd x(3);
    x << 0.0, 0.5, 1.0;
    Eigen::VectorXd y(3);
    y << 0.0, 0.5, 1.0;
    Eigen::VectorXd z(3);
    z << 0.0, 0.5, 1.0;

    Eigen::MatrixXd grid = createCustomGridMatrix(x, y, z);

    Eigen::MatrixXd qi(1, 3);
    qi << 0.5, 0.5, 0.5;

    Eigen::VectorXd results = trilinearInterpolateMultiple(data, grid, qi);

    ASSERT_NEAR(results(0), 14.0, 0.01) << "Interpolation failed at query point.";
}

TEST(VolumeRenderTest, MultipleTrilinearInterpolation) {
    Eigen::Tensor<double, 3> data(3, 3, 3);
    int value = 1;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                data(i, j, k) = value++;
            }
        }
    }

    Eigen::VectorXd x(3);
    x << 0.0, 0.5, 1.0;
    Eigen::VectorXd y(3);
    y << 0.0, 0.5, 1.0;
    Eigen::VectorXd z(3);
    z << 0.0, 0.5, 1.0;

    Eigen::MatrixXd grid = createCustomGridMatrix(x, y, z);

    Eigen::MatrixXd qi(4, 3);
    qi << 0.25, 0.25, 0.25,
          0.75, 0.75, 0.75,
          1.0,  1.0,  1.0,
          0.5,  0.5,  0.5;

    Eigen::VectorXd expected(4);
    expected << 7.5, 20.5, 27.0, 14.0;
    Eigen::VectorXd interpolatedValues = trilinearInterpolateMultiple(data, grid, qi);

    for (int i = 0; i < expected.size(); ++i) {
        ASSERT_NEAR(interpolatedValues(i), expected(i), 0.01) << "Interpolation at index " << i << " failed.";
    }
}

TEST(VolumeRenderTest, EdgeCaseInterpolation) {
    // Create a 3x3x3 tensor with linearly increasing values
    Tensor<double, 3> data(3, 3, 3);
    int value = 1;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                data(i, j, k) = value++;
            }
        }
    }

    Eigen::VectorXd x(3);
    x << 0.0, 0.5, 1.0;
    Eigen::VectorXd y(3);
    y << 0.0, 0.5, 1.0;
    Eigen::VectorXd z(3);
    z << 0.0, 0.5, 1.0;

    Eigen::MatrixXd grid = createCustomGridMatrix(x, y, z);

    MatrixXd qi(4, 3);
    qi << 0.0, 0.0, 0.0,
          1.0, 1.0, 1.0,
          0.0, 1.0, 0.0,
          1.0, 0.0, 1.0;

    VectorXd expected(4);
    expected << 1., 27.,  7., 21.;

    VectorXd interpolatedValues = trilinearInterpolateMultiple(data, grid, qi);

    for (int i = 0; i < expected.size(); ++i) {
        ASSERT_NEAR(interpolatedValues(i), expected(i), 0.01) << "Interpolation at index " << i << " failed.";
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
