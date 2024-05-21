#include <iostream>
#include <Eigen/Dense>
#include <H5Cpp.h>
#include <vector>

using namespace H5;
using namespace Eigen;
using namespace std;


vector<MatrixXf> HDF5Loading(const string& filename, const string& datasetName){
    H5std_string FILE_NAME(filename);
    H5std_string DATASET_NAME(datasetName);

    vector<MatrixXf> datacube;

    try{
        H5File file(FILE_NAME, H5F_ACC_RDONLY);
        DataSet dataset = file.openDataSet(DATASET_NAME);
        DataSpace dataspace = dataset.getSpace();

        const int ndims = dataspace.getSimpleExtentNdims();
        hsize_t dims_out[3];
        dataspace.getSimpleExtentDims(dims_out, NULL);
        int Nx = dims_out[0], Ny = dims_out[1], Nz = dims_out[2];

        datacube.resize(Nz, MatrixXf(Nx, Ny));
        for(int iz = 0; iz < Nz; ++iz){
            hsize_t start[3] = {static_cast<hsize_t>(iz), 0, 0};
            hsize_t count[3] = {1, static_cast<hsize_t>(Nx), static_cast<hsize_t>(Ny)};
            hsize_t stride[3] = {1, 1, 1};
            hsize_t block[3] = {1, 1, 1};

            dataspace = dataset.getSpace();
            dataspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
            DataSpace memspace(3, count);

            dataset.read(datacube[iz].data(), PredType::NATIVE_FLOAT, memspace, dataspace);
        }

        cout << "Data loaded\n";
    }catch(const FileIException& error){
        error.printErrorStack();
    }

    return datacube;
}

Vector4d transferFunction(double x){
    double r = 1.0 * exp(-(x - 9.0) * (x - 9.0) / 1.0) + 0.1 * exp(-(x - 3.0) * (x - 3.0) / 0.1) + 0.1 * exp(-(x + 3.0) * (x + 3.0) / 0.5);
    double g = 1.0 * exp(-(x - 9.0) * (x - 9.0) / 1.0) + 1.0 * exp(-(x - 3.0) * (x - 3.0) / 0.1) + 0.1 * exp(-(x + 3.0) * (x + 3.0) / 0.5);
    double b = 0.1 * exp(-(x - 9.0) * (x - 9.0) / 1.0) + 0.1 * exp(-(x - 3.0) * (x - 3.0) / 0.1) + 1.0 * exp(-(x + 3.0) * (x + 3.0) / 0.5);
    double a = 0.6 * exp(-(x - 9.0) * (x - 9.0) / 1.0) + 0.1 * exp(-(x - 3.0) * (x - 3.0) / 0.1) + 0.01 * exp(-(x + 3.0) * (x + 3.0) / 0.5);
    
    return Vector4d(r, g, b, a);
}

double trilinearInterpolate(const std::vector<double>& data, int Nx, int Ny, int Nz, double x, double y, double z){
    int x0 = floor(x);
    int x1 = x0 + 1;
    int y0 = floor(y);
    int y1 = y0 + 1;
    int z0 = floor(z);
    int z1 = z0 + 1;

    if(x1 >= Nx) x1 = Nx - 1;
    if(y1 >= Ny) y1 = Ny - 1;
    if(z1 >= Nz) z1 = Nz - 1;

    double xd = x - x0;
    double yd = y - y0;
    double zd = z - z0;

    double c00 = data[x0 * Ny * Nz + y0 * Nz + z0] * (1 - xd) + data[x1 * Ny * Nz + y0 * Nz + z0] * xd;
    double c01 = data[x0 * Ny * Nz + y0 * Nz + z1] * (1 - xd) + data[x1 * Ny * Nz + y0 * Nz + z1] * xd;
    double c10 = data[x0 * Ny * Nz + y1 * Nz + z0] * (1 - xd) + data[x1 * Ny * Nz + y1 * Nz + z0] * xd;
    double c11 = data[x0 * Ny * Nz + y1 * Nz + z1] * (1 - xd) + data[x1 * Ny * Nz + y1 * Nz + z1] * xd;

    double c0 = c00 * (1 - yd) + c10 * yd;
    double c1 = c01 * (1 - yd) + c11 * yd;

    return c0 * (1 - zd) + c1 * zd;
}

int main(){
    try{
        auto datacube = HDF5Loading("datacube.hdf5", "density");
    }catch(const exception& e){
        cerr << "Failed to load data: " << e.what() << endl;
    }

    return 0;
}