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

Vector4d transferFunction(double x) {
    double r = 1.0 * exp(-(x - 9.0) * (x - 9.0) / 1.0) + 0.1 * exp(-(x - 3.0) * (x - 3.0) / 0.1) + 0.1 * exp(-(x + 3.0) * (x + 3.0) / 0.5);
    double g = 1.0 * exp(-(x - 9.0) * (x - 9.0) / 1.0) + 1.0 * exp(-(x - 3.0) * (x - 3.0) / 0.1) + 0.1 * exp(-(x + 3.0) * (x + 3.0) / 0.5);
    double b = 0.1 * exp(-(x - 9.0) * (x - 9.0) / 1.0) + 0.1 * exp(-(x - 3.0) * (x - 3.0) / 0.1) + 1.0 * exp(-(x + 3.0) * (x + 3.0) / 0.5);
    double a = 0.6 * exp(-(x - 9.0) * (x - 9.0) / 1.0) + 0.1 * exp(-(x - 3.0) * (x - 3.0) / 0.1) + 0.01 * exp(-(x + 3.0) * (x + 3.0) / 0.5);
    
    return Vector4d(r, g, b, a);
}

int main(){
    try{
        auto datacube = HDF5Loading("datacube.hdf5", "density");
    }catch(const exception& e){
        cerr << "Failed to load data: " << e.what() << endl;
    }

    return 0;
}