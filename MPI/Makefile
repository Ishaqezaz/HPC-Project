EIGEN_INCLUDE := /Private/lib/eigen/include/eigen3
OPENCV_INCLUDE := /Private/lib/opencv/include/opencv4
OPENCV_LIB := /Private/lib/opencv/lib64
HDF5_INCLUDE := /Private/lib/hdf5/include
HDF5_LIB := /Private/lib/hdf5/lib

CXXFLAGS := -std=c++17 \
            -I$(OPENCV_INCLUDE) \
            -I$(EIGEN_INCLUDE) \
            -I$(HDF5_INCLUDE) \
            -I.

LDFLAGS := -L$(OPENCV_LIB) \
           -L$(HDF5_LIB) \
           -lhdf5_cpp -lhdf5 \
           -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_imgproc \
           -lpthread

all: volumerender

volumerender: Utilities.cpp main.cpp
        CC -O2 $(CXXFLAGS) Utilities.cpp main.cpp $(LDFLAGS) -o volumerender

clean: 
        rm -f volumerender