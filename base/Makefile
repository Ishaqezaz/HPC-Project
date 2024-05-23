VCPKG_ROOT := $(HOME)/vcpkg
VCPKG_TRIPLET := x64-osx

# Paths to vcpkg includes and libraries
VCPKG_INCLUDE := $(VCPKG_ROOT)/installed/$(VCPKG_TRIPLET)/include
VCPKG_LIB := $(VCPKG_ROOT)/installed/$(VCPKG_TRIPLET)/lib

# Compiler flags
CXXFLAGS := -std=c++17 \
            -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3 \
            -I/usr/local/Cellar/hdf5/1.12.1/include \
            -I/usr/local/Cellar/opencv/4.9.0_8.reinstall/include/opencv4 \
            -I$(VCPKG_INCLUDE)

# Linker flags
LDFLAGS := -L/usr/local/Cellar/hdf5/1.14.3_1/lib \
           -L/usr/local/Cellar/opencv/4.9.0_8.reinstall/lib \
           -L/usr/local/opt/ffmpeg@6/lib \
           -L$(VCPKG_LIB) \
           -lhdf5_cpp -lhdf5 -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_imgproc \
           -lpthread

# Targets
all: volumerender runTests

volumerender: Utilities.cpp main.cpp
	g++ $(CXXFLAGS) Utilities.cpp main.cpp $(LDFLAGS) -o volumerender

runTests: test.cpp Utilities.cpp
	g++ $(CXXFLAGS) test.cpp Utilities.cpp $(LDFLAGS) -lgtest -o runTests

run: volumerender
	DYLD_LIBRARY_PATH=/usr/local/Cellar/opencv/4.9.0_8.reinstall/lib:/usr/local/opt/ffmpeg@6/lib ./volumerender

.PHONY: runtests
runtests: runTests
	DYLD_LIBRARY_PATH=/usr/local/Cellar/opencv/4.9.0_8.reinstall/lib:/usr/local/opt/ffmpeg@6/lib ./runTests

clean:
	rm -f volumerender runTests
