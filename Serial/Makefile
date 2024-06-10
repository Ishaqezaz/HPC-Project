# Fetch library installation paths
OPENCV_PATH := $(shell brew --prefix opencv)
EIGEN_PATH := $(shell brew --prefix eigen)
GTEST_PATH := $(shell brew --prefix googletest)

CXX := g++
CXXFLAGS := -O2 -std=c++17 -I$(OPENCV_PATH)/include/opencv4 -I$(EIGEN_PATH)/include/eigen3 -I$(GTEST_PATH)/include -I.

LDFLAGS := -L$(OPENCV_PATH)/lib -L$(GTEST_PATH)/lib -L/usr/local/lib -lhdf5 -lhdf5_cpp -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lgtest
FRAMEWORKS := -framework OpenCL

TARGET := volumerender
TEST_TARGET := tests

SRC := Utilities.cpp main.cpp
TEST_SRC := test/test.cpp Utilities.cpp

all: $(TARGET) $(TEST_TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) $(FRAMEWORKS) -o $@

$(TEST_TARGET): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	DYLD_LIBRARY_PATH=$(OPENCV_PATH)/lib:$(GTEST_PATH)/lib:/usr/local/lib ./$(TEST_TARGET)

runtests:
	DYLD_LIBRARY_PATH=$(OPENCV_PATH)/lib:$(GTEST_PATH)/lib:/usr/local/lib ./$(TEST_TARGET)

run: $(TARGET)
	DYLD_LIBRARY_PATH=$(OPENCV_PATH)/lib:/usr/local/lib ./$(TARGET) $(filter-out $@,$(MAKECMDGOALS))

clean:
	rm -f $(TARGET) $(TEST_TARGET)

# Phony targets
.PHONY: all runtests run clean
