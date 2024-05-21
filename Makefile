main	:	volumerender.cpp
	g++	-I /usr/local/Cellar/eigen/3.4.0_1/include/eigen3 -I /usr/local/Cellar/hdf5/1.12.1/include -L /usr/local/Cellar/hdf5/1.14.3_1/lib -lhdf5_cpp -lhdf5	-o	volumerender	 volumerender.cpp -std=c++11

run	:	volumerender.cpp
	./volumerender

del	:	volumerender.cpp
	rm volumerender
