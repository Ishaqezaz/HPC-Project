# Volume render of a 3D datacube

<div style="display: flex; justify-content: space-around;">
  <img src="Serial/images/volumerender0.png" alt="Volume Render 0" width="30%" />
  <img src="Serial/images/volumerender0.png" alt="Volume Render 1" width="30%" />
</div>

## Introduction
Volume rendering is widely used in fields like entertainment, medicine, and scientific research. This computation-intensive process demands high performance solutions for efficient handling of large datasets.

This project aimed to convert a Python based volume rendering code into a serialized C++ implementation. We then developed parallel versions using OpenMP and MPI, followed by a performance analysis to evaluate the improvements.

## Dependencies
This project was implemented with the following dependencies, which are the latest:
- Eigen: Version 3.4.0
- OpenCV: Version 4.9.0
- HDF5: Version 1.14.3
- GoogleTest: Version 1.14.0

To install the dependenices run on the project root directory:
```
chmod +x ./dependencies.sh
```

## Compiling and running the serial code
To compile the serial program and the tests:
'''
$ make 
'''

To run the serial program:
```
$ make run ARG
```
where 'ARG' is the number of images you wish to process. The default is 10 images.

To run the tests:
```
$ make run tests
```

To clean up:
```
$ make clean
```

## Compiling and running the OpenMp version.




